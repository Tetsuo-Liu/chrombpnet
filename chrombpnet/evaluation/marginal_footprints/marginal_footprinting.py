import pyBigWig
import pandas as pd
import numpy as np
import deepdish as dd
import os
import pyfaidx
import random
import pickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import json
import chrombpnet.training.utils.losses as losses
from chrombpnet.training.utils.data_utils import get_seq as get_seq
import chrombpnet.training.utils.one_hot as one_hot
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from chrombpnet.training.utils.bed_utils import read_bed_with_summit


NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]
PWM_SCHEMA = ["MOTIF_NAME", "MOTIF_PWM_FWD"]

def load_model_wrapper(args_or_path):
    """
    Load model with backward compatibility.
    
    Args:
        args_or_path: Either args object with .model_h5 attribute, or direct path string
        
    Returns:
        Loaded Keras model
    """
    # Handle both args object and direct path string
    if isinstance(args_or_path, str):
        model_path = args_or_path
    else:
        model_path = args_or_path.model_h5
    
    # read .h5 model
    custom_objects={"multinomial_nll":losses.multinomial_nll, "tf": tf}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_path, compile=False)
    print("got the model")
    model.summary()
    return model


def fetch_footprinting_args():
    parser=argparse.ArgumentParser(description="get marginal footprinting for given model and given motifs")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-r", "--regions", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-fl", "--chr_fold_path", type=str, required=True, help="Path to file containing chromosome splits; we will only use the test chromosomes")
    parser.add_argument("-m", "--model_h5", type=str, required=True, help="Path to trained model, can be both bias or chrombpnet model")
    parser.add_argument("-bs", "--batch_size", type=int, default="64", help="input batch size for the model")
    parser.add_argument("-o", "--output_prefix", type=str, required=True, help="Output prefix")
    parser.add_argument("-pwm_f", "--motifs_to_pwm", type=str, required=True, help="Path to a TSV file containing motifs in first column and motif string to use for footprinting in second column")    
    parser.add_argument("--ylim",default=None,type=tuple, required=False,help="lower and upper y-limits for plotting the motif footprint, in the form of a tuple i.e. \
    (0,0.8). If this is set to None, ylim will be autodetermined.")
    parser.add_argument("--scaling-factor", type=float, default=1.0, help="Scaling factor for 2-input models (default: 1.0, used when model has 2 inputs)")
    
    args = parser.parse_args()
    return args

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def get_footprint_for_motif(seqs, motif, model, inputlen, batch_size, scaling_factor=1.0):
    '''
    Returns footprints for a given motif. Motif is inserted in both the actual sequence and reverse complemented version.
    seqs input is already assumed to be one-hot encoded. motif is in sequence format.
    '''
    midpoint=inputlen//2

    w_mot_seqs = seqs.copy()
    w_mot_seqs[:, midpoint-len(motif)//2:midpoint-len(motif)//2+len(motif)] = one_hot.dna_to_one_hot([motif])

    # Check if model has 2 inputs (dynamic scaling model)
    is_2input_model = len(model.inputs) == 2
    
    # midpoint of motif is the midpoint of sequence
    if is_2input_model:
        scaling_factors = np.full((len(w_mot_seqs), 1), scaling_factor, dtype=np.float32)
        pred_output=model.predict([w_mot_seqs, scaling_factors], batch_size=batch_size, verbose=True)
    else:
        pred_output=model.predict(w_mot_seqs, batch_size=batch_size, verbose=True)
    footprint_for_motif_fwd = softmax(pred_output[0])*(np.exp(pred_output[1])-1)

    # reverse complement the sequence
    w_mot_seqs_revc = w_mot_seqs[:, ::-1, ::-1]
    if is_2input_model:
        scaling_factors_revc = np.full((len(w_mot_seqs_revc), 1), scaling_factor, dtype=np.float32)
        pred_output_rev=model.predict([w_mot_seqs_revc, scaling_factors_revc], batch_size=batch_size, verbose=True)
    else:
        pred_output_rev=model.predict(w_mot_seqs_revc, batch_size=batch_size, verbose=True)
    footprint_for_motif_rev = softmax(pred_output_rev[0])*(np.exp(pred_output_rev[1])-1)

    # add fwd sequence predictions and reverse sesquence predictions (not we flip the rev predictions)
    counts_for_motif = np.exp(pred_output_rev[1]) - 1 + np.exp(pred_output[1]) - 1
    footprint_for_motif_tot = footprint_for_motif_fwd+footprint_for_motif_rev[:,::-1]
    footprint_for_motif =  footprint_for_motif_tot / footprint_for_motif_tot.sum(axis=1)[:, np.newaxis]

    return footprint_for_motif.mean(0), counts_for_motif.mean(0)

def run_footprinting(model, background_seqs, motifs_df, inputlen, outputlen, batch_size, scaling_factor=1.0, output_prefix=None, ylim=None):
    """
    Core footprinting logic extracted for reuse.
    
    Args:
        model: Loaded Keras model
        background_seqs: Background sequences (one-hot encoded, shape: [num_seqs, inputlen, 4])
        motifs_df: DataFrame with columns 'MOTIF_NAME' and 'MOTIF_PWM_FWD'
        inputlen: Model input length
        outputlen: Model output length
        batch_size: Batch size for prediction
        scaling_factor: Scaling factor for 2-input models (default: 1.0)
        output_prefix: Optional output prefix for saving plots
        ylim: Optional tuple for y-axis limits
        
    Returns:
        Dictionary with motif names as keys and [footprint, counts] as values
    """
    footprints_at_motifs = {}
    avg_response_at_tn5 = []
    
    # Control motif (empty sequence)
    motif = "control"
    motif_to_insert_fwd = ""
    print("inserting motif: ", motif)
    print(motif_to_insert_fwd)
    motif_footprint, motif_counts = get_footprint_for_motif(
        background_seqs, motif_to_insert_fwd, model, inputlen, batch_size, scaling_factor
    )
    footprints_at_motifs[motif] = [motif_footprint, motif_counts]
    
    if output_prefix is not None:
        plt.figure()
        plt.plot(range(200), motif_footprint[outputlen//2-100:outputlen//2+100])
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel("200bp around motif insertion", fontsize=11)
        plt.ylabel("Probability", fontsize=11)
        plt.xticks(ticks=[0,100,200], labels=[-100,0,100])
        plt.tight_layout()
        plt.savefig(f"{output_prefix}.{motif}.footprint.png")
        plt.close()
    
    # Process each motif from the dataframe
    for index, row in motifs_df.iterrows():
        motif = row["MOTIF_NAME"]
        motif_to_insert_fwd = row["MOTIF_PWM_FWD"]
        print("inserting motif: ", motif)
        print(motif_to_insert_fwd)
        motif_footprint, motif_counts = get_footprint_for_motif(
            background_seqs, motif_to_insert_fwd, model, inputlen, batch_size, scaling_factor
        )
        footprints_at_motifs[motif] = [motif_footprint, motif_counts]
        
        # Track Tn5/DNase motifs for bias correction validation
        if ("tn5" in motif.lower()) or ("dnase" in motif.lower()):
            avg_response_at_tn5.append(np.round(np.max(motif_footprint), 3))
        
        if output_prefix is not None:
            plt.figure()
            plt.plot(range(200), motif_footprint[outputlen//2-100:outputlen//2+100])
            if ylim is not None:
                plt.ylim(ylim)
            plt.xlabel("200bp around motif insertion", fontsize=11)
            plt.ylabel("Probability", fontsize=11)
            plt.xticks(ticks=[0,100,200], labels=[-100,0,100])
            plt.tight_layout()
            plt.savefig(f"{output_prefix}.{motif}.footprint.png")
            plt.close()
    
    # Save max bias response if Tn5 motifs were processed
    if len(avg_response_at_tn5) > 0 and output_prefix is not None:
        if np.all(np.array(avg_response_at_tn5) < 0.003):
            with open(f"{output_prefix}_max_bias_response.txt", "w") as ofile:
                ofile.write("corrected_" + str(round(np.mean(avg_response_at_tn5), 3)) + "_" + "/".join(list(map(str, avg_response_at_tn5))))
        else:
            with open(f"{output_prefix}_max_bias_response.txt", "w") as ofile:
                ofile.write("uncorrected_" + str(round(np.mean(avg_response_at_tn5), 3)) + "_" + "/".join(list(map(str, avg_response_at_tn5))))
    
    return footprints_at_motifs

def main(args):

	pwm_df = pd.read_csv(args.motifs_to_pwm, sep='\t',names=PWM_SCHEMA)
	print(pwm_df.head())
	genome_fasta = pyfaidx.Fasta(args.genome)

	model=load_model_wrapper(args)
	# Handle both 1-input and 2-input models
	if len(model.inputs) == 2:
		inputlen = model.inputs[0].shape[1]
	else:
		inputlen = model.input_shape[1]
	outputlen = model.output_shape[0][1] 
	print("inferred model inputlen: ", inputlen)
	print("inferred model outputlen: ", outputlen)
	
	# Check if model has 2 inputs
	is_2input_model = len(model.inputs) == 2
	if is_2input_model:
		print(f"Detected 2-input model, using scaling_factor={args.scaling_factor}")

	splits_dict = json.load(open(args.chr_fold_path))
	chroms_to_keep = set(splits_dict["test"])

	regions_df = read_bed_with_summit(args.regions)
	regions_subsample = regions_df[(regions_df["chr"].isin(chroms_to_keep))]
	regions_seqs = get_seq(regions_subsample, genome_fasta, inputlen)

	# Use refactored run_footprinting function
	footprints_at_motifs = run_footprinting(
		model=model,
		background_seqs=regions_seqs,
		motifs_df=pwm_df,
		inputlen=inputlen,
		outputlen=outputlen,
		batch_size=args.batch_size,
		scaling_factor=args.scaling_factor,
		output_prefix=args.output_prefix,
		ylim=args.ylim
	)

	print("Saving marginal footprints")
	dd.io.save("{}_footprints.h5".format(args.output_prefix),
		footprints_at_motifs,
		compression='blosc')


if __name__ == '__main__':
    args=fetch_footprinting_args()
    main(args)
