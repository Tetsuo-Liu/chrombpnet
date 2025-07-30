import argparse
import pyfaidx
import pyBigWig
import pandas as pd
import numpy as np
import os
import json
import chrombpnet.helpers.hyperparameters.param_utils as param_utils
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools

def parse_data_args():
    parser=argparse.ArgumentParser(description="find hyper-parameters for chrombpnet defined in src/training/models/chrombpnet_with_bias_model.py")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-i", "--bigwig", type=str, required=True, help="Bigwig of tn5 insertions. Ensure it is +4/-4 shifted")
    parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    parser.add_argument("-b", "--bias-threshold-factor", type=float, default=0.5, help="A threshold is applied on maximum count of non-peak region for training bias model, which is set as this threshold x min(count over peak regions)")
    parser.add_argument("-oth", "--outlier-threshold", type=float, default=0.9999, help="threshold to use to filter outlies")
    parser.add_argument("-j", "--max-jitter", type=int, default=50, help="Maximum jitter applied on either side of region (default 500 for chrombpnet model)")
    parser.add_argument("-fl", "--chr-fold-path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    parser.add_argument("--jobs", type=int, default=None, help="Number of parallel jobs to use")
    return parser

def parse_model_args(parser):
    # arguments here defined the following model - src/training/models/chrombpnet_with_bias_model.py
    parser.add_argument("-il", "--inputlen", type=int, help="Sequence input length")
    parser.add_argument("-ol", "--outputlen", type=int, help="Prediction output length")
    parser.add_argument("-fil", "--filters", type=int, default=128, help="Number of filters to use in chrombpnet mode")
    parser.add_argument("-dil", "--n-dilation-layers", type=int, default=4, help="Number of dilation layers to use in chrombpnet model")
    parser.add_argument("-op", "--output-prefix", required=True, help="output prefix for storing hyper-param TSV for chrombpnet")
    args = parser.parse_args()
    return args

def process_counts_worker(worker_args):
    """
    Worker function to process counts for a single chromosome.
    """
    regions_df, bigwig_path, genome_path, inputlen, outputlen = worker_args
    
    with pyBigWig.open(bigwig_path) as bw:
        genome = pyfaidx.Fasta(genome_path)
        counts, _ = param_utils.get_seqs_cts(genome, bw, regions_df, inputlen, outputlen)
    
    return counts

def main(args):    
    # read the fold information
    splits_dict=json.load(open(args.chr_fold_path))
    chroms_to_keep=splits_dict["train"]+splits_dict["valid"]
    test_chroms_to_keep=splits_dict["test"]
    print("evaluating hyperparameters on the following chromosomes",chroms_to_keep)

    # read peaks and non peaks    
    in_peaks =  pd.read_csv(args.peaks, sep='\t', header=None, names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])
    in_nonpeaks =  pd.read_csv(args.nonpeaks, sep='\t', header=None, names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])

    assert(in_peaks.shape[0] != 0)
    assert(in_nonpeaks.shape[0] !=0)
    assert(args.inputlen >= args.outputlen)

    # get train/valid peaks and test peaks seperately
    peaks = in_peaks[in_peaks["chr"].isin(chroms_to_keep)]
    test_peaks = in_peaks[in_peaks["chr"].isin(test_chroms_to_keep)]
    nonpeaks = in_nonpeaks[in_nonpeaks["chr"].isin(chroms_to_keep)]
    test_nonpeaks = in_nonpeaks[in_nonpeaks["chr"].isin(test_chroms_to_keep)]

    # Edge filtering
    with pyBigWig.open(args.bigwig) as bw:
        nonpeaks = param_utils.filter_edge_regions(nonpeaks, bw, args.inputlen, peaks_bool=0)
        test_nonpeaks = param_utils.filter_edge_regions(test_nonpeaks, bw, args.inputlen, peaks_bool=0)
        peaks = param_utils.filter_edge_regions(peaks, bw, args.inputlen, peaks_bool=1)
        test_peaks = param_utils.filter_edge_regions(test_peaks, bw, args.inputlen, peaks_bool=1)

    # Parallelized count retrieval
    if args.jobs is None:
        n_jobs = min(cpu_count(), 8) # Limit to 8 cores by default
    else:
        n_jobs = args.jobs
    print(f"Using {n_jobs} parallel jobs for count retrieval.")

    with Pool(processes=n_jobs) as pool:
        # Prepare arguments for peak counts
        peak_worker_args = [(df, args.bigwig, args.genome, args.inputlen, args.outputlen) for _, df in peaks.groupby('chr')]
        print(f"Processing {len(peak_worker_args)} chromosomes for peak counts...")
        peak_results = list(tqdm(pool.imap(process_counts_worker, peak_worker_args), total=len(peak_worker_args), desc="Peak Counts"))
        peak_cnts = np.concatenate(peak_results)

        # Prepare arguments for non-peak counts
        nonpeak_worker_args = [(df, args.bigwig, args.genome, args.inputlen, args.outputlen) for _, df in nonpeaks.groupby('chr')]
        print(f"Processing {len(nonpeak_worker_args)} chromosomes for non-peak counts...")
        nonpeak_results = list(tqdm(pool.imap(process_counts_worker, nonpeak_worker_args), total=len(nonpeak_worker_args), desc="Non-peak Counts"))
        nonpeak_cnts = np.concatenate(nonpeak_results)

    assert(len(peak_cnts) == peaks.shape[0])
    assert(len(nonpeak_cnts) == nonpeaks.shape[0])

    # Count-based filtering
    final_cnts = nonpeak_cnts
    counts_threshold = np.quantile(peak_cnts,0.01)*args.bias_threshold_factor
    assert(counts_threshold > 0)
   
    final_cnts = final_cnts[final_cnts < counts_threshold]
    print("Upper bound counts cut-off for bias model training: ", counts_threshold)
    print("Number of nonpeaks after the upper-bount cut-off: ", len(final_cnts))
    assert(len(final_cnts) > 0)

    # Outlier filtering
    upper_thresh = np.quantile(final_cnts, args.outlier_threshold)
    lower_thresh = np.quantile(final_cnts, 1-args.outlier_threshold)
    nonpeaks = nonpeaks[(nonpeak_cnts<upper_thresh) & (nonpeak_cnts>lower_thresh)]
    print("Number of nonpeaks after applying upper-bound cut-off and removing outliers : ", nonpeaks.shape[0])

    # combine train valid and test peak set and store them in a new file
    all_nonpeaks = pd.concat([nonpeaks, test_nonpeaks])
    all_nonpeaks.to_csv(f"{args.output_prefix}filtered.bias_nonpeaks.bed", sep="\t", header=False, index=False)
    all_peaks = pd.concat([peaks, test_peaks])
    all_peaks.to_csv(f"{args.output_prefix}filtered.bias_peaks.bed", sep="\t", header=False, index=False)

    # find counts loss weight for model training
    counts_loss_weight = np.median(final_cnts[(final_cnts < upper_thresh) & (final_cnts>lower_thresh)])/10
    print("counts_loss_weight:", counts_loss_weight)
    assert(counts_loss_weight != 0)

    if counts_loss_weight < 1.0:
        counts_loss_weight = 1.0
        print("WARNING: you are training on low-read depth data")

    # store the parameters being used
    with open(f"{args.output_prefix}bias_data_params.tsv", "w") as f:
        f.write(f"counts_sum_min_thresh\t{round(lower_thresh,2)}\n")
        f.write(f"counts_sum_max_thresh\t{round(upper_thresh,2)}\n")
        f.write(f"trainings_pts_post_thresh\t{sum((final_cnts<upper_thresh) & (final_cnts>lower_thresh))}\n")

    with open(f"{args.output_prefix}bias_model_params.tsv", "w") as f:
        f.write(f"counts_loss_weight\t{round(counts_loss_weight,2)}\n")
        f.write(f"filters\t{args.filters}\n")
        f.write(f"n_dil_layers\t{args.n_dilation_layers}\n")
        f.write(f"inputlen\t{args.inputlen}\n")
        f.write(f"outputlen\t{args.outputlen}\n")
        f.write(f"max_jitter\t{args.max_jitter}\n")
        f.write(f"chr_fold_path\t{args.chr_fold_path}\n")
        f.write("negative_sampling_ratio\t1.0\n")

if __name__=="__main__":
    parser = parse_data_args()
    args = parse_model_args(parser)
    main(args)
