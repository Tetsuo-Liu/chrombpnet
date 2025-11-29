import pandas as pd
import os
import scipy.stats
import numpy as np
import json
import h5py
import tensorflow as tf
import chrombpnet.training.utils.argmanager as argmanager
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.metrics as metrics
import chrombpnet.training.data_generators.initializers as initializers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
#from scipy import nanmean, nanstd

def write_predictions_h5py(output_prefix, profile, logcts, coords):
    # open h5 file for writing predictions
    output_h5_fname = "{}_predictions.h5".format(output_prefix)
    h5_file = h5py.File(output_h5_fname, "w")
    # create groups
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")

    num_examples=len(coords)

    coords_chrom_dset =  [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset =  [int(coords[i][1]) for i in range(num_examples)]
    coords_peak_dset =  [int(coords[i][3]) for i in range(num_examples)]

    dt = h5py.special_dtype(vlen=str)

    # create the "coords" group datasets
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", data=np.array(coords_chrom_dset, dtype=dt),
        dtype=dt, compression="gzip")
    coords_start_dset = coord_group.create_dataset(
        "coords_center", data=coords_center_dset, dtype=int, compression="gzip")
    coords_end_dset = coord_group.create_dataset(
        "coords_peak", data=coords_peak_dset, dtype=int, compression="gzip")

    # create the "predictions" group datasets
    profs_dset = pred_group.create_dataset(
        "profs",
        data=profile,
        dtype=float, compression="gzip")
    logcounts_dset = pred_group.create_dataset(
        "logcounts", data=logcts,
        dtype=float, compression="gzip")

    # close hdf5 file
    h5_file.close()


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
    custom_objects={"tf": tf, "multinomial_nll":losses.multinomial_nll}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_path, compile=False)
    print("got the model")
    #model.summary()
    return model

def softmax(x, temp=1):
    norm_x = x - np.mean(x,axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def run_predictions(model, data_generator, scaling_factor=1.0):
    """
    Core prediction logic extracted for reuse.
    
    Args:
        model: Loaded Keras model
        data_generator: Data generator object
        scaling_factor: Scaling factor for 2-input models (default: 1.0)
        
    Returns:
        Tuple of (true_counts, profile_predictions, counts_predictions, coordinates)
        - true_counts: True profile counts (shape: [num_regions, outputlen])
        - profile_predictions: Predicted profile probabilities (shape: [num_regions, outputlen])
        - counts_predictions: Predicted log counts (shape: [num_regions])
        - coordinates: Region coordinates (shape: [num_regions, 4])
    """
    num_batches=len(data_generator)
    profile_probs_predictions = []
    true_counts = []
    counts_sum_predictions = []
    true_counts_sum = []
    coordinates = []

    for idx in range(num_batches):
        if idx%100==0:
            print(str(idx)+'/'+str(num_batches))
        
        batch_data = data_generator[idx]
        
        # Handle different generator types
        # CelltypeGenerator returns 3-element tuple (inputs, targets, sample_weights)
        # and provides coordinates via get_coords() method
        if hasattr(data_generator, 'get_coords'):
            # CelltypeGenerator: use get_coords() method
            X, y = batch_data[:2]  # inputs, targets
            coords = data_generator.get_coords(idx)
        elif len(batch_data) == 3:
            # Standard generators: (inputs, targets, coords)
            X, y, coords = batch_data
        else:
            # Fallback: (inputs, targets) only
            X, y = batch_data
            coords = None

        # Check if model expects 2 inputs (dynamic scaling model)
        # and if X is a tuple (2 inputs: sequence + scaling_factor)
        is_2input_model = len(model.inputs) == 2
        is_2input_data = isinstance(X, tuple) and len(X) == 2
        
        # Check if multitask model (multiple outputs or dictionary targets)
        is_multitask_model = len(model.outputs) > 2
        is_multitask_targets = isinstance(y, dict)
        
        # Handle 2-input model case
        if is_2input_model:
            if is_2input_data:
                # Model expects 2 inputs and data provides 2 inputs: use as-is
                preds = model.predict_on_batch(X)
            else:
                # Model expects 2 inputs but data provides 1 input: use provided scaling_factor
                batch_size = X.shape[0] if hasattr(X, 'shape') else len(X)
                scaling_factors = np.full((batch_size, 1), scaling_factor, dtype=np.float32)
                preds = model.predict_on_batch([X, scaling_factors])
        else:
            # 1-input model: handle both tuple and single input
            if is_2input_data:
                # Model expects 1 input but data provides 2 inputs: use only sequence input
                X = X[0]  # Extract sequence input from tuple
            preds = model.predict_on_batch(X)

        # Handle multitask model predictions
        if is_multitask_model and is_multitask_targets:
            # Get celltype indices for this batch
            batch_celltype_indices = None
            if hasattr(data_generator, 'get_celltype_indices'):
                batch_celltype_indices = data_generator.get_celltype_indices(idx)
            
            # Extract celltype names from model output names (more reliable than target keys)
            # Model outputs are ordered as: [profile_1, ..., profile_N, count_1, ..., count_N]
            # where N is the number of celltypes
            celltype_names = []
            celltype_to_profile_idx = {}
            celltype_to_count_idx = {}
            
            # Get celltype order from model output names
            num_celltypes = len(model.outputs) // 2
            for i, output_layer in enumerate(model.outputs):
                output_name = output_layer.name
                if output_name.startswith('logits_profile_'):
                    celltype_name = output_name.replace('logits_profile_', '')
                    if celltype_name not in celltype_names:
                        celltype_names.append(celltype_name)
                    celltype_to_profile_idx[celltype_name] = i
                elif output_name.startswith('logcount_'):
                    celltype_name = output_name.replace('logcount_', '')
                    celltype_to_count_idx[celltype_name] = i
            
            # Get index-to-celltype mapping from generator if available
            index_to_celltype = None
            if hasattr(data_generator, 'index_to_celltype'):
                index_to_celltype = data_generator.index_to_celltype
            
            # For each sample, find the correct celltype and extract corresponding predictions
            batch_size = len(list(y.values())[0])
            
            # Extract predictions for each sample based on its celltype
            batch_true_counts = []
            batch_profile_preds = []
            batch_true_counts_sum = []
            batch_counts_preds = []
            
            for i in range(batch_size):
                # Find the correct celltype for this sample
                sample_celltype_name = None
                if batch_celltype_indices is not None and index_to_celltype is not None:
                    celltype_idx = batch_celltype_indices[i]
                    if celltype_idx >= 0 and celltype_idx in index_to_celltype:
                        sample_celltype_name = index_to_celltype[celltype_idx]
                
                # If celltype not found, try to find non-zero target
                if sample_celltype_name is None:
                    for celltype_name in celltype_names:
                        profile_key = f'logits_profile_{celltype_name}'
                        if profile_key in y and np.any(y[profile_key][i] != 0):
                            sample_celltype_name = celltype_name
                            break
                
                # Default to first celltype if still not found
                if sample_celltype_name is None:
                    sample_celltype_name = celltype_names[0]
                
                # Extract true labels for this celltype
                profile_key = f'logits_profile_{sample_celltype_name}'
                count_key = f'logcount_{sample_celltype_name}'
                
                batch_true_counts.append(y[profile_key][i])
                batch_true_counts_sum.append(y[count_key][i, 0])
                
                # Extract predictions for this celltype
                profile_idx = celltype_to_profile_idx[sample_celltype_name]
                count_idx = celltype_to_count_idx[sample_celltype_name]
                
                batch_profile_preds.append(preds[profile_idx][i])
                batch_counts_preds.append(preds[count_idx][i, 0])
            
            # Convert to arrays and extend lists
            true_counts.extend(batch_true_counts)
            profile_probs_predictions.extend(softmax(np.array(batch_profile_preds)))
            true_counts_sum.extend(batch_true_counts_sum)
            counts_sum_predictions.extend(batch_counts_preds)
        else:
            # Standard single-task model
            # get counts predictions
            true_counts.extend(y[0])
            profile_probs_predictions.extend(softmax(preds[0]))

            # get profile predictions
            true_counts_sum.extend(y[1][:,0])
            counts_sum_predictions.extend(preds[1][:,0])
        
        if coords is not None:
            coordinates.extend(coords)

    return np.array(true_counts), np.array(profile_probs_predictions), np.array(true_counts_sum), np.array(counts_sum_predictions), np.array(coordinates)


def predict_on_batch_wrapper(model, test_generator):
    """
    Backward compatibility wrapper for run_predictions.
    Uses default scaling_factor=1.0 for 2-input models.
    """
    return run_predictions(model, test_generator, scaling_factor=1.0)


def main(args):


    metrics_dictionary = {"counts_metrics":{}, "profile_metrics":{}}

    # get model architecture to load - can load .hdf5 and .weights/.arch
    model=load_model_wrapper(args)


    test_generator = initializers.initialize_generators(args, mode="test", parameters=None, return_coords=True)
    true_counts, profile_probs_predictions, true_counts_sum, counts_sum_predictions, coordinates = run_predictions(model, test_generator, scaling_factor=1.0)


    # generate prediction on test set and store metrics
    write_predictions_h5py(args.output_prefix, profile_probs_predictions, counts_sum_predictions, coordinates)

    # store regions, their predictions and corresponding pointwise metrics
    mnll_pw, mnll_norm, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, mnll_rnd, mnll_rnd_norm =  metrics.profile_metrics(true_counts,profile_probs_predictions)

    # including both metrics    
    if args.peaks != "None" and args.nonpeaks != "None":
        spearman_cor, pearson_cor, mse = metrics.counts_metrics(true_counts_sum, counts_sum_predictions,args.output_prefix+"_peaks_and_nonpeaks", "Both peaks and non peaks")
        metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"] = {}
        metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["mse"] = mse

        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"] = {}
        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

        metrics.plot_histogram(jsd_pw, jsd_rnd, args.output_prefix+"_peaks_and_nonpeaks", "Both peaks and non peaks")


    # including only nonpeak metrics
    if args.nonpeaks != "None":
        non_peaks_idx = coordinates[:,3] == '0'
        spearman_cor, pearson_cor, mse = metrics.counts_metrics(true_counts_sum[non_peaks_idx], counts_sum_predictions[non_peaks_idx],args.output_prefix+"_only_nonpeaks", "Only non peaks")
        metrics_dictionary["counts_metrics"]["nonpeaks"] = {}
        metrics_dictionary["counts_metrics"]["nonpeaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["mse"] = mse

        metrics_dictionary["profile_metrics"]["nonpeaks"] = {}
        metrics_dictionary["profile_metrics"]["nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw[non_peaks_idx])        
        metrics_dictionary["profile_metrics"]["nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm[non_peaks_idx])

        metrics.plot_histogram(jsd_pw[non_peaks_idx], jsd_rnd[non_peaks_idx], args.output_prefix+"_only_nonpeaks", "Only non peaks")

    # including only peak metrics
    if args.peaks != "None":
        peaks_idx = coordinates[:,3] == '1'
        spearman_cor, pearson_cor, mse = metrics.counts_metrics(true_counts_sum[peaks_idx], counts_sum_predictions[peaks_idx],args.output_prefix+"_only_peaks", "Only peaks")
        metrics_dictionary["counts_metrics"]["peaks"] = {}
        metrics_dictionary["counts_metrics"]["peaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["peaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["peaks"]["mse"] = mse

        metrics_dictionary["profile_metrics"]["peaks"] = {}
        metrics_dictionary["profile_metrics"]["peaks"]["median_jsd"] = np.nanmedian(jsd_pw[peaks_idx])        
        metrics_dictionary["profile_metrics"]["peaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm[peaks_idx])
        metrics.plot_histogram(jsd_pw[peaks_idx], jsd_rnd[peaks_idx], args.output_prefix+"_only_peaks", "Only peaks")

        #ofile = open(args.output_prefix+"_pearson_cor.txt","w")
        #ofile.write(str(round(pearson_cor,2)))
        #ofile.close()

        #ofile = open(args.output_prefix+"_norm_jsd.txt","w")
        #ofile.write(str(round(metrics_dictionary["profile_metrics"]["peaks"]["median_norm_jsd"],2)))
        #ofile.close()
    # store dictionary
    with open(args.output_prefix+'_metrics.json', 'w') as fp:
            json.dump(metrics_dictionary, fp,  indent=4)

if __name__=="__main__":
    # read arguments
    args=argmanager.fetch_predict_args()
    main(args)

