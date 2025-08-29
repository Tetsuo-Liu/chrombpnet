# Adapted from chrombpnet-lite

import deepdish as dd
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import shap
import pyfaidx
import shutil
import errno
import os
import argparse
import chrombpnet.evaluation.interpret.shap_utils as shap_utils
import chrombpnet.evaluation.interpret.input_utils as input_utils
from chrombpnet.training.utils.bed_utils import read_bed_with_summit
from tqdm import tqdm
import logging
import gc
import psutil

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]

# disable eager execution so shap deep explainer wont break
tf.compat.v1.disable_eager_execution()

def fetch_interpret_args():
    parser = argparse.ArgumentParser(description="get sequence contribution scores for the model")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-r", "--regions", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-m", "--model_h5", type=str, required=True, help="Path to trained model, can be both bias or chrombpnet model")
    parser.add_argument("-o", "--output-prefix", type=str, required=True, help="Output prefix")
    parser.add_argument("-d", "--debug_chr", nargs="+", type=str, default=None, help="Run for specific chromosomes only (e.g. chr1 chr2) for debugging")
    parser.add_argument("-p", "--profile_or_counts", nargs="+", type=str, default=["counts", "profile"], choices=["counts", "profile"],
                        help="use either counts or profile or both for running shap")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for memory-efficient processing (auto-detect if not specified)")

    args = parser.parse_args()
    return args


def generate_shap_dict(seqs, scores):
    assert(seqs.shape==scores.shape)
    assert(seqs.shape[2]==4)

    # construct a dictionary for the raw shap scores and the
    # the projected shap scores
    # MODISCO workflow expects one hot sequences with shape (None,4,inputlen)
    d = {
            'raw': {'seq': np.transpose(seqs, (0, 2, 1)).astype(np.int8)},
            'shap': {'seq': np.transpose(scores, (0, 2, 1)).astype(np.float16)},
            'projected_shap': {'seq': np.transpose(seqs*scores, (0, 2, 1)).astype(np.float16)}
        }

    return d

def get_optimal_chunk_size(total_sequences, available_memory_gb=None):
    """Calculate optimal chunk size based on available memory and sequence count."""
    if available_memory_gb is None:
        # Use 60% of available memory, leave 40% for system and other processes
        total_memory = psutil.virtual_memory().total / (1024 ** 3)
        available_memory_gb = total_memory * 0.6
    
    # Conservative estimate: each sequence uses ~1MB during SHAP processing
    memory_per_sequence_gb = 1e-3  # 1MB in GB
    memory_per_sequence_gb *= 5  # Safety factor for SHAP computation overhead
    
    # Calculate chunk size
    chunk_size = int(available_memory_gb / memory_per_sequence_gb)
    
    # Apply reasonable bounds
    chunk_size = max(100, min(chunk_size, total_sequences))  # At least 100, at most total
    chunk_size = min(chunk_size, 2000)  # Cap at 2000 for reasonable processing time per chunk
    
    logging.info(f"Calculated optimal chunk size: {chunk_size} "
                f"(total sequences: {total_sequences}, available memory: {available_memory_gb:.1f}GB)")
    
    return chunk_size

def combine_chunk_results(chunk_results):
    """Combine results from multiple chunks into a single dictionary."""
    if not chunk_results:
        return {}
    
    combined = {'raw': {'seq': []}, 'shap': {'seq': []}, 'projected_shap': {'seq': []}}
    
    for chunk_dict in chunk_results:
        combined['raw']['seq'].append(chunk_dict['raw']['seq'])
        combined['shap']['seq'].append(chunk_dict['shap']['seq'])
        combined['projected_shap']['seq'].append(chunk_dict['projected_shap']['seq'])
    
    # Concatenate arrays
    combined['raw']['seq'] = np.concatenate(combined['raw']['seq'], axis=0)
    combined['shap']['seq'] = np.concatenate(combined['shap']['seq'], axis=0)
    combined['projected_shap']['seq'] = np.concatenate(combined['projected_shap']['seq'], axis=0)
    
    return combined

def interpret_chunked(model, seqs, output_prefix, profile_or_counts, chunk_size=None):
    """
    Perform interpretation with chunked processing for memory efficiency.
    """
    total_sequences = len(seqs)
    print(f"Seqs dimension : {seqs.shape}")
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(total_sequences)
    
    logging.info(f"Starting chunked interpretation of {total_sequences} sequences with chunk size {chunk_size}")
    
    # Initialize explainers once
    explainers = {}
    
    if "counts" in profile_or_counts:
        logging.info("Initializing counts explainer...")
        explainers['counts'] = shap.explainers.deep.TFDeepExplainer(
            (model.input, tf.reduce_sum(model.outputs[1], axis=-1)),
            shap_utils.shuffle_several_times,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)
    
    if "profile" in profile_or_counts:
        logging.info("Initializing profile explainer...")
        weightedsum_meannormed_logits = shap_utils.get_weightedsum_meannormed_logits(model)
        explainers['profile'] = shap.explainers.deep.TFDeepExplainer(
            (model.input, weightedsum_meannormed_logits),
            shap_utils.shuffle_several_times,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)
    
    # Process in chunks
    all_results = {'counts': [], 'profile': []}
    
    num_chunks = (total_sequences + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_sequences)
        chunk_seqs = seqs[start_idx:end_idx]
        
        chunk_size_actual = len(chunk_seqs)
        logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} with {chunk_size_actual} sequences")
        
        # Process counts
        if "counts" in profile_or_counts:
            print(f"Generating 'counts' shap scores for chunk {chunk_idx + 1}/{num_chunks}")
            try:
                counts_shap_scores = explainers['counts'].shap_values(
                    chunk_seqs, progress_message=max(1, chunk_size_actual // 20))
                
                counts_dict = generate_shap_dict(chunk_seqs, counts_shap_scores)
                all_results['counts'].append(counts_dict)
                
                del counts_shap_scores
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error computing counts SHAP for chunk {chunk_idx + 1}: {e}")
                continue
        
        # Process profile
        if "profile" in profile_or_counts:
            print(f"Generating 'profile' shap scores for chunk {chunk_idx + 1}/{num_chunks}")
            try:
                profile_shap_scores = explainers['profile'].shap_values(
                    chunk_seqs, progress_message=max(1, chunk_size_actual // 20))
                
                profile_dict = generate_shap_dict(chunk_seqs, profile_shap_scores)
                all_results['profile'].append(profile_dict)
                
                del profile_shap_scores
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error computing profile SHAP for chunk {chunk_idx + 1}: {e}")
                continue
        
        # Memory status
        memory_usage = psutil.virtual_memory().percent
        logging.info(f"Chunk {chunk_idx + 1} complete. Memory usage: {memory_usage:.1f}%")
    
    # Combine and save results
    logging.info("Combining and saving results...")
    
    if "counts" in profile_or_counts and all_results['counts']:
        print("Saving combined 'counts' scores")
        combined_counts = combine_chunk_results(all_results['counts'])
        dd.io.save(f"{output_prefix}.counts_scores.h5", combined_counts, compression='blosc')
        logging.info(f"Counts scores saved to {output_prefix}.counts_scores.h5")
    
    if "profile" in profile_or_counts and all_results['profile']:
        print("Saving combined 'profile' scores")
        combined_profile = combine_chunk_results(all_results['profile'])
        dd.io.save(f"{output_prefix}.profile_scores.h5", combined_profile, compression='blosc')
        logging.info(f"Profile scores saved to {output_prefix}.profile_scores.h5")

def interpret(model, seqs, output_prefix, profile_or_counts, chunk_size=None):
    """
    Wrapper function to maintain backward compatibility.
    Uses chunked processing for large datasets.
    """
    # Use chunked processing for datasets larger than 1000 sequences
    if len(seqs) > 1000:
        logging.info(f"Large dataset detected ({len(seqs)} sequences), using chunked processing")
        return interpret_chunked(model, seqs, output_prefix, profile_or_counts, chunk_size)
    
    # Original implementation for smaller datasets
    print("Seqs dimension : {}".format(seqs.shape))

    if "counts" in profile_or_counts:
        profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
            (model.input, tf.reduce_sum(model.outputs[1], axis=-1)),
            shap_utils.shuffle_several_times,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        print("Generating 'counts' shap scores")
        counts_shap_scores = profile_model_counts_explainer.shap_values(
            seqs, progress_message=100)

        counts_scores_dict = generate_shap_dict(seqs, counts_shap_scores)

        print("Saving 'counts' scores")
        dd.io.save("{}.counts_scores.h5".format(output_prefix),
                    counts_scores_dict,
                    compression='blosc')

        del counts_shap_scores, counts_scores_dict

    if "profile" in profile_or_counts:
        weightedsum_meannormed_logits = shap_utils.get_weightedsum_meannormed_logits(model)
        profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
            (model.input, weightedsum_meannormed_logits),
            shap_utils.shuffle_several_times,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        print("Generating 'profile' shap scores")
        profile_shap_scores = profile_model_profile_explainer.shap_values(
            seqs, progress_message=100)

        profile_scores_dict = generate_shap_dict(seqs, profile_shap_scores)

        print("Saving 'profile' scores")
        dd.io.save("{}.profile_scores.h5".format(output_prefix),
                    profile_scores_dict,
                    compression='blosc')


def interpret_regions(genome_path, regions_path, model_h5_path, output_prefix, 
                     profile_or_counts=["counts", "profile"], debug_chr=None, chunk_size=None):
    """
    Core interpretation logic extracted from main().
    
    Args:
        genome_path: Path to genome FASTA file
        regions_path: Path to regions BED file
        model_h5_path: Path to trained model H5 file
        output_prefix: Output file prefix
        profile_or_counts: List of analysis types to run
        debug_chr: Optional list of chromosomes for debugging
        chunk_size: Optional chunk size for memory-efficient processing
        
    Returns:
        dict: Information about processed regions and output files
    """
    regions_df = read_bed_with_summit(regions_path)

    if debug_chr:
        regions_df = regions_df[regions_df['chr'].isin(debug_chr)]
    
    # Load model directly with path string (no more mock class needed!)
    model = input_utils.load_model_wrapper(model_h5_path)

    # infer input length
    inputlen = model.input_shape[1] # if bias model (1 input only)
    print("inferred model inputlen: ", inputlen)

    # load sequences
    # NOTE: it will pull out sequences of length inputlen
    #       centered at the summit (start + 10th column) and peaks used after filtering

    genome = pyfaidx.Fasta(genome_path)
    seqs, peaks_used = input_utils.get_seq(regions_df, genome, inputlen)
    genome.close()

    regions_df[peaks_used].to_csv("{}.interpreted_regions.bed".format(output_prefix), header=False, index=False, sep='\t')

    interpret(model, seqs, output_prefix, profile_or_counts, chunk_size)
    
    return {
        'total_regions': len(regions_df),
        'processed_regions': len(regions_df[peaks_used]),
        'inputlen': inputlen,
        'output_files': [
            f"{output_prefix}.interpreted_regions.bed",
            f"{output_prefix}.counts_scores.h5" if "counts" in profile_or_counts else None,
            f"{output_prefix}.profile_scores.h5" if "profile" in profile_or_counts else None
        ]
    }


def main(args):
    """Main function for command-line interface."""
    
    # write all the command line arguments to a json file
    with open("{}.interpret.args.json".format(args.output_prefix), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=4)

    # Call core function with extracted parameters
    interpret_regions(
        genome_path=args.genome,
        regions_path=args.regions,
        model_h5_path=args.model_h5,
        output_prefix=args.output_prefix,
        profile_or_counts=args.profile_or_counts,
        debug_chr=args.debug_chr,
        chunk_size=getattr(args, 'chunk_size', None)
    )

if __name__ == '__main__':
    # parse the command line arguments
    args = fetch_interpret_args()
    main(args)

