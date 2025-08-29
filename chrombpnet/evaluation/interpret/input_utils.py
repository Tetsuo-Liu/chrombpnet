import os
import sys
import tensorflow as tf
import numpy as np
import chrombpnet.training.utils.losses as losses
from chrombpnet.training.utils.data_utils import one_hot
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Global variables for worker processes (following data_utils.py pattern)
genome_obj = None

def worker_init_seq(genome_path):
    """
    Initialize genome object once per worker process.
    This avoids multiprocessing deadlock from simultaneous file access.
    Pattern from data_utils.py
    """
    global genome_obj
    import pyfaidx
    genome_obj = pyfaidx.Fasta(genome_path)

def process_chromosome_seq_interpret(worker_args):
    """
    Worker function to process sequences for interpretation.
    Uses pre-initialized global objects and chromosome-level memory loading.
    Adapted from data_utils.py pattern
    """
    chrom, group, width = worker_args
    global genome_obj
    
    if chrom not in genome_obj:
        logging.warning(f"Chromosome {chrom} not found in genome, skipping.")
        return chrom, [], []
    
    # Load entire chromosome sequence into memory once (data_utils.py pattern)
    chrom_seq = str(genome_obj[chrom][:])
    sequences = []
    peaks_used = []
    
    for _, r in group.iterrows():
        center = r['start'] + r['summit']
        start = center - width // 2
        end = center + width // 2
        
        # Check bounds and sequence length
        if start >= 0 and end <= len(chrom_seq):
            sequence = chrom_seq[start:end]
            if len(sequence) == width:
                sequences.append(sequence)
                peaks_used.append(True)
            else:
                peaks_used.append(False)
        else:
            peaks_used.append(False)
    
    return chrom, sequences, peaks_used

def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    OPTIMIZED: Uses parallel processing with chromosome-level memory loading.
    """
    if len(peaks_df) == 0:
        return one_hot.dna_to_one_hot([]), np.array([])
    
    # Determine optimal number of jobs (from data_utils.py pattern)
    n_jobs = min(max(1, cpu_count() // 2), 16)
    
    # Group by chromosome for parallel processing
    worker_args = [(chrom, group, width) for chrom, group in peaks_df.groupby('chr')]
    
    # Get genome file path for worker initialization
    genome_path = genome.filename
    
    try:
        with Pool(processes=min(n_jobs, len(worker_args)), 
                  initializer=worker_init_seq, 
                  initargs=(genome_path,)) as pool:
            # Process chromosomes in parallel with progress bar
            results = list(tqdm(pool.imap(process_chromosome_seq_interpret, worker_args), 
                              total=len(worker_args), 
                              desc="Loading sequences"))
    except Exception as e:
        logging.error(f"Parallel processing failed: {e}")
        # Fallback to sequential processing (original code)
        logging.info("Falling back to sequential processing")
        vals = []
        peaks_used = []
        for i, r in peaks_df.iterrows():
            sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
            if len(sequence) == width:
                vals.append(sequence)
                peaks_used.append(True)
            else:
                peaks_used.append(False)
        return one_hot.dna_to_one_hot(vals), np.array(peaks_used)
    
    # Merge results while preserving original order
    all_sequences = []
    all_peaks_used = []
    result_dict = dict((chrom, (seqs, used)) for chrom, seqs, used in results)
    
    # Preserve original chromosome order
    for chrom in peaks_df['chr'].unique():
        if chrom in result_dict:
            seqs, used = result_dict[chrom]
            all_sequences.extend(seqs)
            all_peaks_used.extend(used)
    
    return one_hot.dna_to_one_hot(all_sequences), np.array(all_peaks_used)


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
    custom_objects={"multinomial_nll": losses.multinomial_nll, "tf": tf}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_path, compile=False)
    print("got the model")
    model.summary()
    return model

