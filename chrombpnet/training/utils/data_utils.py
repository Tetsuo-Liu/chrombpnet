import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from chrombpnet.training.utils import one_hot
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Global variables for worker processes (following find_bias_hyperparams.py pattern)
genome_obj = None
bw_obj = None

def _filter_invalid_regions(peaks_df, genome, width):
    """
    Helper function to filter out regions that would fall off the chromosome edges.
    This ensures get_seq and get_cts operate on the same set of valid regions.
    """
    if len(peaks_df) == 0:
        return peaks_df

    original_size = len(peaks_df)
    valid_indices = []
    
    for chrom, group in peaks_df.groupby('chr'):
        if chrom not in genome:
            logging.warning(f"Chromosome {chrom} not found in genome, filtering out {len(group)} regions.")
            continue

        chrom_len = len(genome[chrom])
        
        centers = group['start'] + group['summit']
        starts = centers - width // 2
        ends = centers + width // 2
        
        # Find indices of rows within the current group that are valid
        group_valid_indices = group.index[(starts >= 0) & (ends <= chrom_len)]
        valid_indices.extend(group_valid_indices)

    filtered_df = peaks_df.loc[valid_indices]
    num_filtered = original_size - len(filtered_df)
    
    if num_filtered > 0:
        logging.warning(f"Filtered out {num_filtered} regions that fall off chromosome edges.")
    
    return filtered_df

def worker_init_seq(genome_path):
    """
    Initialize genome object once per worker process.
    This avoids multiprocessing deadlock from simultaneous file access.
    Pattern from find_bias_hyperparams.py
    """
    global genome_obj
    import pyfaidx
    genome_obj = pyfaidx.Fasta(genome_path)

def worker_init_cts(bw_path):
    """
    Initialize BigWig object once per worker process.
    This avoids multiprocessing deadlock from simultaneous file access.
    Pattern from find_bias_hyperparams.py
    """
    global bw_obj
    import pyBigWig
    bw_obj = pyBigWig.open(bw_path)

def process_chromosome_seq(worker_args):
    """
    Worker function to process sequences for a single chromosome.
    Uses pre-initialized global objects and chromosome-level memory loading.
    Combines patterns from find_bias_hyperparams.py and param_utils.py
    """
    chrom, group, width = worker_args
    global genome_obj
    
    if chrom not in genome_obj:
        logging.warning(f"Chromosome {chrom} not found in genome, skipping.")
        return chrom, []
    
    # Load entire chromosome sequence into memory once (param_utils.py pattern)
    chrom_seq = str(genome_obj[chrom][:])
    sequences = []
    
    for _, r in group.iterrows():
        center = r['start'] + r['summit']
        start = center - width // 2
        end = center + width // 2
        
        # Pre-filtering ensures start/end are always valid, so no need for checks here
        sequences.append(chrom_seq[start:end])
    
    return chrom, sequences

def process_chromosome_cts(worker_args):
    """
    Worker function to process counts for a single chromosome.
    Uses pre-initialized global objects.
    Pattern from find_bias_hyperparams.py
    """
    chrom, group, width = worker_args
    global bw_obj
    
    vals = []
    for _, r in group.iterrows():
        center = r['start'] + r['summit']
        start = center - width // 2
        end = center + width // 2
        
        try:
            counts = bw_obj.values(chrom, start, end)
            vals.append(np.nan_to_num(counts))
        except RuntimeError:
            logging.warning(f"Could not fetch bigWig values for {chrom}:{start}-{end}, using zeros.")
            vals.append(np.zeros(width))
    
    return chrom, vals

def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    OPTIMIZED: Uses parallel processing with chromosome-level memory loading.
    
    Note: This function assumes the input peaks_df has already been filtered 
    for edge regions by the caller. For standard usage, call via 
    get_seq_cts_coords() which handles edge filtering consistently.
    """
    if len(peaks_df) == 0:
        return one_hot.dna_to_one_hot([])
    
    # Determine optimal number of jobs (from find_bias_hyperparams.py pattern)
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
            results = list(tqdm(pool.imap(process_chromosome_seq, worker_args), 
                              total=len(worker_args), 
                              desc="Fetching sequences"))
    except Exception as e:
        logging.error(f"Parallel processing failed: {e}")
        # Fallback to sequential processing
        logging.info("Falling back to sequential processing")
        vals = []
        for i, r in peaks_df.iterrows():
            sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
            vals.append(sequence)
        return one_hot.dna_to_one_hot(vals)
    
    # Merge results while preserving original order
    all_sequences = []
    result_dict = dict(results)
    
    # Preserve original chromosome order
    for chrom in peaks_df['chr'].unique():
        if chrom in result_dict:
            all_sequences.extend(result_dict[chrom])
    
    return one_hot.dna_to_one_hot(all_sequences)


def get_cts(peaks_df, bw, width):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.
    OPTIMIZED: Uses parallel processing with chromosome grouping.

    "cts" = per base counts across a region
    
    Note: This function assumes the input peaks_df has already been filtered 
    for edge regions by the caller. For standard usage, call via 
    get_seq_cts_coords() which handles edge filtering consistently.
    """
    if len(peaks_df) == 0:
        return np.array([])

    # Determine optimal number of jobs (from find_bias_hyperparams.py pattern)
    n_jobs = min(max(1, cpu_count() // 2), 16)
    
    # Group by chromosome for parallel processing
    worker_args = [(chrom, group, width) for chrom, group in peaks_df.groupby('chr')]
    
    # Get BigWig file path for worker initialization
    bw_file = bw.filename() if hasattr(bw, 'filename') else None
    if bw_file is None:
        logging.warning("Could not get BigWig filename, falling back to sequential processing")
        # Fallback to original sequential processing
        vals = []
        for i, r in peaks_df.iterrows():
            vals.append(np.nan_to_num(bw.values(r['chr'], 
                                                r['start'] + r['summit'] - width//2,
                                                r['start'] + r['summit'] + width//2)))
        return np.array(vals)
    
    try:
        with Pool(processes=min(n_jobs, len(worker_args)), 
                  initializer=worker_init_cts, 
                  initargs=(bw_file,)) as pool:
            # Process chromosomes in parallel with progress bar
            results = list(tqdm(pool.imap(process_chromosome_cts, worker_args), 
                              total=len(worker_args), 
                              desc="Fetching counts"))
    except Exception as e:
        logging.error(f"Parallel processing failed: {e}")
        # Fallback to sequential processing
        logging.info("Falling back to sequential processing")
        vals = []
        for i, r in peaks_df.iterrows():
            vals.append(np.nan_to_num(bw.values(r['chr'], 
                                                r['start'] + r['summit'] - width//2,
                                                r['start'] + r['summit'] + width//2)))
        return np.array(vals)
    
    # Merge results while preserving original order
    all_vals = []
    result_dict = dict(results)
    
    # Preserve original chromosome order
    for chrom in peaks_df['chr'].unique():
        if chrom in result_dict:
            all_vals.extend(result_dict[chrom])
    
    return np.array(all_vals)

def get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit, strand, peaks_flag)
    
    Note: This function assumes the input peaks_df has already been filtered 
    for edge regions by the caller. For standard usage, call via 
    get_seq_cts_coords() which handles edge filtering consistently.
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['summit'], "f", str(peaks_bool)])

    return np.array(vals)

def get_seq_cts_coords(peaks_df, genome, bw, input_width, output_width, peaks_bool):
    """
    Fetches sequences, counts, and coordinates for peak/non-peak regions.
    
    This function handles edge region filtering internally using the more 
    conservative of the two width parameters to ensure consistent data 
    between seq, cts, and coords outputs.
    
    Args:
        peaks_df: DataFrame with chr, start, summit columns
        genome: pyfaidx.Fasta genome object
        bw: pyBigWig object 
        input_width: Width for sequence extraction
        output_width: Width for count extraction
        peaks_bool: 1 for peaks, 0 for non-peaks
        
    Returns:
        tuple: (sequences, counts, coordinates) arrays
    """
    # Filter invalid regions using the more conservative width requirement
    # This ensures all three outputs (seq, cts, coords) operate on the same valid regions
    filter_width = max(input_width, output_width)
    valid_peaks_df = _filter_invalid_regions(peaks_df, genome, filter_width)

    seq = get_seq(valid_peaks_df, genome, input_width)
    cts = get_cts(valid_peaks_df, bw, output_width)
    coords = get_coords(valid_peaks_df, peaks_bool)

    return seq, cts, coords

def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter):
    """
    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.

    If outliers is not None, removes training examples with counts > outlier%ile
    """

    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_coords=None
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(bed_regions,
                                              genome,
                                              cts_bw,
                                              inputlen+2*max_jitter,
                                              outputlen+2*max_jitter,
                                              peaks_bool=1)
    
    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(nonpeak_regions,
                                              genome,
                                              cts_bw,
                                              inputlen,
                                              outputlen,
                                              peaks_bool=0)



    cts_bw.close()
    genome.close()

    return (train_peaks_seqs, train_peaks_cts, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords)
