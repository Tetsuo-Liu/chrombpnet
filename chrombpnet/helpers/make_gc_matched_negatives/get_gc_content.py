import pyfaidx
import argparse
from tqdm import tqdm 
import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import functools

def parse_args():
    parser=argparse.ArgumentParser(description="get gc content from a foreground bed file")
    parser.add_argument("-i","--input_bed", help="bed file in narrow peak format - we will find gc content of these regions centered on the summit")
    parser.add_argument("-c","--chrom_sizes",type=str, required=True, help="TSV file with chromosome name in first column and size in the second column")
    parser.add_argument("-g", "--genome", help="reference genome fasta")
    parser.add_argument("-op", "--output_prefix", help="output file prefix for storing gc-content values of given foreground bed")
    parser.add_argument("-il","--inputlen",type=int,default=2114, help="inputlen to use to find gc-content")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Number of parallel jobs (default: auto-detect)")
    return parser.parse_args()

def calculate_gc_content_vectorized(sequences):
    """
    Vectorized GC content calculation that produces identical results to the original.
    
    Args:
        sequences: List of DNA sequences (strings)
        
    Returns:
        numpy array of GC fractions rounded to 2 decimal places
    """
    gc_fractions = []
    for seq in sequences:
        # Use identical calculation logic as original
        g = seq.count('G')
        c = seq.count('C')
        gc = g + c
        gc_fract = round(gc / len(seq), 2)
        gc_fractions.append(gc_fract)
    return np.array(gc_fractions)

def process_chromosome_worker(args_tuple):
    """
    Worker function for parallel chromosome processing.
    
    Args:
        args_tuple: (chrom, regions, genome_file, chrom_sizes_dict, inputlen)
        
    Returns:
        list of (original_index, chrom, adjusted_start, adjusted_end, gc_fract) tuples
    """
    chrom, regions, genome_file, chrom_sizes_dict, inputlen = args_tuple
    
    # Create a new pyfaidx.Fasta object for this worker (thread-safe)
    ref = pyfaidx.Fasta(genome_file)

    # [OPTIMIZATION] Load entire chromosome sequence into memory to avoid I/O bottleneck.
    chrom_sequence = ref[chrom][:].seq.upper()
    
    results = []
    sequences = []
    valid_regions = []
    
    # Pre-calculate adjusted coordinates and filter invalid regions
    # Add a tqdm progress bar for processing regions within a chromosome
    for original_idx, start, end, summit in tqdm(regions, desc=f"Processing {chrom}", unit="peaks"):
        adjusted_start = summit - inputlen // 2
        adjusted_end = summit + inputlen // 2
        
        # Apply same filtering logic as original, but check against in-memory sequence length
        if adjusted_start >= 0 and adjusted_end <= len(chrom_sequence):
            valid_regions.append((original_idx, adjusted_start, adjusted_end))
            # [OPTIMIZATION] Slice from in-memory sequence instead of file I/O
            seq = chrom_sequence[adjusted_start:adjusted_end]
            sequences.append(seq)
    
    if sequences:
        # Calculate GC content for all sequences at once (but maintaining identical calculation)
        gc_fractions = calculate_gc_content_vectorized(sequences)
        
        # Combine results
        for (original_idx, adjusted_start, adjusted_end), gc_fract in zip(valid_regions, gc_fractions):
            results.append((original_idx, chrom, adjusted_start, adjusted_end, gc_fract))
    
    return results

def main(args):
    # Identical initialization as original
    chrom_sizes_dict = {line.strip().split("\t")[0]:int(line.strip().split("\t")[1]) for line in open(args.chrom_sizes).readlines()}
    data = pd.read_csv(args.input_bed, header=None, sep='\t')
    assert(args.inputlen % 2 == 0)  # for symmetry
    
    num_rows = str(data.shape[0])
    print("num_rows:" + num_rows)
    
    # Determine number of parallel jobs
    if args.jobs is None:
        # Use 75% of available cores, but cap at number of chromosomes
        # Leave 25% for system stability and other processes
        max_safe_cores = max(1, int(cpu_count() * 0.75))
        n_jobs = min(max_safe_cores, 23)  # Don't exceed number of chromosomes
    else:
        n_jobs = args.jobs
    
    print(f"Using {n_jobs} parallel jobs for chromosome processing")
    
    # Group regions by chromosome for batch processing
    chrom_regions = defaultdict(list)
    
    # Prepare data with same logic as original but without processing yet
    for index, row in data.iterrows():
        chrom = row[0]
        start = row[1]
        end = row[2]
        summit = start + row[9]
        
        # Store for batch processing
        chrom_regions[chrom].append((index, start, end, summit))
    
    # Filter chromosomes and prepare worker arguments
    worker_args = []
    filtered_points = 0
    
    for chrom, regions in chrom_regions.items():
        if chrom not in chrom_sizes_dict:
            # Skip chromosomes not in sizes file (same as original behavior)
            filtered_points += len(regions)
            continue
        
        # Filter regions using same logic as original
        valid_regions = []
        for original_idx, start, end, summit in regions:
            adjusted_start = summit - args.inputlen // 2
            adjusted_end = summit + args.inputlen // 2
            
            # Apply identical filtering conditions
            if adjusted_start < 0 or adjusted_end > chrom_sizes_dict[chrom]:
                filtered_points += 1
            else:
                valid_regions.append((original_idx, start, end, summit))
        
        if valid_regions:
            # Prepare arguments for worker
            worker_args.append((chrom, valid_regions, args.genome, chrom_sizes_dict, args.inputlen))
    
    # Process chromosomes in parallel
    all_results = []
    
    if worker_args:
        if n_jobs == 1:
            # Sequential processing for debugging or single-core systems
            print("Processing chromosomes sequentially")
            for worker_arg in tqdm(worker_args, desc="Processing chromosomes"):
                chrom_results = process_chromosome_worker(worker_arg)
                all_results.extend(chrom_results)
        else:
            # Parallel processing
            print(f"Processing {len(worker_args)} chromosomes in parallel")
            print(f"Chromosomes to process: {[chrom for chrom, _ in [(arg[0], len(arg[1])) for arg in worker_args]]}")
            
            with Pool(processes=n_jobs) as pool:
                print("Starting parallel chromosome processing...")
                
                results_iterator = pool.imap_unordered(process_chromosome_worker, worker_args)
                
                completed_count = 0
                total_tasks = len(worker_args)
                
                for chrom_results in results_iterator:
                    completed_count += 1
                    # chrom_results is a list of tuples, get chrom name from the first result
                    if chrom_results:
                        chrom_name = chrom_results[0][1]
                        print(f"  [PROGRESS] Completed: {chrom_name} ({completed_count}/{total_tasks})")
                    
                    all_results.extend(chrom_results)
                
                print(f"✅ Completed processing {total_tasks} chromosomes.")
    
    # Sort results by original index to maintain identical output order
    all_results.sort(key=lambda x: x[0])
    
    # Write output in identical format as original
    outf = open(args.output_prefix + ".bed", 'w')
    for original_idx, chrom, start, end, gc_fract in all_results:
        outf.write(chrom + '\t' + str(start) + '\t' + str(end) + '\t' + str(gc_fract) + "\n")
    outf.close()
    
    # Identical final reporting as original
    print("Number of regions filtered because inputlen sequence cannot be constructed: " + str(filtered_points))
    print("Percentage of regions filtered " + str(round(filtered_points * 100.0 / data.shape[0], 3)) + "%")
    if round(filtered_points * 100.0 / data.shape[0], 3) > 25:
        print("WARNING: If percentage of regions filtered is high (>25%) - your genome is very small - consider using a reduced input/output length for your genome")

if __name__ == "__main__":
    args = parse_args()
    main(args)