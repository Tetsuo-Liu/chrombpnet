import numpy as np
import chrombpnet.training.utils.one_hot as one_hot
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import chrombpnet.training.utils.losses as losses

def filter_edge_regions(peaks_df, bw, width, peaks_bool):
    """
    Filter regions in bed file that are on edges i.e regions that cannot be used to construct
    input length + jitter length of the sequence
    """
    input_shape = peaks_df.shape[0]

    # left edge case
    filtered = np.array((peaks_df['start'] + peaks_df['summit'] - width//2) < 0)
    peaks_df = peaks_df[~filtered]
    num_filtered = sum(filtered)

    # right edge case
    chrom_to_sizes = bw.chroms()
    filtered = []
    for i, r in peaks_df.iterrows():
        if r['start'] + r['summit'] + width//2 > chrom_to_sizes.get(r['chr'], 0) :
            filtered.append(True)
        else:
            filtered.append(False)
    filtered=np.array(filtered)
    peaks_df = peaks_df[~filtered]
    num_filtered += sum(filtered)

    if peaks_bool:
        print("Number of peaks input: ",input_shape)
        print("Number of peaks filtered because the input/output is on the edge: ", num_filtered)
        print("Number of peaks being used: ",peaks_df.shape[0])
    else:
        print("Number of non peaks input: ",input_shape)
        print("Number of non peaks filtered because the input/output is on the edge: ", num_filtered)
        print("Number of non peaks being used: ",peaks_df.shape[0])
    return peaks_df

def get_seqs_cts(genome, bw, peaks_df, input_width=2114, output_width=1000):
    """
    Output counts (not log counts)
    Output one-hot encoded sequence
    OPTIMIZED VERSION: Load chromosome sequences in memory to avoid I/O bottleneck
    """
    if len(peaks_df) == 0:
        return (np.array([]), one_hot.dna_to_one_hot([]))
    
    # Group by chromosome to minimize genome object access
    chrom_sequences = {}
    
    vals = []
    seqs = []
    
    # Process each chromosome's peaks in batch
    for chrom in peaks_df['chr'].unique():
        chrom_peaks = peaks_df[peaks_df['chr'] == chrom]
        
        # Load entire chromosome sequence into memory once per chromosome
        if chrom not in chrom_sequences:
            chrom_sequences[chrom] = str(genome[chrom][:]).upper()
        
        chrom_seq = chrom_sequences[chrom]
        
        # Process all peaks for this chromosome
        for _, row in chrom_peaks.iterrows():
            # Calculate coordinates (same logic as original)
            seq_start = int(row['start'] + row['summit'] - input_width // 2)
            seq_end = int(row['start'] + row['summit'] + input_width // 2)
            val_start = int(row['start'] + row['summit'] - output_width // 2)
            val_end = int(row['start'] + row['summit'] + output_width // 2)
            
            # Extract sequence from in-memory chromosome (major optimization)
            if seq_start >= 0 and seq_end <= len(chrom_seq):
                sequence = chrom_seq[seq_start:seq_end]
            else:
                # Handle edge case with padding (maintain original behavior)
                sequence = str(genome[chrom][seq_start:seq_end])
            
            seqs.append(sequence)
            
            # Get bigwig values (this is still I/O bound but unavoidable)
            bigwig_vals = np.nan_to_num(bw.values(chrom, val_start, val_end))
            vals.append(bigwig_vals)
    
    return (np.sum(np.array(vals), axis=1), one_hot.dna_to_one_hot(seqs))

def load_model_wrapper(model_h5):
    # read .h5 model
    custom_objects={"tf": tf, "multinomial_nll":losses.multinomial_nll}    
    get_custom_objects().update(custom_objects)    
    model=load_model(model_h5)
    print("got the model")
    model.summary()
    return model


