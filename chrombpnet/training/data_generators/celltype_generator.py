"""
CelltypeGenerator for ChromBPNet Training with Dynamic Bias Scaling

This module implements the cell-type-level aggregation approach for DP model training.
It replaces the pseudobulk-level weighted dynamic pairing approach with a more efficient
cell-type-level aggregation strategy that addresses scale mismatch issues.

Key Features:
- Cell-type-level data loading: Uses aggregated bigWigs per cell type instead of individual pseudobulks
- Dynamic bias scaling: Provides scaling factors for each sample to correct bias model predictions
- Loss weighting: Applies cell-type-specific loss weights to ensure fair representation
- Returns 3-element tuple: ((batch_seq, batch_scaling_factors), (batch_cts, ...), batch_loss_weights)

Migration Note:
The original DPGenerator (weighted_dynamic) used pseudobulk-level pairing, which had two issues:
1. Computational inefficiency: Too many individual bigWig files to manage
2. Scale mismatch: Bias model trained on aggregated_all.bw vs individual pseudobulk scales

This CelltypeGenerator addresses both by:
1. Aggregating data at cell-type level (much fewer bigWig files)
2. Providing dynamic scaling factors to correct bias predictions to cell-type scale
"""

from tensorflow import keras
from chrombpnet.training.utils import augment
from chrombpnet.training.utils import data_utils
import tensorflow as tf
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import random
import math
import os
import json
import logging
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Global variables for worker processes (following ChromBPNet patterns)
genome_obj = None
bw_obj = None

def worker_init_bigwig_batch(genome_path, bigwig_path):
    """
    Initialize genome object once per worker process.
    This avoids multiprocessing deadlock from simultaneous file access.
    Pattern from find_bias_hyperparams.py and get_gc_content.py
    
    NOTE: bigwig_path parameter is kept for API compatibility but is not used.
    BigWig files are opened per task in process_celltype_peaks_worker to handle
    multiple files per worker (each task processes a different cell-type BigWig).
    This prevents resource leaks when a worker processes multiple tasks sequentially.
    """
    global genome_obj, bw_obj
    import pyfaidx
    
    genome_obj = pyfaidx.Fasta(genome_path)
    # BigWig files are opened per task, not per worker
    # This allows each worker to process multiple tasks with different BigWig files
    bw_obj = None

def process_celltype_peaks_worker(worker_args):
    """
    Worker function to process peak regions from a single cell-type BigWig file.
    Uses pre-initialized genome object and opens/closes BigWig file per task.
    Returns results with celltype metadata (scaling_factor, loss_weight_normalized, celltype_idx).
    
    CRITICAL: BigWig file is opened per task and closed in finally block to prevent
    resource leaks when a worker processes multiple tasks with different BigWig files.
    """
    peak_indices, peak_regions, inputlen, outputlen, bigwig_path, scaling_factor, loss_weight, celltype_idx = worker_args
    global genome_obj
    
    results = []
    chrom_sequences = {}  # Cache chromosome sequences per worker
    
    # Open BigWig file for this task
    # CRITICAL: Each task may process a different BigWig file, so we open/close per task
    # This prevents resource leaks when a worker processes multiple tasks sequentially
    bw_obj = None
    try:
        if bigwig_path is not None:
            import pyBigWig
            bw_obj = pyBigWig.open(bigwig_path)
        
        # Group by chromosome for memory-efficient processing
        chrom_groups = defaultdict(list)
        for i, (peak_idx, peak_row) in enumerate(zip(peak_indices, peak_regions.itertuples())):
            chrom = peak_row.chr
            chrom_groups[chrom].append((i, peak_idx, peak_row))
        
        # Process each chromosome in batch
        for chrom, chrom_peaks in chrom_groups.items():
            # Load entire chromosome sequence into memory once
            if chrom not in chrom_sequences:
                if chrom in genome_obj:
                    chrom_sequences[chrom] = str(genome_obj[chrom][:]).upper()
                else:
                    continue  # Skip chromosome not in genome
            
            chrom_seq = chrom_sequences[chrom]
            
            # Process all peaks for this chromosome
            for local_idx, original_peak_idx, peak_row in chrom_peaks:
                try:
                    # Calculate coordinates
                    center = peak_row.start + peak_row.summit
                    seq_start = center - inputlen // 2
                    seq_end = center + inputlen // 2
                    val_start = center - outputlen // 2
                    val_end = center + outputlen // 2
                    
                    # Extract sequence from in-memory chromosome
                    if seq_start >= 0 and seq_end <= len(chrom_seq):
                        sequence = chrom_seq[seq_start:seq_end]
                    else:
                        # Handle edge case
                        sequence = str(genome_obj[chrom][seq_start:seq_end])
                    
                    # Get bigwig values
                    if bw_obj is not None:
                        bigwig_vals = np.nan_to_num(bw_obj.values(chrom, val_start, val_end))
                    else:
                        bigwig_vals = np.zeros(outputlen)
                    
                    # Convert sequence to one-hot
                    from chrombpnet.training.utils import one_hot
                    seq_onehot = one_hot.dna_to_one_hot([sequence])[0]
                    
                    results.append({
                        'original_peak_idx': original_peak_idx,
                        'celltype_idx': celltype_idx,  # Critical: needed to match correct (peak_idx, celltype_idx) pair
                        'local_idx': local_idx,
                        'sequence': seq_onehot,
                        'counts': bigwig_vals,
                        'coords': (chrom, seq_start, seq_end),
                        'scaling_factor': scaling_factor,
                        'loss_weight': loss_weight
                    })
                    
                except Exception as e:
                    logging.warning(f"Error processing peak {original_peak_idx} in {chrom}: {e}")
                    continue
    
    finally:
        # CRITICAL: Always close BigWig file to prevent resource leaks
        # This is especially important when a worker processes multiple tasks
        # with different BigWig files (e.g., when using pool.imap)
        if bw_obj is not None:
            try:
                bw_obj.close()
            except Exception as e:
                logging.warning(f"Error closing BigWig file {bigwig_path}: {e}")
    
    return results


class CelltypeGenerator(keras.utils.Sequence):
    """
    Cell-type-level aggregation generator for ChromBPNet training with dynamic bias scaling.
    
    This generator implements the cell-type-level aggregation approach defined in the
    DP model v1.3 improvement plan. It addresses scale mismatch issues by:
    1. Loading data from cell-type aggregated bigWigs (much fewer files than pseudobulks)
    2. Providing scaling factors to dynamically adjust bias model predictions
    3. Applying loss weights to ensure fair representation of rare cell types
    
    Key Design:
    - Data structure: (peak_index, celltype_index) pairs
    - Input: Sequences + scaling factors (2-input model)
    - Output: Counts + log counts
    - Sample weights: Loss weights per sample
    """
    
    def __init__(self, 
                 peak_regions: Optional[pd.DataFrame], 
                 nonpeak_regions: Optional[pd.DataFrame],
                 celltype_metadata_path: str,
                 aggregated_bigwig_path: str,
                 genome_fasta: str, 
                 batch_size: int,
                 inputlen: int, 
                 outputlen: int, 
                 max_jitter: int,
                 negative_sampling_ratio: float,
                 add_revcomp: bool, 
                 return_coords: bool,
                 shuffle_at_epoch_start: bool,
                 mode: str = "train",
                 seed: Optional[int] = None,
                 override_scaling_factor: Optional[float] = None):
        """
        Initialize the Celltype Generator.
        
        Args:
            peak_regions: DataFrame with peak region coordinates
            nonpeak_regions: DataFrame with non-peak region coordinates (uses aggregated BigWig)
            celltype_metadata_path: Path to celltype_metadata_scaled.tsv file
            aggregated_bigwig_path: Path to aggregated all cell types BigWig (for non-peaks)
            genome_fasta: Path to genome FASTA file
            batch_size: Training batch size
            inputlen: Input sequence length
            outputlen: Output profile length
            max_jitter: Maximum jitter for training augmentation
            negative_sampling_ratio: Ratio of negative to positive samples
            add_revcomp: Apply reverse complement augmentation
            return_coords: Return coordinates with batches
            shuffle_at_epoch_start: Shuffle data at epoch start
            mode: Training mode ('train', 'valid', 'test')
            seed: Random seed for reproducibility
            override_scaling_factor: If provided, override all scaling factors with this value.
                                    Useful for prediction/interpretation to use standard scale (1.0).
                                    If None, uses cell-type-specific scaling factors from metadata.
        """
        
        # Store basic parameters
        self.peak_regions = peak_regions
        self.nonpeak_regions = nonpeak_regions
        self.genome_fasta = genome_fasta
        self.batch_size = batch_size
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.negative_sampling_ratio = negative_sampling_ratio
        self.return_coords = return_coords
        
        # Mode-specific augmentation
        self.mode = mode
        self.max_jitter = max_jitter if mode == 'train' else 0
        self.add_revcomp = add_revcomp if mode == 'train' else False
        self.shuffle_at_epoch_start = shuffle_at_epoch_start if mode == 'train' else False
        
        # Independent seed management for deterministic reproducibility
        self.base_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.rng = np.random.RandomState(self.base_seed)
        self.current_epoch = 0
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"CelltypeGenerator initialized with base seed: {self.base_seed}")
        self.logger.info(f"Mode: {mode}, Reproducible: {'Yes' if seed is not None else 'No'}")
        
        # Store override_scaling_factor for use in _load_celltype_metadata
        self.override_scaling_factor = override_scaling_factor
        
        # Initialize data components
        self._load_celltype_metadata(celltype_metadata_path)
        self.aggregated_bigwig_path = aggregated_bigwig_path
        
        # Validate aggregated BigWig existence
        if not Path(aggregated_bigwig_path).exists():
            raise FileNotFoundError(
                f"Aggregated BigWig file not found: {aggregated_bigwig_path}. "
                f"Please run Stage 05.3 (shared bias model training) first."
            )
        
        # Initialize epoch data
        self._initialize_epoch_data()
        
        # Initialize coordinates cache for return_coords=True
        self.coords_cache = {} if return_coords else None
        
        # Apply data augmentation for epoch 0
        self._crop_revcomp_data()
        
    def _load_celltype_metadata(self, celltype_metadata_path: str):
        """
        Load cell-type metadata with scaling factors and loss weights.
        
        Args:
            celltype_metadata_path: Path to celltype_metadata_scaled.tsv
        """
        self.logger.info(f"Loading cell-type metadata from: {celltype_metadata_path}")
        
        if not Path(celltype_metadata_path).exists():
            raise FileNotFoundError(
                f"Cell-type metadata file not found: {celltype_metadata_path}. "
                f"Please run Stage 05.1.5 (prepare celltype scaling factors) first."
            )
        
        # Load metadata
        self.celltype_metadata = pd.read_csv(celltype_metadata_path, sep='\t')
        
        # Validate required columns
        # Note: loss_weight_normalized is used for training to match non-peak loss weight scale (1.0)
        required_columns = ['cell_type', 'aggregated_bigwig_path', 'scaling_factor', 'loss_weight_normalized']
        missing_columns = [col for col in required_columns if col not in self.celltype_metadata.columns]
        if missing_columns:
            raise ValueError(
                f"Cell-type metadata file is missing required columns: {missing_columns}. "
                f"Please ensure Stage 05.1.5 completed successfully."
            )
        
        # Filter out rows with missing aggregated_bigwig_path
        self.celltype_metadata = self.celltype_metadata[
            self.celltype_metadata['aggregated_bigwig_path'].notna()
        ].copy()
        
        # Override scaling factors if specified
        # CRITICAL: For prediction/interpretation, use standard scale (1.0) as per plan
        # This ensures prediction results are independent of cell-type-specific scales
        if self.override_scaling_factor is not None:
            self.logger.info(
                f"Overriding all scaling factors with {self.override_scaling_factor} "
                f"(mode: {self.mode}, reason: {'prediction/interpretation' if self.mode == 'test' else 'user-specified'})"
            )
            self.celltype_metadata['scaling_factor'] = self.override_scaling_factor
        
        # Validate bigWig files exist
        for idx, row in self.celltype_metadata.iterrows():
            bigwig_path = Path(row['aggregated_bigwig_path'])
            if not bigwig_path.exists():
                raise FileNotFoundError(
                    f"Cell-type aggregated bigWig not found: {bigwig_path} "
                    f"for cell type: {row['cell_type']}"
                )
        
        self.logger.info(f"Loaded metadata for {len(self.celltype_metadata)} cell types:")
        for _, row in self.celltype_metadata.iterrows():
            self.logger.info(
                f"  {row['cell_type']:20s}: "
                f"scaling_factor={row['scaling_factor']:.6f}, "
                f"loss_weight_normalized={row['loss_weight_normalized']:.6f}"
            )
        
        # Create index mapping for efficient lookup
        self.celltype_to_index = {
            row['cell_type']: idx 
            for idx, row in self.celltype_metadata.iterrows()
        }
        self.index_to_celltype = {
            idx: row['cell_type'] 
            for idx, row in self.celltype_metadata.iterrows()
        }
        
    def _initialize_epoch_data(self):
        """
        Initialize data for the first epoch.
        """
        self.logger.info("Initializing epoch data...")
        
        # Load non-peak data from aggregated BigWig
        self._load_nonpeak_data_from_aggregated()
        
        # Initialize peak data based on mode
        if self.mode == 'train':
            self._plan_training_epoch()
        else:  # 'valid' or 'test'
            self._create_fixed_validation_set()
            
    def _load_nonpeak_data_from_aggregated(self):
        """
        Load non-peak region data from the single aggregated BigWig file.
        """
        if self.nonpeak_regions is None or len(self.nonpeak_regions) == 0:
            self.nonpeak_seqs = None
            self.nonpeak_cts = None
            self.nonpeak_coords = None
            self.nonpeak_scaling_factors = None
            self.nonpeak_loss_weights = None
            return
            
        self.logger.info(f"Loading {len(self.nonpeak_regions)} non-peak regions from aggregated BigWig...")
        
        # Use standard ChromBPNet data loading for non-peak regions
        genome = pyfaidx.Fasta(self.genome_fasta)
        cts_bw = pyBigWig.open(self.aggregated_bigwig_path)
        
        try:
            self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords = data_utils.get_seq_cts_coords(
                self.nonpeak_regions,
                genome,
                cts_bw,
                self.inputlen,
                self.outputlen,
                peaks_bool=0  # Non-peak regions
            )
        finally:
            cts_bw.close()
            genome.close()
            
        # For non-peak regions, use scaling_factor=1.0 (aggregated scale matches bias model)
        # and uniform loss weights
        num_nonpeaks = len(self.nonpeak_seqs) if self.nonpeak_seqs is not None else 0
        self.nonpeak_scaling_factors = np.ones(num_nonpeaks, dtype=np.float32)
        self.nonpeak_loss_weights = np.ones(num_nonpeaks, dtype=np.float32)
            
        self.logger.info(
            f"Loaded non-peak data: {num_nonpeaks} regions"
        )
    
    def _plan_training_epoch(self):
        """
        Plan training epoch with (peak, celltype) pairings.
        """
        if self.peak_regions is None or len(self.peak_regions) == 0:
            self.peak_seqs = None
            self.peak_cts = None  
            self.peak_coords = None
            self.peak_scaling_factors = None
            self.peak_loss_weights = None
            return
            
        epoch_start_time = time.time()
        self.logger.info("Planning training epoch with cell-type pairing...")
        
        # Generate (peak_index, celltype_index) pairs for this epoch
        self.epoch_peak_celltype_pairs = self._generate_peak_celltype_pairs()
        
        # Load peak data
        self._load_peak_data()
        
        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch planning completed in {epoch_time:.2f}s")
        
    def _create_fixed_validation_set(self):
        """
        Create fixed validation set with deterministic cell-type pairing.
        """
        if self.peak_regions is None or len(self.peak_regions) == 0:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
            self.peak_scaling_factors = None
            self.peak_loss_weights = None
            return
            
        self.logger.info("Creating fixed validation set...")
        
        # Create fixed (peak, celltype) pairs deterministically
        # Cycle through cell types for each peak
        celltype_indices = list(range(len(self.celltype_metadata)))
        validation_pairs = []
        
        for peak_idx in range(len(self.peak_regions)):
            # Cycle through cell types deterministically
            celltype_idx = peak_idx % len(celltype_indices)
            validation_pairs.append((peak_idx, celltype_idx))
        
        self.fixed_validation_pairs = validation_pairs
        self.logger.info(f"Generated {len(validation_pairs)} fixed validation pairs")
        
        # Load validation data
        self._load_peak_data()
        
    def _generate_peak_celltype_pairs(self):
        """
        Generate (peak_index, celltype_index) pairs for the current epoch.
        
        For training, we can use uniform sampling or weighted sampling.
        For simplicity, we use uniform sampling across cell types.
        """
        self.logger.info(f"Generating pairs for {len(self.peak_regions)} peaks...")
        
        # Set deterministic epoch-specific seed
        epoch_seed = self.base_seed + self.current_epoch * 997
        epoch_rng = np.random.RandomState(epoch_seed)
        
        self.logger.debug(f"Epoch {self.current_epoch} using seed: {epoch_seed}")
        
        # Shuffle peaks for this epoch
        shuffled_peak_indices = epoch_rng.permutation(len(self.peak_regions))
        
        pairs = []
        num_celltypes = len(self.celltype_metadata)
        
        for peak_idx in shuffled_peak_indices:
            # Uniform sampling across cell types
            celltype_idx = epoch_rng.randint(0, num_celltypes)
            pairs.append((peak_idx, celltype_idx))
        
        self.logger.info(f"Generated {len(pairs)} peak-celltype pairs")
        return pairs
    
    def _load_peak_data(self):
        """
        Load peak data from cell-type aggregated bigWigs with parallel processing.
        
        Uses multiprocessing.Pool for efficient parallel loading when multiple
        cell-type bigWig files need to be processed.
        """
        io_start_time = time.time()
        self.logger.info("Loading peak data from cell-type aggregated bigWigs...")
        
        # Determine which pairs to use
        if self.mode == 'train':
            pairs = self.epoch_peak_celltype_pairs
        else:
            pairs = self.fixed_validation_pairs
        
        if not pairs:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
            self.peak_scaling_factors = None
            self.peak_loss_weights = None
            return
        
        # Group pairs by cell type (bigWig file) for efficient I/O
        celltype_groups = defaultdict(list)
        for peak_idx, celltype_idx in pairs:
            celltype_groups[celltype_idx].append((peak_idx, celltype_idx))
        
        num_celltypes = len(celltype_groups)
        self.logger.info(f"Processing {num_celltypes} cell-type bigWig files...")
        
        # Determine optimal number of workers
        # Use 75% of available cores, cap at reasonable number
        max_safe_cores = max(1, int(cpu_count() * 0.75))
        max_workers = min(max_safe_cores, num_celltypes, 16)
        
        # Estimate memory usage
        total_peaks = len(pairs)
        estimated_memory = self._estimate_memory_usage(total_peaks, self.inputlen, self.outputlen)
        self.logger.info(f"Estimated memory usage: {estimated_memory:.2f}GB")
        
        # Use parallel processing if multiple cell types and sufficient memory
        if num_celltypes > 1 and max_workers > 1 and estimated_memory < 10.0:  # 10GB limit
            self.logger.info(f"Using parallel processing with {max_workers} workers")
            self._load_peak_data_parallel(celltype_groups, max_workers, pairs)
        else:
            self.logger.info("Using sequential processing")
            self._load_peak_data_sequential(celltype_groups)
        
        io_time = time.time() - io_start_time
        self.logger.info(
            f"Loaded peak data: {len(self.peak_seqs) if self.peak_seqs is not None else 0} regions in {io_time:.2f}s"
        )
    
    def _estimate_memory_usage(self, total_peaks: int, inputlen: int, outputlen: int) -> float:
        """
        Estimate memory usage for peak data loading.
        """
        # Rough estimation: sequence (4 * inputlen) + counts (outputlen) + overhead
        bytes_per_peak = 4 * inputlen * 4 + outputlen * 8 + 1000  # One-hot + counts + overhead
        total_bytes = total_peaks * bytes_per_peak
        return total_bytes / (1024**3)  # Convert to GB
    
    def _load_peak_data_parallel(self, celltype_groups: Dict, max_workers: int, pairs: List):
        """
        Load peak data using parallel processing.
        """
        # Prepare worker arguments
        worker_args_list = []
        for celltype_idx, group_pairs in celltype_groups.items():
            celltype_row = self.celltype_metadata.iloc[celltype_idx]
            bigwig_path = str(Path(celltype_row['aggregated_bigwig_path']))
            scaling_factor = float(celltype_row['scaling_factor'])
            loss_weight = float(celltype_row['loss_weight_normalized'])  # Use normalized weights
            
            # Extract peak regions
            peak_indices = [pair[0] for pair in group_pairs]
            file_peak_regions = self.peak_regions.iloc[peak_indices]
            
            worker_args_list.append((
                peak_indices,
                file_peak_regions,
                self.inputlen + 2 * self.max_jitter,
                self.outputlen + 2 * self.max_jitter,
                bigwig_path,
                scaling_factor,
                loss_weight,
                celltype_idx  # Critical: pass celltype_idx to worker
            ))
        
        # Process cell-type bigWigs in parallel
        all_results = []
        try:
            with Pool(processes=max_workers,
                     initializer=worker_init_bigwig_batch,
                     initargs=(self.genome_fasta, None)) as pool:
                
                # Use imap for progress tracking
                results_iter = pool.imap(process_celltype_peaks_worker, worker_args_list)
                
                # Collect results with progress tracking
                for file_results in tqdm(results_iter,
                                       total=len(worker_args_list),
                                       desc="Processing cell-type bigWigs"):
                    all_results.extend(file_results)
        
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            self.logger.info("Falling back to sequential processing")
            self._load_peak_data_sequential(celltype_groups)
            return
        
        # Reorganize results maintaining original pair order
        self._reorganize_parallel_results(all_results, pairs)
    
    def _load_peak_data_sequential(self, celltype_groups: Dict):
        """
        Load peak data using sequential processing (fallback).
        """
        # Initialize containers
        all_seqs = []
        all_cts = []
        all_coords = []
        all_scaling_factors = []
        all_loss_weights = []
        
        genome = pyfaidx.Fasta(self.genome_fasta)
        
        try:
            # Process each cell type's bigWig file sequentially
            for celltype_idx, group_pairs in celltype_groups.items():
                celltype_row = self.celltype_metadata.iloc[celltype_idx]
                bigwig_path = Path(celltype_row['aggregated_bigwig_path'])
                scaling_factor = float(celltype_row['scaling_factor'])
                loss_weight = float(celltype_row['loss_weight_normalized'])  # Use normalized weights
                
                self.logger.debug(
                    f"Processing {len(group_pairs)} peaks from {celltype_row['cell_type']} "
                    f"(scaling_factor={scaling_factor:.6f})"
                )
                
                # Extract peak regions
                peak_indices = [pair[0] for pair in group_pairs]
                file_peak_regions = self.peak_regions.iloc[peak_indices]
                
                # Load data from this cell type's bigWig
                cts_bw = pyBigWig.open(str(bigwig_path))
                try:
                    group_seqs, group_cts, group_coords = data_utils.get_seq_cts_coords(
                        file_peak_regions,
                        genome,
                        cts_bw,
                        self.inputlen + 2 * self.max_jitter,
                        self.outputlen + 2 * self.max_jitter,
                        peaks_bool=1  # Peak regions
                    )
                finally:
                    cts_bw.close()
                
                # Store data with corresponding scaling factors and loss weights
                all_seqs.extend(group_seqs)
                all_cts.extend(group_cts)
                all_coords.extend(group_coords)
                all_scaling_factors.extend([scaling_factor] * len(group_seqs))
                all_loss_weights.extend([loss_weight] * len(group_seqs))
        
        finally:
            genome.close()
        
        # Convert to numpy arrays
        if all_seqs:
            self.peak_seqs = np.array(all_seqs)
            self.peak_cts = np.array(all_cts)
            self.peak_coords = np.array(all_coords)
            self.peak_scaling_factors = np.array(all_scaling_factors, dtype=np.float32)
            self.peak_loss_weights = np.array(all_loss_weights, dtype=np.float32)
        else:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
            self.peak_scaling_factors = None
            self.peak_loss_weights = None
    
    def _reorganize_parallel_results(self, all_results: List[Dict], pairs: List):
        """
        Reorganize parallel processing results maintaining original pair order.
        Also extracts scaling factors and loss weights from results.
        
        Uses (peak_idx, celltype_idx) as key to correctly handle cases where
        the same peak_idx appears with multiple celltype_idx values.
        """
        # Build lookup dictionary: (peak_idx, celltype_idx) -> pair index
        # This ensures correct matching even when same peak_idx appears multiple times
        peak_celltype_to_pair_idx = {
            (peak_idx, celltype_idx): i 
            for i, (peak_idx, celltype_idx) in enumerate(pairs)
        }
        
        peak_data_map = {}
        
        # Map results back to original pair indices using (peak_idx, celltype_idx) key
        for result in all_results:
            key = (result['original_peak_idx'], result['celltype_idx'])
            
            if key in peak_celltype_to_pair_idx:
                original_pair_idx = peak_celltype_to_pair_idx[key]
                
                peak_data_map[original_pair_idx] = (
                    result['sequence'],
                    result['counts'],
                    result['coords'],
                    result['scaling_factor'],
                    result['loss_weight']
                )
            else:
                # This should not happen, but log warning if it does
                self.logger.warning(
                    f"Could not find pair for peak_idx={result['original_peak_idx']}, "
                    f"celltype_idx={result['celltype_idx']}"
                )
        
        # Build final arrays maintaining pair order
        if peak_data_map:
            ordered_seqs = []
            ordered_cts = []
            ordered_coords = []
            ordered_scaling_factors = []
            ordered_loss_weights = []
            
            for pair_idx in range(len(pairs)):
                if pair_idx in peak_data_map:
                    seq, cts, coord, scaling_factor, loss_weight = peak_data_map[pair_idx]
                    ordered_seqs.append(seq)
                    ordered_cts.append(cts)
                    ordered_coords.append(coord)
                    ordered_scaling_factors.append(scaling_factor)
                    ordered_loss_weights.append(loss_weight)
            
            self.peak_seqs = np.array(ordered_seqs)
            self.peak_cts = np.array(ordered_cts)
            self.peak_coords = np.array(ordered_coords)
            self.peak_scaling_factors = np.array(ordered_scaling_factors, dtype=np.float32)
            self.peak_loss_weights = np.array(ordered_loss_weights, dtype=np.float32)
        else:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
            self.peak_scaling_factors = None
            self.peak_loss_weights = None
    
    def _crop_revcomp_data(self):
        """
        Apply cropping, reverse complement, and other augmentations.
        Maintains scaling factors and loss weights through augmentation.
        """
        # Combine peak and non-peak data
        if (self.peak_seqs is not None) and (self.nonpeak_seqs is not None):
            # Use epoch-specific RandomState for reproducible cropping
            epoch_seed_crop = self.base_seed + self.current_epoch * 997 + 4
            epoch_rng_crop = np.random.RandomState(epoch_seed_crop)
            
            # Apply random crop to peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, 
                self.peak_coords, rng=epoch_rng_crop
            )
            
            # Maintain scaling factors and loss weights through cropping
            # (cropping doesn't change the number of samples, just their content)
            cropped_scaling_factors = self.peak_scaling_factors.copy()
            cropped_loss_weights = self.peak_loss_weights.copy()
            
            # Handle negative sampling ratio
            if self.negative_sampling_ratio < 1.0:
                sampled_nonpeak_seqs, sampled_nonpeak_cts, sampled_nonpeak_coords = self._subsample_nonpeak_data(
                    self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords, 
                    len(self.peak_seqs), self.negative_sampling_ratio
                )
                sampled_nonpeak_scaling_factors = self.nonpeak_scaling_factors[
                    :len(sampled_nonpeak_seqs)
                ]
                sampled_nonpeak_loss_weights = self.nonpeak_loss_weights[
                    :len(sampled_nonpeak_seqs)
                ]
                
                self.seqs = np.vstack([cropped_peaks, sampled_nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, sampled_nonpeak_cts]) 
                self.coords = np.vstack([cropped_coords, sampled_nonpeak_coords])
                self.scaling_factors = np.concatenate([
                    cropped_scaling_factors, sampled_nonpeak_scaling_factors
                ])
                self.loss_weights = np.concatenate([
                    cropped_loss_weights, sampled_nonpeak_loss_weights
                ])
            else:
                self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.nonpeak_coords])
                self.scaling_factors = np.concatenate([
                    cropped_scaling_factors, self.nonpeak_scaling_factors
                ])
                self.loss_weights = np.concatenate([
                    cropped_loss_weights, self.nonpeak_loss_weights
                ])
                
        elif self.peak_seqs is not None:
            # Only peak data
            # Use epoch-specific RandomState for reproducible cropping
            epoch_seed_crop = self.base_seed + self.current_epoch * 997 + 4
            epoch_rng_crop = np.random.RandomState(epoch_seed_crop)
            
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, 
                self.peak_coords, rng=epoch_rng_crop
            )
            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords
            self.scaling_factors = self.peak_scaling_factors.copy()
            self.loss_weights = self.peak_loss_weights.copy()
            
        elif self.nonpeak_seqs is not None:
            # Only non-peak data
            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
            self.coords = self.nonpeak_coords
            self.scaling_factors = self.nonpeak_scaling_factors.copy()
            self.loss_weights = self.nonpeak_loss_weights.copy()
        else:
            raise ValueError("Both peak and non-peak arrays are empty")
            
        # Apply final augmentation (crop, reverse complement, shuffle)
        # CRITICAL: Maintain scaling factors and loss weights through augmentation
        # We need to manually manage shuffling to ensure all arrays stay aligned
        
        if self.shuffle_at_epoch_start:
            # Generate deterministic shuffle indices before augmentation
            epoch_seed = self.base_seed + self.current_epoch * 997 + 2
            epoch_rng = np.random.RandomState(epoch_seed)
            shuffle_indices = epoch_rng.permutation(len(self.seqs))
            
            # Shuffle all arrays with the same indices BEFORE augmentation
            shuffled_seqs = self.seqs[shuffle_indices]
            shuffled_cts = self.cts[shuffle_indices]
            shuffled_coords = self.coords[shuffle_indices]
            shuffled_scaling_factors = self.scaling_factors[shuffle_indices]
            shuffled_loss_weights = self.loss_weights[shuffle_indices]
            
            # Apply reverse complement augmentation (without shuffle, as we already shuffled)
            # Use epoch-specific RandomState for reproducibility
            epoch_seed_augment = self.base_seed + self.current_epoch * 997 + 3
            epoch_rng_augment = np.random.RandomState(epoch_seed_augment)
            self.cur_seqs, self.cur_cts, self.cur_coords = augment.crop_revcomp_augment(
                shuffled_seqs, shuffled_cts, shuffled_coords, self.inputlen, self.outputlen,
                self.add_revcomp, shuffle=False, rng=epoch_rng_augment  # Use RandomState for reproducibility
            )
            
            # Scaling factors and loss weights are already shuffled and aligned
            self.cur_scaling_factors = shuffled_scaling_factors
            self.cur_loss_weights = shuffled_loss_weights
        else:
            # No shuffling, just apply reverse complement augmentation
            # Use epoch-specific RandomState for reproducibility
            epoch_seed_augment = self.base_seed + self.current_epoch * 997 + 3
            epoch_rng_augment = np.random.RandomState(epoch_seed_augment)
            self.cur_seqs, self.cur_cts, self.cur_coords = augment.crop_revcomp_augment(
                self.seqs, self.cts, self.coords, self.inputlen, self.outputlen,
                self.add_revcomp, shuffle=False, rng=epoch_rng_augment  # Use RandomState for reproducibility
            )
            self.cur_scaling_factors = self.scaling_factors.copy()
            self.cur_loss_weights = self.loss_weights.copy()
        
    def _subsample_nonpeak_data(self, nonpeak_seqs, nonpeak_cts, nonpeak_coords, 
                               peak_data_size, negative_sampling_ratio):
        """
        Randomly sample a portion of non-peak data.
        CRITICAL: Uses epoch-specific seed for deterministic reproducibility.
        """
        num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
        
        # Use epoch-specific seed for deterministic sampling
        epoch_seed = self.base_seed + self.current_epoch * 997 + 1
        epoch_rng = np.random.RandomState(epoch_seed)
        
        nonpeak_indices_to_keep = epoch_rng.choice(
            len(nonpeak_seqs), size=num_nonpeak_samples, replace=False
        )
        
        self.logger.debug(f"Subsampled {num_nonpeak_samples} non-peak regions using seed {epoch_seed}")
        
        return (
            nonpeak_seqs[nonpeak_indices_to_keep],
            nonpeak_cts[nonpeak_indices_to_keep], 
            nonpeak_coords[nonpeak_indices_to_keep]
        )
        
    def __len__(self):
        """Return number of batches per epoch."""
        return math.ceil(len(self.cur_seqs) / self.batch_size)
    
    def __getitem__(self, idx):
        """
        Get batch data with scaling factors and loss weights.
        
        Returns:
            3-element tuple: ((batch_seq, batch_scaling_factors), (batch_cts, batch_log_cts), batch_loss_weights)
            Coordinates are available via get_coords(idx) if return_coords=True
        """
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        
        batch_seq = self.cur_seqs[start_idx:end_idx]
        batch_cts = self.cur_cts[start_idx:end_idx]
        batch_scaling_factors = self.cur_scaling_factors[start_idx:end_idx]
        batch_loss_weights = self.cur_loss_weights[start_idx:end_idx]
        
        # Calculate log counts
        batch_log_cts = np.log(1 + batch_cts.sum(-1, keepdims=True))
        
        # Reshape scaling factors to (batch_size, 1) for model input
        batch_scaling_factors = batch_scaling_factors.reshape(-1, 1)
        
        # Store coordinates in cache if return_coords=True
        if self.return_coords:
            if self.coords_cache is None:
                self.coords_cache = {}
            self.coords_cache[idx] = self.cur_coords[start_idx:end_idx]
        
        # Always return 3-element tuple for Keras compatibility
        return (
            (batch_seq, batch_scaling_factors),
            (batch_cts, batch_log_cts),
            batch_loss_weights
        )
    
    def get_coords(self, idx):
        """
        Get coordinates for a specific batch index.
        Only available if return_coords=True.
        
        Args:
            idx: Batch index
            
        Returns:
            Coordinates array for the batch, or None if return_coords=False or coordinates not available
        """
        if not self.return_coords:
            return None
        if self.coords_cache is None:
            return None
        return self.coords_cache.get(idx, None)
    
    def on_epoch_end(self):
        """
        End-of-epoch processing.
        """
        self.current_epoch += 1
        self.logger.debug(f"Starting epoch {self.current_epoch}")
        
        # Clear coordinates cache at epoch end
        if self.return_coords:
            self.coords_cache = {}
        
        if self.mode == 'train':
            # Re-plan epoch for training
            self._plan_training_epoch()
        # For validation/test, use fixed data (no re-planning needed)
        
        # Apply augmentation processing
        self._crop_revcomp_data()
    
    def close(self):
        """
        Clean up resources when generator is closed.
        """
        self.logger.info("CelltypeGenerator closed")

