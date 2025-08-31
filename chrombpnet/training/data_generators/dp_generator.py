"""
DP (Weighted Dynamic Pairing) Generator for ChromBPNet Training

This module implements the core innovation of the DP model: weighted dynamic pairing
data loader that ensures fair representation of rare cell types during training by
probabilistically sampling pseudobulks based on their sampling weights.

Key Features:
- Hybrid architecture: DP for peak regions, aggregated BigWig for non-peak regions
- Mode-specific augmentation: training vs validation/test reproducibility
- Global representative selection for cross-validation consistency
- Comprehensive validation integration with mandatory error termination
"""

from tensorflow import keras
from chrombpnet.training.utils import augment
from chrombpnet.training.utils import data_utils
from chrombpnet.training.utils.pseudobulk_validation import (
    PseudobulkMetadata, 
    load_pseudobulk_metadata, 
    validate_and_enforce_file_integrity
)
import tensorflow as tf
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import random
import string
import math
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import psutil
import gc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from collections import defaultdict


# Global variables for worker processes (following ChromBPNet patterns)
genome_obj = None
bw_obj = None

def worker_init_bigwig_batch(genome_path, bigwig_path):
    """
    Initialize genome and bigwig objects once per worker process.
    This avoids multiprocessing deadlock from simultaneous file access.
    Pattern from find_bias_hyperparams.py and get_gc_content.py
    """
    global genome_obj, bw_obj
    import pyfaidx
    import pyBigWig
    
    genome_obj = pyfaidx.Fasta(genome_path)
    bw_obj = pyBigWig.open(bigwig_path)

def process_peaks_batch_worker(worker_args):
    """
    Worker function to process peak regions from a single BigWig file.
    Uses pre-initialized global objects and chromosome-level memory loading.
    Adapted from get_gc_content.py and param_utils.py patterns
    """
    peak_indices, peak_regions, inputlen, outputlen, bigwig_path = worker_args
    global genome_obj, bw_obj
    
    if bw_obj is None:
        # Fallback initialization if needed
        import pyBigWig
        bw_obj = pyBigWig.open(bigwig_path)
    
    results = []
    chrom_sequences = {}  # Cache chromosome sequences per worker
    
    # Group by chromosome for memory-efficient processing
    chrom_groups = defaultdict(list)
    for i, (peak_idx, peak_row) in enumerate(zip(peak_indices, peak_regions.itertuples())):
        chrom = peak_row.chr
        chrom_groups[chrom].append((i, peak_idx, peak_row))
    
    # Process each chromosome in batch
    for chrom, chrom_peaks in chrom_groups.items():
        # Load entire chromosome sequence into memory once (param_utils.py pattern)
        if chrom not in chrom_sequences:
            if chrom in genome_obj:
                chrom_sequences[chrom] = str(genome_obj[chrom][:]).upper()
            else:
                continue  # Skip chromosome not in genome
        
        chrom_seq = chrom_sequences[chrom]
        
        # Process all peaks for this chromosome
        for local_idx, original_peak_idx, peak_row in chrom_peaks:
            try:
                # Calculate coordinates (following ChromBPNet patterns)
                center = peak_row.start + peak_row.summit
                seq_start = center - inputlen // 2
                seq_end = center + inputlen // 2
                val_start = center - outputlen // 2
                val_end = center + outputlen // 2
                
                # Extract sequence from in-memory chromosome (major optimization)
                if seq_start >= 0 and seq_end <= len(chrom_seq):
                    sequence = chrom_seq[seq_start:seq_end]
                else:
                    # Handle edge case (maintain ChromBPNet behavior)
                    sequence = str(genome_obj[chrom][seq_start:seq_end])
                
                # Get bigwig values (I/O bound but unavoidable)
                bigwig_vals = np.nan_to_num(bw_obj.values(chrom, val_start, val_end))
                
                # Convert sequence to one-hot
                from chrombpnet.training.utils import one_hot
                seq_onehot = one_hot.dna_to_one_hot([sequence])[0]
                
                results.append({
                    'original_peak_idx': original_peak_idx,
                    'local_idx': local_idx,
                    'sequence': seq_onehot,
                    'counts': bigwig_vals,
                    'coords': (chrom, seq_start, seq_end)
                })
                
            except Exception as e:
                logging.warning(f"Error processing peak {original_peak_idx} in {chrom}: {e}")
                continue
    
    return results

class BigWigConnectionPool:
    """
    Connection pool for BigWig files to minimize file open/close overhead.
    Implements resource management following ChromBPNet patterns.
    """
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = {}  # {filepath: pyBigWig_object}
        self.access_times = {}  # {filepath: last_access_time}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self, bigwig_path: str):
        """
        Get BigWig connection with automatic resource management.
        """
        with self.lock:
            current_time = time.time()
            
            # Return existing connection if available
            if bigwig_path in self.connections:
                self.access_times[bigwig_path] = current_time
                return self.connections[bigwig_path]
            
            # Clean up old connections if pool is full
            if len(self.connections) >= self.max_connections:
                self._cleanup_old_connections()
            
            # Create new connection
            try:
                bw = pyBigWig.open(bigwig_path)
                if bw is None:
                    raise ValueError(f"Failed to open BigWig file: {bigwig_path}")
                
                self.connections[bigwig_path] = bw
                self.access_times[bigwig_path] = current_time
                
                self.logger.debug(f"Opened new BigWig connection: {Path(bigwig_path).name}")
                return bw
                
            except Exception as e:
                self.logger.error(f"Failed to open BigWig file {bigwig_path}: {e}")
                raise
    
    def _cleanup_old_connections(self):
        """
        Close least recently used connections.
        """
        if len(self.connections) < self.max_connections:
            return
        
        # Sort by access time and close oldest
        sorted_paths = sorted(self.access_times.items(), key=lambda x: x[1])
        paths_to_close = [path for path, _ in sorted_paths[:len(sorted_paths)//2]]
        
        for path in paths_to_close:
            if path in self.connections:
                try:
                    self.connections[path].close()
                    del self.connections[path]
                    del self.access_times[path]
                    self.logger.debug(f"Closed old BigWig connection: {Path(path).name}")
                except Exception as e:
                    self.logger.warning(f"Error closing BigWig connection: {e}")
    
    def close_all(self):
        """
        Close all connections and clean up resources.
        """
        with self.lock:
            for path, bw in self.connections.items():
                try:
                    bw.close()
                    self.logger.debug(f"Closed BigWig connection: {Path(path).name}")
                except Exception:
                    pass
            
            self.connections.clear()
            self.access_times.clear()

class EpochDataPlanner:
    """
    Epoch-level data planning system for memory-efficient pre-loading.
    Implements performance optimization patterns from ChromBPNet helpers.
    """
    
    def __init__(self, memory_limit_gb: float = 2.0):
        self.memory_limit_gb = memory_limit_gb
        self.logger = logging.getLogger(__name__)
    
    def plan_peak_data_loading(self, peak_pseudobulk_pairs, inputlen: int, outputlen: int, 
                             genome_fasta: str, max_workers: int = None) -> Dict:
        """
        Plan efficient data loading for peak regions using parallel processing.
        Follows patterns from get_gc_content.py and find_bias_hyperparams.py
        """
        start_time = time.time()
        
        # Group pairs by BigWig file for batch processing
        file_groups = defaultdict(list)
        for peak_idx, pseudobulk_metadata in peak_pseudobulk_pairs:
            bigwig_path = str(pseudobulk_metadata['bigwig_path'])
            file_groups[bigwig_path].append((peak_idx, pseudobulk_metadata))
        
        self.logger.info(f"Planning data loading for {len(file_groups)} BigWig files")
        
        # Determine optimal number of workers (following ChromBPNet patterns)
        if max_workers is None:
            # Use 75% of available cores for optimal performance while maintaining system stability
            max_safe_cores = max(1, int(cpu_count() * 0.75))
            # Cap at reasonable number for BigWig files
            max_workers = min(max_safe_cores, len(file_groups), 16)
        
        self.logger.info(f"Using {max_workers} parallel workers for data loading")
        
        # Check memory usage before planning
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < self.memory_limit_gb:
            self.logger.warning(f"Low memory detected: {available_memory:.1f}GB available")
        
        planning_time = time.time() - start_time
        self.logger.info(f"Data loading planning completed in {planning_time:.2f}s")
        
        return {
            'file_groups': file_groups,
            'max_workers': max_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'planning_time': planning_time
        }
    
    def estimate_memory_usage(self, total_peaks: int, inputlen: int, outputlen: int) -> float:
        """
        Estimate memory usage for peak data loading.
        """
        # Rough estimation: sequence (4 * inputlen) + counts (outputlen) + overhead
        bytes_per_peak = 4 * inputlen * 4 + outputlen * 8 + 1000  # One-hot + counts + overhead
        total_bytes = total_peaks * bytes_per_peak
        return total_bytes / (1024**3)  # Convert to GB


class DPGenerator(keras.utils.Sequence):
    """
    Weighted Dynamic Pairing (DP) Generator for ChromBPNet Training

    This generator implements the core innovation of the DP model: a weighted dynamic
    pairing data loader. Its primary goal is to ensure fair representation of rare
    cell types during training by probabilistically sampling pseudobulks based on
    pre-calculated sampling weights.

    Key Design Principles:

    1.  **Hybrid Data Loading for Consistency and Efficiency**:
        To accurately model both biological signals and technical biases while
        maintaining performance, a hybrid data loading strategy is employed:

        a) **Peak Regions (Dynamic Pairing)**: Loaded from individual pseudobulk
        BigWigs. This is the core of the DP strategy, ensuring that sequence
        signals from rare cell types (e.g., hematopoietic stem cells) are
        adequately represented during the learning of transcription factor
        motifs and regulatory syntax.

        b) **Non-Peak Regions (Aggregated Access)**: Loaded from a single,
        pre-aggregated BigWig file. This design choice is critical for two
        reasons:
        i.  **Model Consistency**: The shared bias model (from Stage 05.3) was
            trained on this same aggregated data. By using the aggregated
            BigWig for non-peaks during DP model training, we ensure that the
            background signal distribution is perfectly consistent with the
            distribution on which the bias model was trained. This is crucial
            for the effective factorization of bias from biological signals.
        ii. **Performance**: Accessing a single file for the large number of
            non-peak regions is vastly more efficient than performing dynamic
            pairing, significantly reducing I/O overhead.

    2.  **Reproducible Validation for Scientific Rigor**:
        To ensure scientifically valid and reproducible model evaluation,
        validation and test data are generated deterministically. This is
        achieved by:

        a) **Fixed Representative Pairing**: Validation peaks are paired with a
        fixed set of "global representative" pseudobulks. These representatives
        are selected deterministically (based on total signal and ID) and,
        critically, are kept consistent across all cross-validation folds. This
        allows for fair and unbiased comparison of model performance across
        different folds.

        b) **Disabled Augmentation**: All stochastic data augmentations (jitter,
        reverse complement, and shuffling) are strictly disabled for validation
        and test modes. This guarantees that the evaluation dataset is identical
        for every run, a cornerstone of reproducible research.

    3.  **Deterministic Reproducibility**:
        All random operations are controlled by independent seed management to
        ensure complete reproducibility across runs, regardless of global random
        state or execution timing.
    """
    
    def __init__(self, 
                 peak_regions: Optional[pd.DataFrame], 
                 nonpeak_regions: Optional[pd.DataFrame],
                 pseudobulk_metadata_path: str,
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
                 seed: Optional[int] = None):
        """
        Initialize the DP Generator with hybrid data loading strategy.
        
        Args:
            peak_regions: DataFrame with peak region coordinates (for weighted DP)
            nonpeak_regions: DataFrame with non-peak region coordinates (uses aggregated BigWig)
            pseudobulk_metadata_path: Path to pseudobulk_metadata.tsv file
            aggregated_bigwig_path: Path to aggregated pseudobulk BigWig (from Stage 05.3)
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
        
        # CRITICAL: Mode-specific augmentation (following ChromBPNetBatchGenerator)
        self.mode = mode
        self.max_jitter = max_jitter if mode == 'train' else 0
        self.add_revcomp = add_revcomp if mode == 'train' else False
        self.shuffle_at_epoch_start = shuffle_at_epoch_start if mode == 'train' else False
        
        # CRITICAL: Independent seed management for deterministic reproducibility
        self.base_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.rng = np.random.RandomState(self.base_seed)
        self.current_epoch = 0
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log seed information for reproducibility
        self.logger.info(f"DPGenerator initialized with base seed: {self.base_seed}")
        self.logger.info(f"Mode: {mode}, Reproducible: {'Yes' if seed is not None else 'No'}")
        
        # Initialize performance optimization components (Step 3)
        self.bigwig_pool = BigWigConnectionPool(max_connections=10)
        self.epoch_planner = EpochDataPlanner(memory_limit_gb=2.0)
        self.performance_stats = {
            'total_io_time': 0.0,
            'total_processing_time': 0.0,
            'epochs_processed': 0
        }
        
        # Initialize data components
        self._initialize_validation_and_metadata(pseudobulk_metadata_path)
        self._initialize_data_sources(aggregated_bigwig_path)
        
        # Initialize epoch data
        self._initialize_epoch_data()
        
    def _initialize_validation_and_metadata(self, pseudobulk_metadata_path: str):
        """
        Load and validate pseudobulk metadata with comprehensive error checking.
        
        CRITICAL: This implements mandatory validation from Step1-1 - no training
        proceeds if any files are missing or corrupted.
        """
        self.logger.info("Loading pseudobulk metadata...")
        
        # Load metadata with comprehensive validation
        metadata_list = load_pseudobulk_metadata(pseudobulk_metadata_path)
        
        # MANDATORY: Validate all BigWig files before training
        self.logger.info("Validating pseudobulk BigWig files...")
        validation_report = validate_and_enforce_file_integrity(metadata_list)
        
        # Convert to DataFrame for easier manipulation
        self.pseudobulk_metadata = pd.DataFrame([
            {
                'pseudobulk_id': meta.pseudobulk_id,
                'cell_type': meta.cell_type,
                'sampling_weight': meta.sampling_weight,
                'bigwig_path': meta.bigwig_path,
                'total_signal': meta.total_signal,
                'file_exists': meta.file_exists
            }
            for meta in metadata_list
        ])
        
        # CRITICAL: Global representative selection for cross-validation consistency
        self._select_global_representatives()
        
    def _select_global_representatives(self):
        """
        Select representative pseudobulks for each cell type using fold-independent 
        deterministic criteria to ensure strict cross-validation consistency.
        
        CRITICAL: Same representatives must be used across all folds (0,1,2,3,4)
        for proper cross-validation comparison.
        """
        representatives = {}
        
        for cell_type in self.pseudobulk_metadata['cell_type'].unique():
            cell_type_data = self.pseudobulk_metadata[
                self.pseudobulk_metadata['cell_type'] == cell_type
            ].copy()
            
            # Deterministic selection: highest total_signal with lexicographic tiebreaker
            representative_idx = cell_type_data.sort_values(
                ['total_signal', 'pseudobulk_id'], 
                ascending=[False, True]
            ).iloc[0].name
            
            representative = self.pseudobulk_metadata.loc[representative_idx]
            representatives[cell_type] = representative
            
            self.logger.info(
                f"Selected global representative for {cell_type}: "
                f"{representative['pseudobulk_id']} (signal: {representative['total_signal']:.2f})"
            )
        
        self.global_representatives = pd.DataFrame(representatives).T
        
    def _initialize_data_sources(self, aggregated_bigwig_path: str):
        """
        Initialize data source handlers for hybrid loading strategy.
        """
        # Store aggregated BigWig path for non-peak regions
        self.aggregated_bigwig_path = aggregated_bigwig_path
        
        # Validate aggregated BigWig existence
        if not Path(aggregated_bigwig_path).exists():
            raise FileNotFoundError(
                f"Aggregated BigWig file not found: {aggregated_bigwig_path}. "
                f"Please run Stage 05.3 (shared bias model training) first."
            )
        
        self.logger.info(f"Using aggregated BigWig for non-peak regions: {aggregated_bigwig_path}")
        
    def _initialize_epoch_data(self):
        """
        Initialize data for the first epoch.
        """
        self.logger.info("Initializing epoch data...")
        
        # Load non-peak data from aggregated BigWig (efficient, single source)
        self._load_nonpeak_data_from_aggregated()
        
        # Initialize peak data based on mode
        if self.mode == 'train':
            self._plan_training_epoch()
        else:  # 'valid' or 'test'
            self._create_fixed_validation_set()
            
    def _load_nonpeak_data_from_aggregated(self):
        """
        Load non-peak region data from the single aggregated BigWig file.
        
        This provides maximum I/O efficiency for non-peak regions (no dynamic pairing needed).
        """
        if self.nonpeak_regions is None or len(self.nonpeak_regions) == 0:
            self.nonpeak_seqs = None
            self.nonpeak_cts = None
            self.nonpeak_coords = None
            return
            
        self.logger.info(f"Loading {len(self.nonpeak_regions)} non-peak regions from aggregated BigWig...")
        
        # Use standard ChromBPNet data loading for non-peak regions
        # Note: No jitter applied to non-peak regions (consistent with standard ChromBPNet)
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
            
        self.logger.info(
            f"Loaded non-peak data: {self.nonpeak_seqs.shape[0] if self.nonpeak_seqs is not None else 0} regions"
        )
    
    def _plan_training_epoch(self):
        """
        Plan training epoch with weighted dynamic pairing for peak regions.
        
        STEP 3: Enhanced with parallel processing and memory optimization.
        Each epoch generates new (peak, pseudobulk) pairs based on sampling_weight.
        """
        if self.peak_regions is None or len(self.peak_regions) == 0:
            self.peak_seqs = None
            self.peak_cts = None  
            self.peak_coords = None
            return
            
        epoch_start_time = time.time()
        self.logger.info("Planning optimized training epoch with weighted dynamic pairing...")
        
        # Step 1: Generate probabilistic peak-pseudobulk pairings for this epoch
        self.epoch_peak_pseudobulk_pairs = self._generate_weighted_peak_pseudobulk_pairs()
        
        # Step 2: Plan efficient parallel data loading (Step 3 optimization)
        self.data_plan = self.epoch_planner.plan_peak_data_loading(
            self.epoch_peak_pseudobulk_pairs,
            self.inputlen + 2 * self.max_jitter,  # Allow for jittering
            self.outputlen + 2 * self.max_jitter,
            self.genome_fasta
        )
        
        # Step 3: Load peak data using optimized parallel strategy
        self._load_peak_data_with_parallel_processing()
        
        # Update performance statistics
        epoch_time = time.time() - epoch_start_time
        self.performance_stats['total_processing_time'] += epoch_time
        self.performance_stats['epochs_processed'] += 1
        
        self.logger.info(f"Epoch planning completed in {epoch_time:.2f}s")
        
    def _create_fixed_validation_set(self):
        """
        Create fixed validation set with global representatives for reproducibility.
        
        STEP 2-2: Implement fixed validation pairing using global representatives.
        CRITICAL: Same representatives across all folds for consistent cross-validation.
        """
        if self.peak_regions is None or len(self.peak_regions) == 0:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
            return
            
        self.logger.info("Creating fixed validation set with global representatives...")
        
        # Step 1: Create fixed validation pairs using global representatives
        self.fixed_validation_pairs = self._generate_fixed_validation_pairs()
        
        # Step 2: Group pairs by BigWig file for efficient I/O (same as training)
        self.file_grouped_pairs = self._group_pairs_by_bigwig_file(self.fixed_validation_pairs)
        
        # Step 3: Load validation data (reuse weighted loading logic)
        self._load_validation_data_with_fixed_pairs()
        
    def _generate_fixed_validation_pairs(self):
        """
        Generate fixed (peak, representative_pseudobulk) pairs for validation.
        
        STEP 2-2: Fixed validation pairing with global representatives.
        CRITICAL: No shuffling, no randomization - fully reproducible.
        """
        validation_pairs = []
        
        # Use deterministic pairing: each peak paired with all global representatives
        # For efficiency, we cycle through representatives
        representative_list = list(self.global_representatives.iterrows())
        
        for peak_idx in range(len(self.peak_regions)):
            # Cycle through representatives deterministically
            representative_idx = peak_idx % len(representative_list)
            _, representative_data = representative_list[representative_idx]
            
            validation_pairs.append((peak_idx, representative_data))
        
        self.logger.info(f"Generated {len(validation_pairs)} fixed validation pairs")
        self.logger.info(f"Using {len(representative_list)} global representatives")
        
        return validation_pairs
    
    def _load_validation_data_with_fixed_pairs(self):
        """
        Load validation data using fixed representative pairings.
        
        STEP 2-2: Validation data loading with global representatives.
        CRITICAL: No jitter, no augmentation - reproducible evaluation data.
        """
        self.logger.info("Loading validation data with fixed representative pairs...")
        
        # Initialize containers to maintain original pair ordering
        total_pairs = len(self.fixed_validation_pairs)
        peak_data_map = {}
        
        genome = pyfaidx.Fasta(self.genome_fasta)
        
        try:
            # Process each BigWig file group sequentially for efficient I/O
            for bigwig_path, pairs in self.file_grouped_pairs.items():
                self.logger.debug(f"Processing {len(pairs)} validation peaks from {Path(bigwig_path).name}")
                
                # Extract peak regions (no jitter for validation)
                file_peak_indices = [pair[0] for pair in pairs]
                file_peak_regions = self.peak_regions.iloc[file_peak_indices]
                
                # Load validation data (no jitter - exact regions only)
                cts_bw = pyBigWig.open(bigwig_path)
                try:
                    group_seqs, group_cts, group_coords = data_utils.get_seq_cts_coords(
                        file_peak_regions,
                        genome,
                        cts_bw,
                        self.inputlen,    # No jitter for validation
                        self.outputlen,   # No jitter for validation  
                        peaks_bool=1
                    )
                    
                    # Map back to original pair positions
                    for local_idx, (original_peak_idx, _) in enumerate(pairs):
                        original_pair_idx = next(
                            i for i, (peak_idx, _) in enumerate(self.fixed_validation_pairs)
                            if peak_idx == original_peak_idx
                        )
                        
                        peak_data_map[original_pair_idx] = (
                            group_seqs[local_idx],
                            group_cts[local_idx],
                            group_coords[local_idx]
                        )
                    
                finally:
                    cts_bw.close()
        
        finally:
            genome.close()
        
        # Reconstruct arrays in original pair order
        if peak_data_map:
            ordered_seqs = []
            ordered_cts = []
            ordered_coords = []
            
            for pair_idx in range(total_pairs):
                if pair_idx in peak_data_map:
                    seq, cts, coord = peak_data_map[pair_idx]
                    ordered_seqs.append(seq)
                    ordered_cts.append(cts)
                    ordered_coords.append(coord)
            
            self.peak_seqs = np.array(ordered_seqs)
            self.peak_cts = np.array(ordered_cts)
            self.peak_coords = np.array(ordered_coords)
        else:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
        
        self.logger.info(
            f"Loaded fixed validation data: {self.peak_seqs.shape[0] if self.peak_seqs is not None else 0} regions"
        )
        

    
    def _generate_weighted_peak_pseudobulk_pairs(self):
        """
        Generate probabilistic (peak, pseudobulk) pairs for the current epoch.
        
        STEP 2-2: Core weighted dynamic pairing implementation.
        CRITICAL: Uses independent RandomState for deterministic reproducibility.
        
        Returns:
            List[Tuple]: List of (peak_index, pseudobulk_metadata) pairs
        """
        self.logger.info(f"Generating weighted pairs for {len(self.peak_regions)} peaks...")
        
        # Set deterministic epoch-specific seed
        epoch_seed = self.base_seed + self.current_epoch * 997  # Use prime number for better distribution
        epoch_rng = np.random.RandomState(epoch_seed)
        
        self.logger.debug(f"Epoch {self.current_epoch} using seed: {epoch_seed}")
        
        # Shuffle peaks for this epoch using controlled randomness
        shuffled_peak_indices = epoch_rng.permutation(len(self.peak_regions))
        
        pairs = []
        # Pre-compute cumulative weights for efficient weighted sampling
        weights = self.pseudobulk_metadata['sampling_weight'].values
        cumulative_weights = np.cumsum(weights)
        total_weight = cumulative_weights[-1]
        
        for peak_idx in shuffled_peak_indices:
            # Deterministic weighted sampling using controlled randomness
            random_value = epoch_rng.random() * total_weight
            selected_idx = np.searchsorted(cumulative_weights, random_value)
            
            # Ensure index is within bounds
            selected_idx = min(selected_idx, len(self.pseudobulk_metadata) - 1)
            selected_pseudobulk = self.pseudobulk_metadata.iloc[selected_idx]
            
            pairs.append((peak_idx, selected_pseudobulk))
        
        self.logger.info(f"Generated {len(pairs)} weighted peak-pseudobulk pairs")
        
        # Log sampling statistics for verification
        if self.logger.isEnabledFor(logging.DEBUG):
            cell_type_counts = {}
            for _, pseudobulk_data in pairs:
                cell_type = pseudobulk_data['cell_type']
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            
            self.logger.debug("Epoch sampling statistics:")
            for cell_type, count in sorted(cell_type_counts.items()):
                percentage = (count / len(pairs)) * 100
                self.logger.debug(f"  {cell_type}: {count} ({percentage:.1f}%)")
        
        return pairs
    
    def _group_pairs_by_bigwig_file(self, pairs):
        """
        Group (peak, pseudobulk) pairs by BigWig file for efficient I/O.
        
        STEP 2-2: File-grouped loading optimization.
        
        Args:
            pairs: List of (peak_index, pseudobulk_metadata) pairs
            
        Returns:
            Dict: {bigwig_path: [(peak_index, pseudobulk_metadata), ...]}
        """
        file_groups = {}
        
        for peak_idx, pseudobulk_metadata in pairs:
            bigwig_path = str(pseudobulk_metadata['bigwig_path'])
            
            if bigwig_path not in file_groups:
                file_groups[bigwig_path] = []
            
            file_groups[bigwig_path].append((peak_idx, pseudobulk_metadata))
        
        self.logger.info(f"Grouped pairs into {len(file_groups)} BigWig files")
        for bigwig_path, group_pairs in file_groups.items():
            self.logger.debug(f"  {Path(bigwig_path).name}: {len(group_pairs)} peaks")
        
        return file_groups
    
    def _load_peak_data_with_parallel_processing(self):
        """
        Load peak data using optimized parallel processing and BigWig connection pooling.
        
        STEP 3: High-performance implementation with memory optimization.
        CRITICAL: Maintains original pair ordering for consistent batch generation.
        """
        io_start_time = time.time()
        self.logger.info("Loading peak data with parallel processing optimization...")
        
        total_pairs = len(self.epoch_peak_pseudobulk_pairs)
        file_groups = self.data_plan['file_groups']
        max_workers = self.data_plan['max_workers']
        
        # Estimate memory usage
        estimated_memory = self.epoch_planner.estimate_memory_usage(
            total_pairs, self.inputlen, self.outputlen
        )
        self.logger.info(f"Estimated memory usage: {estimated_memory:.2f}GB")
        
        if estimated_memory > self.epoch_planner.memory_limit_gb:
            self.logger.warning(f"Estimated memory usage exceeds limit, using sequential processing")
            self._load_peak_data_sequential_fallback()
            return
        
        # Prepare worker arguments for parallel processing
        worker_args_list = []
        for bigwig_path, pairs in file_groups.items():
            peak_indices = [pair[0] for pair in pairs]
            file_peak_regions = self.peak_regions.iloc[peak_indices]
            
            worker_args_list.append((
                peak_indices,
                file_peak_regions,
                self.inputlen + 2 * self.max_jitter,
                self.outputlen + 2 * self.max_jitter,
                bigwig_path
            ))
        
        # Process BigWig files in parallel
        all_results = []
        if len(worker_args_list) == 1 or max_workers == 1:
            # Sequential processing for single file or single worker
            self.logger.info("Using sequential processing")
            for worker_args in worker_args_list:
                # Initialize worker for sequential processing
                bigwig_path = worker_args[4]
                worker_init_bigwig_batch(self.genome_fasta, bigwig_path)
                results = process_peaks_batch_worker(worker_args)
                all_results.extend(results)
        else:
            # Parallel processing for multiple files
            self.logger.info(f"Using parallel processing with {max_workers} workers")
            try:
                with Pool(processes=max_workers, 
                         initializer=worker_init_bigwig_batch,
                         initargs=(self.genome_fasta, None)) as pool:
                    
                    # Use imap for progress tracking
                    results_iter = pool.imap(process_peaks_batch_worker, worker_args_list)
                    
                    # Collect results with progress tracking
                    for file_results in tqdm(results_iter, 
                                           total=len(worker_args_list),
                                           desc="Processing BigWig files"):
                        all_results.extend(file_results)
            
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}")
                self.logger.info("Falling back to sequential processing")
                self._load_peak_data_sequential_fallback()
                return
        
        # Reorganize results maintaining original pair order
        self._reorganize_parallel_results(all_results, total_pairs)
        
        # Track I/O performance
        io_time = time.time() - io_start_time
        self.performance_stats['total_io_time'] += io_time
        
        self.logger.info(
            f"Loaded optimized peak data: {self.peak_seqs.shape[0] if self.peak_seqs is not None else 0} regions in {io_time:.2f}s"
        )
    
    def _load_peak_data_sequential_fallback(self):
        """
        Fallback to sequential processing when parallel processing fails or memory is insufficient.
        Uses the original file-grouped approach with BigWig connection pooling.
        """
        self.logger.info("Using sequential fallback with connection pooling...")
        
        total_pairs = len(self.epoch_peak_pseudobulk_pairs)
        peak_data_map = {}  # {original_pair_index: (seqs, cts, coords)}
        
        # Group pairs by BigWig file (if not already available)
        if not hasattr(self, 'file_grouped_pairs'):
            self.file_grouped_pairs = self._group_pairs_by_bigwig_file(self.epoch_peak_pseudobulk_pairs)
        
        genome = pyfaidx.Fasta(self.genome_fasta)
        
        try:
            # Process each BigWig file group with connection pooling
            for bigwig_path, pairs in self.file_grouped_pairs.items():
                self.logger.debug(f"Processing {len(pairs)} peaks from {Path(bigwig_path).name}")
                
                # Extract peak regions and track their original positions
                file_peak_indices = [pair[0] for pair in pairs]
                file_peak_regions = self.peak_regions.iloc[file_peak_indices]
                
                # Use connection pool for BigWig access
                cts_bw = self.bigwig_pool.get_connection(bigwig_path)
                
                group_seqs, group_cts, group_coords = data_utils.get_seq_cts_coords(
                    file_peak_regions,
                    genome,
                    cts_bw,
                    self.inputlen + 2 * self.max_jitter,  # Allow for jittering
                    self.outputlen + 2 * self.max_jitter,
                    peaks_bool=1  # Peak regions
                )
                
                # Map back to original pair positions
                for local_idx, (original_peak_idx, _) in enumerate(pairs):
                    # Find the original position in epoch_peak_pseudobulk_pairs
                    original_pair_idx = next(
                        i for i, (peak_idx, _) in enumerate(self.epoch_peak_pseudobulk_pairs)
                        if peak_idx == original_peak_idx
                    )
                    
                    peak_data_map[original_pair_idx] = (
                        group_seqs[local_idx],
                        group_cts[local_idx],
                        group_coords[local_idx]
                    )
        
        finally:
            genome.close()
        
        # Reconstruct arrays in original pair order
        self._build_arrays_from_data_map(peak_data_map, total_pairs)
    
    def _reorganize_parallel_results(self, all_results: List[Dict], total_pairs: int):
        """
        Reorganize parallel processing results maintaining original pair order.
        """
        peak_data_map = {}
        
        # Map results back to original pair indices
        for result in all_results:
            original_peak_idx = result['original_peak_idx']
            
            # Find the original position in epoch_peak_pseudobulk_pairs
            original_pair_idx = next(
                i for i, (peak_idx, _) in enumerate(self.epoch_peak_pseudobulk_pairs)
                if peak_idx == original_peak_idx
            )
            
            peak_data_map[original_pair_idx] = (
                result['sequence'],
                result['counts'],
                result['coords']
            )
        
        # Build final arrays
        self._build_arrays_from_data_map(peak_data_map, total_pairs)
    
    def _build_arrays_from_data_map(self, peak_data_map: Dict, total_pairs: int):
        """
        Build final arrays from data map maintaining pair order.
        """
        if peak_data_map:
            ordered_seqs = []
            ordered_cts = []
            ordered_coords = []
            
            for pair_idx in range(total_pairs):
                if pair_idx in peak_data_map:
                    seq, cts, coord = peak_data_map[pair_idx]
                    ordered_seqs.append(seq)
                    ordered_cts.append(cts)
                    ordered_coords.append(coord)
            
            self.peak_seqs = np.array(ordered_seqs)
            self.peak_cts = np.array(ordered_cts)  
            self.peak_coords = np.array(ordered_coords)
        else:
            self.peak_seqs = None
            self.peak_cts = None
            self.peak_coords = None
        
    def _crop_revcomp_data(self):
        """
        Apply cropping, reverse complement, and other augmentations following
        ChromBPNetBatchGenerator patterns with mode-specific behavior.
        
        CRITICAL: This implements the same augmentation logic as the original
        ChromBPNetBatchGenerator to ensure consistency.
        """
        # Combine peak and non-peak data (following ChromBPNetBatchGenerator logic)
        if (self.peak_seqs is not None) and (self.nonpeak_seqs is not None):
            # Apply random crop to peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords
            )
            
            # Handle negative sampling ratio
            if self.negative_sampling_ratio < 1.0:
                sampled_nonpeak_seqs, sampled_nonpeak_cts, sampled_nonpeak_coords = self._subsample_nonpeak_data(
                    self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords, 
                    len(self.peak_seqs), self.negative_sampling_ratio
                )
                self.seqs = np.vstack([cropped_peaks, sampled_nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, sampled_nonpeak_cts]) 
                self.coords = np.vstack([cropped_coords, sampled_nonpeak_coords])
            else:
                self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.nonpeak_coords])
                
        elif self.peak_seqs is not None:
            # Only peak data
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(
                self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords
            )
            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords
            
        elif self.nonpeak_seqs is not None:
            # Only non-peak data
            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
            self.coords = self.nonpeak_coords
        else:
            raise ValueError("Both peak and non-peak arrays are empty")
            
        # Apply final augmentation (crop, reverse complement, shuffle)
        # CRITICAL: Mode-specific augmentation behavior
        self.cur_seqs, self.cur_cts, self.cur_coords = augment.crop_revcomp_augment(
            self.seqs, self.cts, self.coords, self.inputlen, self.outputlen,
            self.add_revcomp, shuffle=self.shuffle_at_epoch_start
        )
        
    def _subsample_nonpeak_data(self, nonpeak_seqs, nonpeak_cts, nonpeak_coords, 
                               peak_data_size, negative_sampling_ratio):
        """
        Randomly sample a portion of non-peak data (following ChromBPNetBatchGenerator).
        CRITICAL: Uses epoch-specific seed for deterministic reproducibility.
        """
        num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
        
        # Use epoch-specific seed for deterministic sampling
        epoch_seed = self.base_seed + self.current_epoch * 997 + 1  # Offset by 1 from main sampling
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
        return math.ceil(self.seqs.shape[0] / self.batch_size)
    
    def __getitem__(self, idx):
        """
        Get batch data with mode-specific augmentation behavior.
        
        CRITICAL: Follows exact same output format as ChromBPNetBatchGenerator
        for seamless integration.
        """
        batch_seq = self.cur_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_cts = self.cur_cts[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_coords = self.cur_coords[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.return_coords:
            return (batch_seq, (batch_cts, np.log(1 + batch_cts.sum(-1, keepdims=True))), batch_coords)
        else:
            return (batch_seq, (batch_cts, np.log(1 + batch_cts.sum(-1, keepdims=True))))
    
    def on_epoch_end(self):
        """
        End-of-epoch processing with mode-specific behavior and performance optimization.
        
        STEP 3: Enhanced with performance monitoring and resource management.
        CRITICAL: Follows ChromBPNetBatchGenerator pattern while adding DP functionality.
        CRITICAL: Updates epoch counter for deterministic seed progression.
        """
        epoch_start = time.time()
        
        # CRITICAL: Increment epoch counter for deterministic reproducibility
        self.current_epoch += 1
        self.logger.debug(f"Starting epoch {self.current_epoch}")
        
        if self.mode == 'train':
            # Re-plan epoch for training with optimization (Step 3)
            self._plan_training_epoch()
        # For validation/test, use fixed data (no re-planning needed)
        
        # Apply augmentation processing
        self._crop_revcomp_data()
        
        # Log performance statistics
        epoch_time = time.time() - epoch_start
        avg_processing_time = (self.performance_stats['total_processing_time'] / 
                              max(1, self.performance_stats['epochs_processed']))
        avg_io_time = (self.performance_stats['total_io_time'] / 
                      max(1, self.performance_stats['epochs_processed']))
        
        self.logger.debug(
            f"Epoch {self.current_epoch} processing: {epoch_time:.2f}s, "
            f"Avg I/O time: {avg_io_time:.2f}s, "
            f"Avg processing time: {avg_processing_time:.2f}s"
        )
        
        # Periodic memory cleanup (every 10 epochs)
        if self.performance_stats['epochs_processed'] % 10 == 0:
            self._cleanup_resources()
    
    def _cleanup_resources(self):
        """
        Periodic resource cleanup to prevent memory leaks.
        STEP 3: Resource management following ChromBPNet patterns.
        """
        self.logger.debug("Performing periodic resource cleanup...")
        
        # Clean up old BigWig connections
        if hasattr(self, 'bigwig_pool'):
            self.bigwig_pool._cleanup_old_connections()
        
        # Force garbage collection for cached data
        gc.collect()
        
        # Log memory usage
        memory_usage = psutil.Process().memory_info().rss / (1024**3)  # GB
        self.logger.debug(f"Current memory usage: {memory_usage:.2f}GB")
    
    def get_reproducibility_info(self) -> dict:
        """
        Return complete reproducibility information for logging and verification.
        
        Returns:
            dict: Comprehensive seed and state information
        """
        return {
            'base_seed': self.base_seed,
            'current_epoch': self.current_epoch,
            'mode': self.mode,
            'current_epoch_seed': self.base_seed + self.current_epoch * 997,
            'generator_class': self.__class__.__name__,
            'reproducible': True,
            'seed_formula': 'base_seed + current_epoch * 997'
        }
    
    def verify_reproducibility(self, other_generator) -> bool:
        """
        Verify that this generator will produce the same results as another generator.
        
        Args:
            other_generator: Another DPGenerator instance
            
        Returns:
            bool: True if generators should produce identical results
        """
        if not isinstance(other_generator, DPGenerator):
            return False
        
        return (self.base_seed == other_generator.base_seed and
                self.current_epoch == other_generator.current_epoch and
                self.mode == other_generator.mode)
    
    def close(self):
        """
        Clean up resources when generator is closed.
        STEP 3: Proper resource management with reproducibility logging.
        """
        # Log final reproducibility information
        repro_info = self.get_reproducibility_info()
        self.logger.info("Final reproducibility state:")
        for key, value in repro_info.items():
            self.logger.info(f"  {key}: {value}")
        
        if hasattr(self, 'bigwig_pool'):
            self.bigwig_pool.close_all()
            self.logger.info("Closed all BigWig connections")
        
        # Log final performance statistics
        if self.performance_stats['epochs_processed'] > 0:
            avg_processing_time = (self.performance_stats['total_processing_time'] / 
                                  self.performance_stats['epochs_processed'])
            avg_io_time = (self.performance_stats['total_io_time'] / 
                          self.performance_stats['epochs_processed'])
            
            self.logger.info(f"Final performance statistics:")
            self.logger.info(f"  Total epochs processed: {self.performance_stats['epochs_processed']}")
            self.logger.info(f"  Average I/O time per epoch: {avg_io_time:.2f}s")
            self.logger.info(f"  Average processing time per epoch: {avg_processing_time:.2f}s")
