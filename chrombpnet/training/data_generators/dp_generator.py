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
                 mode: str = "train"):
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
        
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
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
        
        STEP 2-2: Implement probabilistic pseudobulk selection for training.
        Each epoch generates new (peak, pseudobulk) pairs based on sampling_weight.
        """
        if self.peak_regions is None or len(self.peak_regions) == 0:
            self.peak_seqs = None
            self.peak_cts = None  
            self.peak_coords = None
            return
            
        self.logger.info("Planning training epoch with weighted dynamic pairing...")
        
        # Step 1: Generate probabilistic peak-pseudobulk pairings for this epoch
        self.epoch_peak_pseudobulk_pairs = self._generate_weighted_peak_pseudobulk_pairs()
        
        # Step 2: Group pairs by BigWig file for efficient I/O
        self.file_grouped_pairs = self._group_pairs_by_bigwig_file(self.epoch_peak_pseudobulk_pairs)
        
        # Step 3: Load peak data using the weighted pairings
        self._load_peak_data_with_weighting()
        
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
        
        Returns:
            List[Tuple]: List of (peak_index, pseudobulk_metadata) pairs
        """
        self.logger.info(f"Generating weighted pairs for {len(self.peak_regions)} peaks...")
        
        # Shuffle peaks for this epoch (following ChromBPNet pattern)
        shuffled_peak_indices = np.random.permutation(len(self.peak_regions))
        
        pairs = []
        for peak_idx in shuffled_peak_indices:
            # Probabilistically select pseudobulk based on sampling_weight
            selected_pseudobulk = self.pseudobulk_metadata.sample(
                n=1, 
                weights='sampling_weight'
            ).iloc[0]
            
            pairs.append((peak_idx, selected_pseudobulk))
        
        self.logger.info(f"Generated {len(pairs)} weighted peak-pseudobulk pairs")
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
    
    def _load_peak_data_with_weighting(self):
        """
        Load peak data using weighted dynamic pairing with file-grouped I/O.
        
        STEP 2-2: Replace basic loading with weighted sampling implementation.
        CRITICAL: Maintains original pair ordering for consistent batch generation.
        """
        self.logger.info("Loading peak data with weighted dynamic pairing...")
        
        # Initialize containers to maintain original pair ordering
        total_pairs = len(self.epoch_peak_pseudobulk_pairs)
        peak_data_map = {}  # {original_pair_index: (seqs, cts, coords)}
        
        genome = pyfaidx.Fasta(self.genome_fasta)
        
        try:
            # Process each BigWig file group sequentially for efficient I/O
            for bigwig_path, pairs in self.file_grouped_pairs.items():
                self.logger.debug(f"Processing {len(pairs)} peaks from {Path(bigwig_path).name}")
                
                # Extract peak regions and track their original positions
                file_peak_indices = [pair[0] for pair in pairs]
                file_peak_regions = self.peak_regions.iloc[file_peak_indices]
                
                # Load data for this group using single BigWig file
                cts_bw = pyBigWig.open(bigwig_path)
                try:
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
            f"Loaded weighted peak data: {self.peak_seqs.shape[0] if self.peak_seqs is not None else 0} regions"
        )
        
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
        """
        num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
        nonpeak_indices_to_keep = np.random.choice(
            len(nonpeak_seqs), size=num_nonpeak_samples, replace=False
        )
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
        End-of-epoch processing with mode-specific behavior.
        
        CRITICAL: Follows ChromBPNetBatchGenerator pattern while adding DP functionality.
        """
        if self.mode == 'train':
            # Re-plan epoch for training (weighted sampling will be added in Step 2)
            self._plan_training_epoch()
        # For validation/test, use fixed data (no re-planning needed)
        
        # Apply augmentation processing
        self._crop_revcomp_data()
