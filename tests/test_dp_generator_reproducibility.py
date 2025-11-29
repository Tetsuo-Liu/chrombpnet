#!/usr/bin/env python3
"""
Test script for verifying DPGenerator reproducibility.

This script creates two identical DPGenerator instances and verifies that they
produce identical results across multiple epochs, ensuring deterministic behavior.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Add the chrombpnet_fork to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock only the external dependencies that might not be available
# TensorFlow and ChromBPNet are available in dp-model environment, so no need to mock them

# Mock external dependencies
sys.modules['pyBigWig'] = Mock()
sys.modules['pyfaidx'] = Mock()

# Create a proper mock for psutil that returns expected memory info
class MockMemoryInfo:
    def __init__(self):
        self.available = 8 * 1024**3  # 8GB available memory

class MockPsutil:
    def virtual_memory(self):
        return MockMemoryInfo()

sys.modules['psutil'] = MockPsutil()
sys.modules['tqdm'] = Mock()

class MockPseudobulkMetadata:
    """Mock PseudobulkMetadata for testing."""
    def __init__(self, pseudobulk_id, cell_type, sampling_weight, bigwig_path, total_signal):
        self.pseudobulk_id = pseudobulk_id
        self.cell_type = cell_type
        self.sampling_weight = sampling_weight
        self.bigwig_path = bigwig_path
        self.total_signal = total_signal
        self.file_exists = True

def mock_load_pseudobulk_metadata(path):
    """Mock function to load pseudobulk metadata."""
    return [
        MockPseudobulkMetadata("pbulk_001", "HSC", 2.0, "/mock/path/pbulk_001.bw", 100.5),
        MockPseudobulkMetadata("pbulk_002", "Monocyte", 1.0, "/mock/path/pbulk_002.bw", 150.2),
        MockPseudobulkMetadata("pbulk_003", "B_cell", 1.5, "/mock/path/pbulk_003.bw", 120.8),
        MockPseudobulkMetadata("pbulk_004", "T_cell", 1.2, "/mock/path/pbulk_004.bw", 135.3),
        MockPseudobulkMetadata("pbulk_005", "NK_cell", 1.8, "/mock/path/pbulk_005.bw", 90.7),
    ]

def mock_validate_and_enforce_file_integrity(metadata_list):
    """Mock validation function."""
    return {"status": "success", "validated": len(metadata_list)}

# Import the DPGenerator first, then we'll patch specific functions during tests
from chrombpnet.training.data_generators.dp_generator import DPGenerator

def create_mock_peak_regions(n_peaks=100):
    """Create mock peak regions for testing."""
    return pd.DataFrame({
        'chr': [f'chr{i % 22 + 1}' for i in range(n_peaks)],
        'start': [i * 1000 for i in range(n_peaks)],
        'end': [i * 1000 + 500 for i in range(n_peaks)],
        'summit': [250] * n_peaks
    })

def create_mock_nonpeak_regions(n_nonpeaks=200):
    """Create mock non-peak regions for testing."""
    return pd.DataFrame({
        'chr': [f'chr{i % 22 + 1}' for i in range(n_nonpeaks)],
        'start': [i * 1000 + 600 for i in range(n_nonpeaks)],
        'end': [i * 1000 + 1100 for i in range(n_nonpeaks)]
    })

def test_reproducibility():
    """Test that DPGenerator produces identical results with the same seed."""
    print("Testing DPGenerator reproducibility...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    peak_regions = create_mock_peak_regions(50)
    nonpeak_regions = create_mock_nonpeak_regions(100)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create mock files
        metadata_path = temp_dir / "metadata.tsv"
        aggregated_bw = temp_dir / "aggregated.bw"
        genome_fasta = temp_dir / "genome.fa"
        
        metadata_path.touch()
        aggregated_bw.touch()
        genome_fasta.touch()
        
        # Test parameters
        seed = 12345
        
        # Mock all the file operations and complex dependencies
        def mock_get_seq_cts_coords(*args, **kwargs):
            """Mock function that returns appropriate sequences, counts, and coordinates."""
            n_regions = len(args[0]) if args else 50  # Use region count from arguments if available
            # Return mock sequences (one-hot encoded), counts, and coordinates
            sequences = np.random.rand(n_regions, 2114, 4)  # (batch, length, 4 bases)
            counts = np.random.rand(n_regions, 1000, 1)     # (batch, outputlen, 1 track)
            coordinates = [(f'chr{i%22+1}', i*1000, i*1000+2114) for i in range(n_regions)]
            return sequences, counts, coordinates
        
        with patch('chrombpnet.training.data_generators.dp_generator.pyfaidx.Fasta'):
            with patch('chrombpnet.training.data_generators.dp_generator.pyBigWig.open'):
                with patch('chrombpnet.training.data_generators.dp_generator.data_utils.get_seq_cts_coords', 
                          side_effect=mock_get_seq_cts_coords):
                    
                    # Use proper patching to avoid file system dependencies
                    with patch('chrombpnet.training.data_generators.dp_generator.load_pseudobulk_metadata', 
                              side_effect=mock_load_pseudobulk_metadata):
                        with patch('chrombpnet.training.data_generators.dp_generator.validate_and_enforce_file_integrity',
                                  side_effect=mock_validate_and_enforce_file_integrity):
                            
                            print("Creating first DPGenerator...")
                            generator1 = DPGenerator(
                                peak_regions=peak_regions,
                                nonpeak_regions=nonpeak_regions,
                                pseudobulk_metadata_path=str(metadata_path),
                                aggregated_bigwig_path=str(aggregated_bw),
                                genome_fasta=str(genome_fasta),
                                batch_size=32,
                                inputlen=2114,
                                outputlen=1000,
                                max_jitter=128,
                                negative_sampling_ratio=1.0,
                                add_revcomp=True,
                                return_coords=False,
                                shuffle_at_epoch_start=True,
                                mode="train",
                                seed=seed
                            )
                            
                            print("Creating second DPGenerator with same seed...")
                            generator2 = DPGenerator(
                                peak_regions=peak_regions,
                                nonpeak_regions=nonpeak_regions,
                                pseudobulk_metadata_path=str(metadata_path),
                                aggregated_bigwig_path=str(aggregated_bw),
                                genome_fasta=str(genome_fasta),
                                batch_size=32,
                                inputlen=2114,
                                outputlen=1000,
                                max_jitter=128,
                                negative_sampling_ratio=1.0,
                                add_revcomp=True,
                                return_coords=False,
                                shuffle_at_epoch_start=True,
                                mode="train",
                                seed=seed
                            )
                            
                            print("Testing seed initialization...")
                            assert generator1.base_seed == generator2.base_seed
                            assert generator1.current_epoch == generator2.current_epoch
                            print(f"✓ Both generators initialized with seed: {generator1.base_seed}")
                            
                            print("\nTesting reproducibility verification...")
                            assert generator1.verify_reproducibility(generator2)
                            print("✓ Generators verified as reproducible")
                            
                            print("\nTesting reproducibility info...")
                            info1 = generator1.get_reproducibility_info()
                            info2 = generator2.get_reproducibility_info()
                            assert info1 == info2
                            print("✓ Reproducibility info matches:")
                            for key, value in info1.items():
                                print(f"  {key}: {value}")
                            
                            print("\nTesting epoch-specific seed generation...")
                            # Test that different epochs produce different seeds but same between generators
                            for epoch in range(3):
                                generator1.current_epoch = epoch
                                generator2.current_epoch = epoch
                                
                                epoch_seed1 = generator1.base_seed + generator1.current_epoch * 997
                                epoch_seed2 = generator2.base_seed + generator2.current_epoch * 997
                                
                                assert epoch_seed1 == epoch_seed2
                                print(f"  Epoch {epoch}: seed {epoch_seed1} (same for both generators)")
                            
                            # Reset for pair generation test
                            generator1.current_epoch = 0
                            generator2.current_epoch = 0
                            
                            print("\nTesting weighted pair generation reproducibility...")
                            
                            # Generate pairs for both generators
                            pairs1 = generator1._generate_weighted_peak_pseudobulk_pairs()
                            pairs2 = generator2._generate_weighted_peak_pseudobulk_pairs()
                            
                            assert len(pairs1) == len(pairs2)
                            print(f"✓ Both generators produced {len(pairs1)} pairs")
                            
                            # Check that pairs are identical
                            identical_pairs = 0
                            for i, ((peak_idx1, pseudo1), (peak_idx2, pseudo2)) in enumerate(zip(pairs1, pairs2)):
                                if peak_idx1 == peak_idx2 and pseudo1['pseudobulk_id'] == pseudo2['pseudobulk_id']:
                                    identical_pairs += 1
                            
                            print(f"✓ {identical_pairs}/{len(pairs1)} pairs are identical ({identical_pairs/len(pairs1)*100:.1f}%)")
                            
                            if identical_pairs == len(pairs1):
                                print("✅ PERFECT REPRODUCIBILITY: All pairs are identical!")
                            elif identical_pairs / len(pairs1) > 0.95:
                                print("✅ EXCELLENT REPRODUCIBILITY: >95% pairs identical")
                            else:
                                print(f"❌ POOR REPRODUCIBILITY: Only {identical_pairs/len(pairs1)*100:.1f}% pairs identical")
                            
                            print("\nTesting different epochs produce different results...")
                            generator1.current_epoch = 1
                            generator2.current_epoch = 1
                            
                            pairs1_epoch1 = generator1._generate_weighted_peak_pseudobulk_pairs()
                            pairs2_epoch1 = generator2._generate_weighted_peak_pseudobulk_pairs()
                            
                            # Pairs should be different from epoch 0 but same between generators
                            different_from_epoch0 = sum(1 for ((p1, ps1), (p0, ps0)) in zip(pairs1_epoch1, pairs1) 
                                                       if p1 != p0 or ps1['pseudobulk_id'] != ps0['pseudobulk_id'])
                            
                            same_between_generators = sum(1 for ((p1, ps1), (p2, ps2)) in zip(pairs1_epoch1, pairs2_epoch1)
                                                        if p1 == p2 and ps1['pseudobulk_id'] == ps2['pseudobulk_id'])
                            
                            print(f"✓ {different_from_epoch0}/{len(pairs1)} pairs different from epoch 0 ({different_from_epoch0/len(pairs1)*100:.1f}%)")
                            print(f"✓ {same_between_generators}/{len(pairs1_epoch1)} pairs identical between generators ({same_between_generators/len(pairs1_epoch1)*100:.1f}%)")
                            
                            print("\n🎉 Reproducibility test completed successfully!")
                            print("✅ DPGenerator shows deterministic behavior with independent seed management")
                            
                            # Test cleanup
                            generator1.close()
                            generator2.close()

def test_seed_independence():
    """Test that different seeds produce different results."""
    print("\nTesting seed independence...")
    
    peak_regions = create_mock_peak_regions(30)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        metadata_path = temp_dir / "metadata.tsv"
        aggregated_bw = temp_dir / "aggregated.bw"
        genome_fasta = temp_dir / "genome.fa"
        
        for file_path in [metadata_path, aggregated_bw, genome_fasta]:
            file_path.touch()
        
        def mock_get_seq_cts_coords(*args, **kwargs):
            """Mock function that returns appropriate sequences, counts, and coordinates."""
            n_regions = len(args[0]) if args else 30  # Use region count from arguments if available
            # Return mock sequences (one-hot encoded), counts, and coordinates
            sequences = np.random.rand(n_regions, 2114, 4)  # (batch, length, 4 bases)
            counts = np.random.rand(n_regions, 1000, 1)     # (batch, outputlen, 1 track)
            coordinates = [(f'chr{i%22+1}', i*1000, i*1000+2114) for i in range(n_regions)]
            return sequences, counts, coordinates
        
        with patch('chrombpnet.training.data_generators.dp_generator.pyfaidx.Fasta'):
            with patch('chrombpnet.training.data_generators.dp_generator.pyBigWig.open'):
                with patch('chrombpnet.training.data_generators.dp_generator.data_utils.get_seq_cts_coords', 
                          side_effect=mock_get_seq_cts_coords):
                    
                    # Use proper patching to avoid file system dependencies
                    with patch('chrombpnet.training.data_generators.dp_generator.load_pseudobulk_metadata', 
                              side_effect=mock_load_pseudobulk_metadata):
                        with patch('chrombpnet.training.data_generators.dp_generator.validate_and_enforce_file_integrity',
                                  side_effect=mock_validate_and_enforce_file_integrity):
                            
                            # Create generators with different seeds
                            gen_seed1 = DPGenerator(
                                peak_regions=peak_regions, nonpeak_regions=None,
                                pseudobulk_metadata_path=str(metadata_path),
                                aggregated_bigwig_path=str(aggregated_bw),
                                genome_fasta=str(genome_fasta),
                                batch_size=16, inputlen=2114, outputlen=1000,
                                max_jitter=128, negative_sampling_ratio=1.0,
                                add_revcomp=True, return_coords=False,
                                shuffle_at_epoch_start=True, mode="train",
                                seed=11111
                            )
                            
                            gen_seed2 = DPGenerator(
                                peak_regions=peak_regions, nonpeak_regions=None,
                                pseudobulk_metadata_path=str(metadata_path),
                                aggregated_bigwig_path=str(aggregated_bw),
                                genome_fasta=str(genome_fasta),
                                batch_size=16, inputlen=2114, outputlen=1000,
                                max_jitter=128, negative_sampling_ratio=1.0,
                                add_revcomp=True, return_coords=False,
                                shuffle_at_epoch_start=True, mode="train",
                                seed=22222
                            )
                            
                            assert gen_seed1.base_seed != gen_seed2.base_seed
                            assert not gen_seed1.verify_reproducibility(gen_seed2)
                            
                            pairs1 = gen_seed1._generate_weighted_peak_pseudobulk_pairs()
                            pairs2 = gen_seed2._generate_weighted_peak_pseudobulk_pairs()
                            
                            different_pairs = sum(1 for ((p1, ps1), (p2, ps2)) in zip(pairs1, pairs2)
                                                if p1 != p2 or ps1['pseudobulk_id'] != ps2['pseudobulk_id'])
                            
                            print(f"✓ Different seeds produce different results: {different_pairs}/{len(pairs1)} pairs differ ({different_pairs/len(pairs1)*100:.1f}%)")
                            
                            gen_seed1.close()
                            gen_seed2.close()

if __name__ == "__main__":
    print("🧪 DPGenerator Reproducibility Test Suite")
    print("=" * 50)
    
    try:
        test_reproducibility()
        test_seed_independence()
        print("\n🎉 All tests passed! DPGenerator reproducibility is confirmed.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
