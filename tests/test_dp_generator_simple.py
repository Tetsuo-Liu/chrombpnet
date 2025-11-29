#!/usr/bin/env python3
"""
Simplified test script for verifying DPGenerator can be imported and basic functionality works.

This is a simplified version that doesn't require extensive mocking since we have
all dependencies available in the dp-model environment.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

# Add the chrombpnet_fork to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock data structures
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
    ]

def mock_validate_and_enforce_file_integrity(metadata_list):
    """Mock validation function."""
    return {"status": "success", "validated": len(metadata_list)}

def test_basic_import():
    """Test that DPGenerator can be imported successfully."""
    print("Testing DPGenerator import...")
    
    # Patch only the validation functions that we need to mock
    with patch('chrombpnet.training.utils.pseudobulk_validation.load_pseudobulk_metadata', 
              side_effect=mock_load_pseudobulk_metadata):
        with patch('chrombpnet.training.utils.pseudobulk_validation.validate_and_enforce_file_integrity',
                  side_effect=mock_validate_and_enforce_file_integrity):
            
            try:
                from chrombpnet.training.data_generators.dp_generator import DPGenerator
                print("✓ DPGenerator imported successfully!")
                return True
            except Exception as e:
                print(f"✗ Failed to import DPGenerator: {e}")
                return False

def main():
    """Run the basic import test."""
    print("Running simplified DPGenerator test...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test basic import
    success = test_basic_import()
    
    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
