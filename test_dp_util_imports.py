#!/usr/bin/env python3
"""
Test script to verify dp_util module imports work correctly after refactoring.
"""

import sys
import os
import traceback

# Add the ChromBPNet root to path
sys.path.insert(0, '/home/tetsuo/shimamura-lab/projects/chrombpnet_fork')

def test_import(module_name, description):
    """Test importing a specific module or component."""
    print(f"Testing {description}...")
    try:
        exec(f"import {module_name}")
        print(f"  ✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import {module_name}: {e}")
        return False

def test_from_import(from_module, import_items, description):
    """Test importing specific items from a module."""
    print(f"Testing {description}...")
    try:
        import_statement = f"from {from_module} import {', '.join(import_items)}"
        exec(import_statement)
        print(f"  ✅ Successfully imported {import_items} from {from_module}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import {import_items} from {from_module}: {e}")
        return False

def main():
    """Run all import tests."""
    print("=" * 60)
    print("DP Util Module Import Testing")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Basic module imports
    tests = [
        ("chrombpnet.training.utils.dp_util.pseudobulk_validation", "Pseudobulk validation module"),
        ("chrombpnet.training.utils.dp_util.bigwig_utils", "BigWig utilities module"),
        ("chrombpnet.training.utils.dp_util.parallel_processing", "Parallel processing module"),
        ("chrombpnet.training.utils.dp_util.epoch_planning", "Epoch planning module"),
    ]
    
    for module, description in tests:
        total_tests += 1
        if test_import(module, description):
            success_count += 1
    
    print("\n" + "=" * 60)
    
    # Test 2: Specific class/function imports
    from_import_tests = [
        ("chrombpnet.training.utils.dp_util.pseudobulk_validation", 
         ["PseudobulkMetadata", "load_pseudobulk_metadata"], 
         "Pseudobulk validation classes"),
        ("chrombpnet.training.utils.dp_util.bigwig_utils", 
         ["BigWigConnectionPool"], 
         "BigWig connection pool"),
        ("chrombpnet.training.utils.dp_util.parallel_processing", 
         ["worker_init_bigwig_batch", "process_peaks_batch_worker"], 
         "Parallel processing functions"),
        ("chrombpnet.training.utils.dp_util.epoch_planning", 
         ["EpochDataPlanner"], 
         "Epoch data planner"),
    ]
    
    for from_module, import_items, description in from_import_tests:
        total_tests += 1
        if test_from_import(from_module, import_items, description):
            success_count += 1
    
    print("\n" + "=" * 60)
    
    # Test 3: dp_util package import
    total_tests += 1
    if test_import("chrombpnet.training.utils.dp_util", "DP util package"):
        success_count += 1
    
    # Test 4: dp_util convenient imports
    total_tests += 1
    if test_from_import("chrombpnet.training.utils.dp_util", 
                       ["PseudobulkMetadata", "BigWigConnectionPool", "EpochDataPlanner"], 
                       "DP util convenient imports"):
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Import Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All imports working correctly!")
        return 0
    else:
        print("⚠️  Some imports failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Test script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

