import os
import subprocess
import filecmp
import pytest

def create_test_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

@pytest.fixture
def setup_test_data(tmp_path):
    """Set up test data files in a temporary directory."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()

    # Create dummy files
    genome_content = ">chr1\n" + "N" * 10000
    peaks_content = "chr1\t1000\t2000\t.\t.\t.\t.\t.\t.\t500\n"
    chrom_sizes_content = "chr1\t10000\n"
    folds_content = '{"train": ["chr2"], "valid": ["chr3"], "test": ["chr1"]}'

    genome_path = test_data_dir / "genome.fa"
    peaks_path = test_data_dir / "peaks.bed"
    chrom_sizes_path = test_data_dir / "chrom.sizes"
    folds_path = test_data_dir / "folds.json"

    create_test_file(genome_path, genome_content)
    create_test_file(peaks_path, peaks_content)
    create_test_file(chrom_sizes_path, chrom_sizes_content)
    create_test_file(folds_path, folds_content)

    return {
        "genome": str(genome_path),
        "peaks": str(peaks_path),
        "chrom_sizes": str(chrom_sizes_path),
        "folds": str(folds_path),
        "tmp_path": str(tmp_path)
    }

def run_prep_nonpeaks(data, output_prefix, precomputed_gc_profile=None):
    """Helper function to run the chrombpnet prep nonpeaks command."""
    cmd = [
        "chrombpnet", "prep", "nonpeaks",
        "-g", data["genome"],
        "-p", data["peaks"],
        "-c", data["chrom_sizes"],
        "-fl", data["folds"],
        "-o", output_prefix,
        "--seed", "1234"
    ]
    if precomputed_gc_profile:
        cmd.extend(["--precomputed-gc-profile", precomputed_gc_profile])

    # Use absolute paths for chrombpnet command if it's not in the system's PATH
    # For this test, we assume it's callable directly.
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"
    return result

def test_prep_nonpeaks_with_precomputed_gc(setup_test_data):
    """
    Tests the --precomputed-gc-profile functionality.
    1. Runs `prep nonpeaks` to generate a GC profile.
    2. Runs `prep nonpeaks` again using the generated profile.
    3. Compares the final negative peak files to ensure they are identical.
    """
    data = setup_test_data
    tmp_path = data["tmp_path"]

    # --- Run 1: Generate the GC profile ---
    output_prefix_1 = os.path.join(tmp_path, "run1_output")
    run_prep_nonpeaks(data, output_prefix_1)

    generated_gc_profile = os.path.join(output_prefix_1 + "_auxiliary", "genomewide_gc.bed")
    negatives_1 = os.path.join(output_prefix_1 + "_negatives.bed")

    assert os.path.exists(generated_gc_profile), "GC profile was not generated in the first run."
    assert os.path.exists(negatives_1), "Negative peaks file was not generated in the first run."

    # --- Run 2: Use the precomputed GC profile ---
    output_prefix_2 = os.path.join(tmp_path, "run2_output")
    run_prep_nonpeaks(data, output_prefix_2, precomputed_gc_profile=generated_gc_profile)
    
    negatives_2 = os.path.join(output_prefix_2 + "_negatives.bed")
    
    assert os.path.exists(negatives_2), "Negative peaks file was not generated in the second run."

    # --- Comparison ---
    # Ensure the final output is identical
    assert filecmp.cmp(negatives_1, negatives_2, shallow=False), \
        "The negative peaks file generated with the precomputed profile is different from the original."

    # Optional: Check logs to be more certain that calculation was skipped.
    # This is more complex as it requires parsing stdout/stderr.
    # For now, the identical output is a strong indicator.
