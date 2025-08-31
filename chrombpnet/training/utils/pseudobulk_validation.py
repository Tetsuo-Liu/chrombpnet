import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class PseudobulkMetadata:
    """
    Data class to store metadata for a pseudobulk sample.

    Attributes:
        pseudobulk_id (str): Unique identifier for the pseudobulk.
        cell_type (str): Cell type classification.
        sampling_weight (float): Sampling weight used during dynamic pairing.
        bigwig_path (Path): Path to the corresponding bigWig file.
        total_signal (float): Total signal amount in the bigWig file.
        file_exists (bool): Whether the file specified by bigwig_path actually exists.
    """
    pseudobulk_id: str
    cell_type: str
    sampling_weight: float
    bigwig_path: Path
    total_signal: float
    file_exists: bool


def load_pseudobulk_metadata(metadata_path: Path) -> List[PseudobulkMetadata]:
    """
    Load, validate, and return a list of pseudobulk metadata objects from a TSV file.

    Args:
        metadata_path (Path): Path to the `pseudobulk_metadata.tsv` file.

    Returns:
        List[PseudobulkMetadata]: A list of validated metadata objects.

    Raises:
        FileNotFoundError: If the specified metadata file is not found.
        ValueError: If the metadata file is missing required columns.
    """
    logger = logging.getLogger(__name__)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    logger.info(f"Loading pseudobulk metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path, sep='\t')

    # Validate required columns (Requirement 4.1)
    required_columns = [
        "pseudobulk_id", "cell_type", "sampling_weight", 
        "bigwig_path", "total_signal"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Metadata file is missing required columns: {missing_columns}")

    metadata_list = []
    for _, row in df.iterrows():
        bw_path = Path(row["bigwig_path"])
        
        # Check for file existence (Requirement 4.5)
        file_exists = bw_path.exists()
        if not file_exists:
            logger.warning(f"BigWig file not found for pseudobulk '{row['pseudobulk_id']}': {bw_path}")

        metadata_list.append(
            PseudobulkMetadata(
                pseudobulk_id=row["pseudobulk_id"],
                cell_type=row["cell_type"],
                sampling_weight=float(row["sampling_weight"]),
                bigwig_path=bw_path,
                total_signal=float(row["total_signal"]),
                file_exists=file_exists
            )
        )
    
    logger.info(f"Successfully loaded and validated {len(metadata_list)} pseudobulk metadata entries.")
    logger.info(f"Found {sum(m.file_exists for m in metadata_list)} existing BigWig files.")
    return metadata_list


def _validate_single_bigwig(metadata: PseudobulkMetadata) -> Dict[str, Any]:
    """
    Validate a single BigWig file for integrity.
    Worker function for parallel validation.
    
    Args:
        metadata (PseudobulkMetadata): Single pseudobulk metadata object
        
    Returns:
        Dict[str, Any]: Validation result for this file
    """
    import pyBigWig
    
    result = {
        "pseudobulk_id": metadata.pseudobulk_id,
        "file_exists": metadata.file_exists,
        "is_valid": False,
        "total_signal": metadata.total_signal,
        "error": None
    }
    
    if not metadata.file_exists:
        result["error"] = "File does not exist"
        return result
    
    try:
        bw = pyBigWig.open(str(metadata.bigwig_path))
        
        # Basic integrity checks
        if bw is None:
            raise ValueError("Failed to open BigWig file")
        
        # Check if file has any data
        chroms = bw.chroms()
        if not chroms:
            raise ValueError("BigWig file contains no chromosomes")
            
        # Verify file can be read (sample first chromosome)
        first_chrom = next(iter(chroms.keys()))
        chrom_length = chroms[first_chrom]
        
        # Try to read a small region to verify integrity
        test_region = bw.stats(first_chrom, 0, min(1000, chrom_length))
        if test_region is None:
            raise ValueError("Cannot read data from BigWig file")
        
        bw.close()
        result["is_valid"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def create_progress_bar(iterable, desc="Processing", **kwargs):
    """
    Create a unified progress bar for standard output.
    Ensures progress is displayed on command line regardless of logging configuration.

    Args:
        iterable: Target iterable for progress display
        desc (str): Progress bar description text
        **kwargs: Additional tqdm parameters

    Returns:
        tqdm: Configured progress bar
    """
    from tqdm import tqdm
    # Explicitly specify file=sys.stdout to avoid logging configuration issues
    kwargs.setdefault('file', sys.stdout)
    return tqdm(iterable, desc=desc, **kwargs)


def validate_pseudobulk_files(metadata_list: List[PseudobulkMetadata], n_jobs: int = None) -> Dict[str, Any]:
    """
    Validate BigWig file existence and integrity for all pseudobulk metadata entries using parallel processing.
    
    Args:
        metadata_list (List[PseudobulkMetadata]): List of pseudobulk metadata objects
        n_jobs (int, optional): Number of parallel jobs. If None, uses CPU count // 2
        
    Returns:
        Dict[str, Any]: Validation report with statistics and any issues found
    """
    from multiprocessing import Pool, cpu_count
    
    logger = logging.getLogger(__name__)
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() // 2)
    
    validation_report = {
        "total_pseudobulks": len(metadata_list),
        "existing_files": 0,
        "missing_files": [],
        "corrupted_files": [],
        "valid_files": 0,
        "total_signal_sum": 0.0
    }
    
    logger.info(f"Validating {len(metadata_list)} pseudobulk BigWig files using {n_jobs} parallel jobs...")
    
    # Use parallel processing to validate files with progress tracking
    with Pool(processes=n_jobs) as pool:
        # Use imap to get results as they complete for progress tracking
        progress_bar = create_progress_bar(
            range(len(metadata_list)), 
            desc="Validating BigWig files",
            unit="files"
        )
        
        validation_results = []
        for result in pool.imap(_validate_single_bigwig, metadata_list):
            validation_results.append(result)
            progress_bar.update(1)
        
        progress_bar.close()
    
    # Process results
    for result in validation_results:
        if result["file_exists"]:
            validation_report["existing_files"] += 1
            
            if result["is_valid"]:
                validation_report["valid_files"] += 1
                validation_report["total_signal_sum"] += result["total_signal"]
            else:
                validation_report["corrupted_files"].append({
                    "pseudobulk_id": result["pseudobulk_id"],
                    "error": result["error"]
                })
        else:
            validation_report["missing_files"].append({
                "pseudobulk_id": result["pseudobulk_id"]
            })
    
    # Log validation summary
    logger.info(f"Validation complete:")
    logger.info(f"  - Total pseudobulks: {validation_report['total_pseudobulks']}")
    logger.info(f"  - Existing files: {validation_report['existing_files']}")
    logger.info(f"  - Valid files: {validation_report['valid_files']}")
    logger.info(f"  - Missing files: {len(validation_report['missing_files'])}")
    logger.info(f"  - Corrupted files: {len(validation_report['corrupted_files'])}")
    logger.info(f"  - Total signal sum: {validation_report['total_signal_sum']:.0f}")
    
    if validation_report["missing_files"]:
        logger.warning(f"Found {len(validation_report['missing_files'])} missing BigWig files")
    
    if validation_report["corrupted_files"]:
        logger.warning(f"Found {len(validation_report['corrupted_files'])} corrupted BigWig files")
    
    return validation_report


def generate_cell_type_failure_report(metadata_list: List[PseudobulkMetadata], validation_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate detailed cell-type-specific failure statistics.
    CRITICAL: Never skip failures - generate detailed error statistics.
    
    Args:
        metadata_list (List[PseudobulkMetadata]): List of pseudobulk metadata objects
        validation_report (Dict[str, Any]): Validation report from validate_pseudobulk_files
        
    Returns:
        Dict[str, Dict[str, Any]]: Cell type statistics with counts and failure details
    """
    logger = logging.getLogger(__name__)
    
    cell_type_stats = {}
    for metadata in metadata_list:
        cell_type = metadata.cell_type
        if cell_type not in cell_type_stats:
            cell_type_stats[cell_type] = {
                "total": 0,
                "valid": 0,
                "missing": 0,
                "corrupted": 0,
                "total_signal": 0.0,
                "missing_ids": [],
                "corrupted_ids": []
            }
        
        cell_type_stats[cell_type]["total"] += 1
        
        if not metadata.file_exists:
            cell_type_stats[cell_type]["missing"] += 1
            cell_type_stats[cell_type]["missing_ids"].append(metadata.pseudobulk_id)
        else:
            # Check if this file is in the validation results
            pseudobulk_id = metadata.pseudobulk_id
            corrupted_file = next((c for c in validation_report["corrupted_files"] 
                                 if c["pseudobulk_id"] == pseudobulk_id), None)
            
            if corrupted_file:
                cell_type_stats[cell_type]["corrupted"] += 1
                cell_type_stats[cell_type]["corrupted_ids"].append({
                    "id": pseudobulk_id,
                    "error": corrupted_file["error"]
                })
            else:
                cell_type_stats[cell_type]["valid"] += 1
                cell_type_stats[cell_type]["total_signal"] += metadata.total_signal
    
    # Log cell type statistics
    logger.info("Cell Type Failure Statistics:")
    total_failures = 0
    for cell_type, stats in sorted(cell_type_stats.items()):
        failures = stats["missing"] + stats["corrupted"]
        total_failures += failures
        
        if stats["total"] > 0:
            success_rate = (stats["valid"] / stats["total"]) * 100
            logger.info(f"  {cell_type}:")
            logger.info(f"    Total: {stats['total']}")
            logger.info(f"    Valid: {stats['valid']}")
            logger.info(f"    Missing: {stats['missing']}")
            logger.info(f"    Corrupted: {stats['corrupted']}")
            logger.info(f"    Success rate: {success_rate:.1f}%")
            
            if stats["missing"] > 0:
                logger.error(f"    Missing files: {', '.join(stats['missing_ids'])}")
            
            if stats["corrupted"] > 0:
                for corrupted in stats["corrupted_ids"]:
                    logger.error(f"    Corrupted file {corrupted['id']}: {corrupted['error']}")
    
    return cell_type_stats


def validate_and_enforce_file_integrity(metadata_list: List[PseudobulkMetadata], n_jobs: int = None) -> Dict[str, Any]:
    """
    Validate pseudobulk files and enforce mandatory failure termination for any issues.
    CRITICAL: NEVER skip missing/corrupted files - terminate training with detailed reports.
    
    Args:
        metadata_list (List[PseudobulkMetadata]): List of pseudobulk metadata objects
        n_jobs (int, optional): Number of parallel jobs
        
    Returns:
        Dict[str, Any]: Validation report (only if all files are valid)
        
    Raises:
        RuntimeError: If ANY files are missing or corrupted (mandatory termination)
    """
    logger = logging.getLogger(__name__)
    
    # Perform validation
    validation_report = validate_pseudobulk_files(metadata_list, n_jobs)
    
    # Generate detailed cell-type statistics
    cell_type_stats = generate_cell_type_failure_report(metadata_list, validation_report)
    
    # Calculate total failures
    total_failures = len(validation_report["missing_files"]) + len(validation_report["corrupted_files"])
    
    if total_failures > 0:
        # Generate actionable error message
        error_message = [
            f"Training cannot proceed: {total_failures} unavailable pseudobulks detected.",
            "",
            "REQUIRED ACTIONS:"
        ]
        
        # Add cell-type-specific recommendations
        for cell_type, stats in sorted(cell_type_stats.items()):
            failures = stats["missing"] + stats["corrupted"]
            if failures > 0:
                error_message.append(f"  {cell_type}: {failures}/{stats['total']} files unavailable")
                
                if stats["missing"] > 0:
                    error_message.append(f"    - Fix missing files: {', '.join(stats['missing_ids'])}")
                
                if stats["corrupted"] > 0:
                    for corrupted in stats["corrupted_ids"]:
                        error_message.append(f"    - Fix corrupted file {corrupted['id']}: {corrupted['error']}")
        
        error_message.extend([
            "",
            "MANDATORY: All pseudobulk files must be valid before training can begin.",
            "Re-run preprocessing stages 05.1.2 and 05.1.3 to regenerate missing/corrupted files."
        ])
        
        # Log detailed error information
        for line in error_message:
            logger.error(line)
        
        # MANDATORY: Fail training if ANY files unavailable
        raise RuntimeError("\n".join(error_message))
    
    logger.info(f"✅ All {validation_report['total_pseudobulks']} pseudobulk files validated successfully")
    return validation_report