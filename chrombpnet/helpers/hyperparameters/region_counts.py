from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyBigWig


_WORKER_BIGWIG = None


def _initialize_worker(bigwig_path):
    global _WORKER_BIGWIG
    _WORKER_BIGWIG = pyBigWig.open(bigwig_path)


def _count_regions(bigwig, regions, output_width):
    results = []
    for row_position, chrom, center in regions:
        values = np.nan_to_num(
            bigwig.values(
                chrom,
                center - output_width // 2,
                center + output_width // 2,
            )
        )
        results.append((row_position, np.sum(values)))
    return results


def _count_regions_worker(worker_args):
    regions, output_width = worker_args
    return _count_regions(_WORKER_BIGWIG, regions, output_width)


def get_region_counts(bigwig_path, regions_df, output_width, jobs=1):
    """Return observed counts aligned to the input DataFrame row order."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")

    regions_by_chrom = defaultdict(list)
    for row_position, row in enumerate(
        regions_df.itertuples(index=False)
    ):
        center = int(row.start + row.summit)
        regions_by_chrom[row.chr].append(
            (row_position, row.chr, center)
        )

    worker_args = [
        (regions, output_width)
        for regions in regions_by_chrom.values()
    ]
    all_results = []
    if jobs == 1:
        bigwig = pyBigWig.open(bigwig_path)
        try:
            for regions, width in worker_args:
                all_results.extend(
                    _count_regions(bigwig, regions, width)
                )
        finally:
            bigwig.close()
    else:
        with Pool(
            processes=jobs,
            initializer=_initialize_worker,
            initargs=(bigwig_path,),
        ) as pool:
            for results in pool.imap_unordered(
                _count_regions_worker, worker_args
            ):
                all_results.extend(results)

    counts = np.empty(len(regions_df), dtype=np.float64)
    for row_position, count in all_results:
        counts[row_position] = count
    return counts
