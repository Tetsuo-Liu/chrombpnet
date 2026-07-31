import argparse
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import pyfaidx
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="get gc content from a foreground bed file"
    )
    parser.add_argument(
        "-i",
        "--input_bed",
        help="10-column peak BED; GC is measured around the summit",
    )
    parser.add_argument(
        "-c",
        "--chrom_sizes",
        type=str,
        required=True,
        help="TSV file with chromosome name and size",
    )
    parser.add_argument("-g", "--genome", help="reference genome fasta")
    parser.add_argument(
        "-op",
        "--output_prefix",
        help="output prefix for foreground GC values",
    )
    parser.add_argument(
        "-il",
        "--inputlen",
        type=int,
        default=2114,
        help="window length for GC calculation",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of chromosome workers",
    )
    return parser.parse_args()


def _process_chromosome(worker_args):
    chrom, regions, genome_path, inputlen = worker_args
    reference = pyfaidx.Fasta(genome_path)
    results = []
    try:
        for row_position, start, end, summit in regions:
            window_start = summit - inputlen // 2
            window_end = summit + inputlen // 2
            sequence = str(reference[chrom][window_start:window_end]).upper()
            gc_fraction = round(
                (sequence.count("G") + sequence.count("C")) / len(sequence),
                2,
            )
            results.append(
                (
                    row_position,
                    chrom,
                    window_start,
                    window_end,
                    gc_fraction,
                )
            )
    finally:
        reference.close()
    return results


def process_gc_content(
    input_bed,
    chrom_sizes,
    genome,
    output_prefix,
    inputlen,
    jobs=1,
):
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if inputlen % 2 != 0:
        raise ValueError("inputlen must be even")

    chrom_sizes_dict = {
        line.strip().split("\t")[0]: int(line.strip().split("\t")[1])
        for line in open(chrom_sizes).readlines()
    }
    data = pd.read_csv(input_bed, header=None, sep="\t")
    if data.shape[1] < 10:
        raise ValueError("input BED must have at least 10 columns")

    print("num_rows:" + str(data.shape[0]))

    regions_by_chrom = defaultdict(list)
    filtered_points = 0
    for row_position, row in enumerate(data.itertuples(index=False, name=None)):
        chrom = row[0]
        start = row[1]
        end = row[2]
        summit = start + row[9]
        window_start = summit - inputlen // 2
        window_end = summit + inputlen // 2
        chrom_size = chrom_sizes_dict[chrom]
        if (
            window_start < 0
            or window_end > chrom_size
        ):
            filtered_points += 1
            continue
        regions_by_chrom[chrom].append(
            (row_position, start, end, summit)
        )

    worker_args = [
        (chrom, regions, genome, inputlen)
        for chrom, regions in regions_by_chrom.items()
    ]
    all_results = []
    if jobs == 1:
        for args in tqdm(worker_args, desc="Processing chromosomes"):
            all_results.extend(_process_chromosome(args))
    else:
        with Pool(processes=jobs) as pool:
            for results in tqdm(
                pool.imap_unordered(_process_chromosome, worker_args),
                total=len(worker_args),
                desc="Processing chromosomes",
            ):
                all_results.extend(results)

    all_results.sort(key=lambda result: result[0])
    with open(output_prefix + ".bed", "w") as output_handle:
        for _, chrom, start, end, gc_fraction in all_results:
            output_handle.write(
                f"{chrom}\t{start}\t{end}\t{gc_fraction}\n"
            )

    print(
        "Number of regions filtered because inputlen sequence cannot be "
        f"constructed: {filtered_points}"
    )
    filtered_percentage = round(
        filtered_points * 100.0 / data.shape[0], 3
    )
    print(f"Percentage of regions filtered {filtered_percentage}%")
    if filtered_percentage > 25:
        print(
            "WARNING: If percentage of regions filtered is high (>25%) - "
            "your genome is very small - consider using a reduced "
            "input/output length for your genome"
        )


def main(args):
    process_gc_content(
        input_bed=args.input_bed,
        chrom_sizes=args.chrom_sizes,
        genome=args.genome,
        output_prefix=args.output_prefix,
        inputlen=args.inputlen,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main(parse_args())
