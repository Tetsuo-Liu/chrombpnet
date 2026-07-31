from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from chrombpnet import CHROMBPNET
from chrombpnet.helpers.hyperparameters.region_counts import (
    get_region_counts,
)
from chrombpnet.helpers.make_gc_matched_negatives.get_gc_content import (
    process_gc_content,
)
from chrombpnet.pipelines import _prepare_bigwig
from chrombpnet.parsers import read_parser


def _write_fasta(path, sequences):
    with path.open("w") as handle:
        for chrom, sequence in sequences.items():
            handle.write(f">{chrom}\n{sequence}\n")


def _write_bigwig(path, chrom_sizes, intervals):
    bigwig = pyBigWig.open(str(path), "w")
    bigwig.addHeader(list(chrom_sizes.items()))
    for chrom in chrom_sizes:
        chrom_intervals = [
            interval for interval in intervals if interval[0] == chrom
        ]
        if not chrom_intervals:
            continue
        bigwig.addEntries(
            [interval[0] for interval in chrom_intervals],
            [interval[1] for interval in chrom_intervals],
            ends=[interval[2] for interval in chrom_intervals],
            values=[interval[3] for interval in chrom_intervals],
        )
    bigwig.close()


def _peak_line(chrom, center):
    return (
        f"{chrom}\t{center - 5}\t{center + 5}\t.\t.\t.\t.\t.\t."
        "\t5\n"
    )


def test_region_counts_preserve_interleaved_input_order(tmp_path):
    bigwig_path = tmp_path / "signal.bw"
    chrom_sizes = {"chr1": 1000, "chr2": 1000}
    _write_bigwig(
        bigwig_path,
        chrom_sizes,
        [
            ("chr1", 0, 1000, 1.0),
            ("chr2", 0, 1000, 2.0),
        ],
    )
    regions = pd.DataFrame(
        [
            ("chr2", 95, 105, 5),
            ("chr1", 195, 205, 5),
            ("chr2", 295, 305, 5),
        ],
        columns=["chr", "start", "end", "summit"],
    )

    sequential = get_region_counts(
        str(bigwig_path), regions, output_width=10, jobs=1
    )
    parallel = get_region_counts(
        str(bigwig_path), regions, output_width=10, jobs=2
    )

    np.testing.assert_array_equal(sequential, [20.0, 10.0, 20.0])
    np.testing.assert_array_equal(parallel, sequential)


def test_foreground_gc_is_order_stable_and_matches_reference(tmp_path):
    fasta_path = tmp_path / "genome.fa"
    chrom_sizes_path = tmp_path / "chrom.sizes"
    peaks_path = tmp_path / "peaks.bed"
    _write_fasta(
        fasta_path,
        {
            "chr1": "A" * 50 + "G" * 50 + "C" * 50 + "T" * 50,
            "chr2": "C" * 100 + "A" * 100,
        },
    )
    chrom_sizes_path.write_text("chr1\t200\nchr2\t200\n")
    peaks_path.write_text(
        _peak_line("chr2", 50)
        + _peak_line("chr1", 100)
        + _peak_line("chr2", 150)
        + _peak_line("chr1", 2)
    )

    sequential_prefix = tmp_path / "sequential"
    parallel_prefix = tmp_path / "parallel"
    process_gc_content(
        str(peaks_path),
        str(chrom_sizes_path),
        str(fasta_path),
        str(sequential_prefix),
        inputlen=10,
        jobs=1,
    )
    process_gc_content(
        str(peaks_path),
        str(chrom_sizes_path),
        str(fasta_path),
        str(parallel_prefix),
        inputlen=10,
        jobs=2,
    )

    expected = (
        "chr2\t45\t55\t1.0\n"
        "chr1\t95\t105\t1.0\n"
        "chr2\t145\t155\t0.0\n"
    )
    assert Path(str(sequential_prefix) + ".bed").read_text() == expected
    assert (
        Path(str(parallel_prefix) + ".bed").read_bytes()
        == Path(str(sequential_prefix) + ".bed").read_bytes()
    )


def test_precomputed_bigwig_is_used_directly(tmp_path):
    bigwig_path = tmp_path / "signal.bw"
    _write_bigwig(
        bigwig_path,
        {"chr1": 100},
        [("chr1", 0, 100, 1.0)],
    )
    args = SimpleNamespace(
        input_bigwig=str(bigwig_path),
        output_dir=str(tmp_path / "output"),
    )

    _prepare_bigwig(args, "")

    assert args.bigwig == str(bigwig_path.resolve())


def test_preprocessing_cli_arguments(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chrombpnet",
            "prep",
            "genomewide-gc",
            "-g",
            "genome.fa",
            "-o",
            "genome_gc",
            "-il",
            "2114",
            "-st",
            "1000",
        ],
    )
    gc_args = read_parser()
    assert gc_args.cmd_prep == "genomewide-gc"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chrombpnet",
            "pipeline",
            "-g",
            "genome.fa",
            "-c",
            "chrom.sizes",
            "-ibw",
            "signal.bw",
            "-o",
            "output",
            "-d",
            "ATAC",
            "-p",
            "peaks.bed",
            "-n",
            "nonpeaks.bed",
            "-fl",
            "fold.json",
            "-b",
            "bias.h5",
            "--jobs",
            "8",
        ],
    )
    pipeline_args = read_parser()
    assert pipeline_args.input_bigwig == "signal.bw"
    assert pipeline_args.jobs == 8


def test_precomputed_gc_profile_matches_on_the_fly_nonpeaks(
    tmp_path, monkeypatch
):
    fasta_path = tmp_path / "genome.fa"
    chrom_sizes_path = tmp_path / "chrom.sizes"
    peaks_path = tmp_path / "peaks.bed"
    folds_path = tmp_path / "folds.json"
    chromosomes = ["chr1", "chr2", "chr3"]
    _write_fasta(
        fasta_path,
        {chrom: "ACGT" * 1250 for chrom in chromosomes},
    )
    chrom_sizes_path.write_text(
        "".join(f"{chrom}\t5000\n" for chrom in chromosomes)
    )
    peaks_path.write_text(
        "".join(_peak_line(chrom, 1000) for chrom in chromosomes)
    )
    folds_path.write_text(
        json.dumps(
            {
                "train": ["chr1"],
                "valid": ["chr2"],
                "test": ["chr3"],
            }
        )
    )

    shared_prefix = tmp_path / "shared_gc"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chrombpnet",
            "prep",
            "genomewide-gc",
            "-g",
            str(fasta_path),
            "-o",
            str(shared_prefix),
            "-il",
            "100",
            "-st",
            "100",
        ],
    )
    CHROMBPNET.main()

    generated_prefix = tmp_path / "generated"
    common_args = [
        "chrombpnet",
        "prep",
        "nonpeaks",
        "-g",
        str(fasta_path),
        "-p",
        str(peaks_path),
        "-c",
        str(chrom_sizes_path),
        "-fl",
        str(folds_path),
        "-il",
        "100",
        "-st",
        "100",
        "-npr",
        "1",
        "-s",
        "1234",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        common_args
        + [
            "-o",
            str(generated_prefix),
            "--jobs",
            "1",
        ],
    )
    CHROMBPNET.main()

    reused_prefix = tmp_path / "reused"
    monkeypatch.setattr(
        sys,
        "argv",
        common_args
        + [
            "-o",
            str(reused_prefix),
            "--genomewide-gc-profile",
            str(shared_prefix) + ".bed",
            "--jobs",
            "2",
        ],
    )
    CHROMBPNET.main()

    assert Path(
        str(generated_prefix) + "_negatives.bed"
    ).read_bytes() == Path(
        str(reused_prefix) + "_negatives.bed"
    ).read_bytes()

    mismatched_profile = tmp_path / "mismatched_gc.bed"
    mismatched_profile.write_text("chr1\t0\t200\t0.5\n")
    monkeypatch.setattr(
        sys,
        "argv",
        common_args
        + [
            "-o",
            str(tmp_path / "mismatched"),
            "--genomewide-gc-profile",
            str(mismatched_profile),
        ],
    )
    with pytest.raises(
        ValueError,
        match="window width does not match inputlen",
    ):
        CHROMBPNET.main()
