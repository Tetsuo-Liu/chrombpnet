"""Tests for optimized hyperparameter preprocessing paths."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyBigWig
import pytest


tf = pytest.importorskip("tensorflow")

from chrombpnet.helpers.hyperparameters import find_bias_hyperparams
from chrombpnet.helpers.hyperparameters import find_chrombpnet_hyperparams


def _write_fasta(path, chrom_sizes):
    with path.open("w") as handle:
        for chrom, size in chrom_sizes.items():
            handle.write(f">{chrom}\n")
            handle.write(("ACGT" * (size // 4 + 1))[:size] + "\n")


def _write_bigwig(path, chrom_sizes, signals):
    bigwig = pyBigWig.open(str(path), "w")
    bigwig.addHeader(list(chrom_sizes.items()))
    for chrom in chrom_sizes:
        chrom_signals = sorted(
            (signal for signal in signals if signal[0] == chrom),
            key=lambda signal: signal[1],
        )
        bigwig.addEntries(
            [signal[0] for signal in chrom_signals],
            [signal[1] - 5 for signal in chrom_signals],
            ends=[signal[1] + 5 for signal in chrom_signals],
            values=[signal[2] for signal in chrom_signals],
        )
    bigwig.close()


def _peak_line(chrom, center):
    return (
        f"{chrom}\t{center - 5}\t{center + 5}\t.\t.\t.\t.\t.\t."
        "\t5\n"
    )


def _write_regions(path, regions):
    path.write_text("".join(_peak_line(*region) for region in regions))


def _make_dataset(tmp_path):
    chrom_sizes = {"chr1": 5000, "chr2": 5000, "chr3": 5000}
    fasta_path = tmp_path / "genome.fa"
    bigwig_path = tmp_path / "signal.bw"
    peaks_path = tmp_path / "peaks.bed"
    nonpeaks_path = tmp_path / "nonpeaks.bed"
    folds_path = tmp_path / "folds.json"

    peaks = []
    nonpeaks = []
    signals = []
    for chrom_index, chrom in enumerate(chrom_sizes):
        for index in range(12):
            peak_center = 200 + index * 40
            nonpeak_center = 1200 + index * 40
            peaks.append((chrom, peak_center))
            nonpeaks.append((chrom, nonpeak_center))
            signals.append(
                (chrom, peak_center, 10.0 + (index % 3))
            )
            signals.append(
                (chrom, nonpeak_center, 1.0 + (index % 8))
            )

    interleaved_peaks = [
        region
        for index in range(12)
        for region in peaks[index::12]
    ]
    interleaved_nonpeaks = [
        region
        for index in range(12)
        for region in nonpeaks[index::12]
    ]
    _write_fasta(fasta_path, chrom_sizes)
    _write_bigwig(bigwig_path, chrom_sizes, signals)
    _write_regions(peaks_path, interleaved_peaks)
    _write_regions(nonpeaks_path, interleaved_nonpeaks)
    folds_path.write_text(
        json.dumps(
            {
                "train": ["chr1"],
                "valid": ["chr2"],
                "test": ["chr3"],
            }
        )
    )
    return {
        "fasta": fasta_path,
        "bigwig": bigwig_path,
        "peaks": peaks_path,
        "nonpeaks": nonpeaks_path,
        "folds": folds_path,
    }


def _assert_files_equal(first_prefix, second_prefix, suffixes):
    for suffix in suffixes:
        assert Path(str(first_prefix) + suffix).read_bytes() == Path(
            str(second_prefix) + suffix
        ).read_bytes()


def test_bias_hyperparameters_match_across_job_counts(tmp_path):
    data = _make_dataset(tmp_path)
    sequential_dir = tmp_path / "bias_sequential"
    parallel_dir = tmp_path / "bias_parallel"
    sequential_dir.mkdir()
    parallel_dir.mkdir()

    common = {
        "genome": str(data["fasta"]),
        "bigwig": str(data["bigwig"]),
        "peaks": str(data["peaks"]),
        "nonpeaks": str(data["nonpeaks"]),
        "bias_threshold_factor": 0.5,
        "outlier_threshold": 0.99,
        "max_jitter": 0,
        "chr_fold_path": str(data["folds"]),
        "inputlen": 20,
        "outputlen": 10,
        "filters": 8,
        "n_dilation_layers": 1,
    }
    sequential_prefix = sequential_dir / "run."
    parallel_prefix = parallel_dir / "run."
    find_bias_hyperparams.main(
        SimpleNamespace(
            **common,
            jobs=1,
            output_prefix=str(sequential_prefix),
        )
    )
    find_bias_hyperparams.main(
        SimpleNamespace(
            **common,
            jobs=2,
            output_prefix=str(parallel_prefix),
        )
    )

    _assert_files_equal(
        sequential_prefix,
        parallel_prefix,
        [
            "filtered.bias_nonpeaks.bed",
            "filtered.bias_peaks.bed",
            "bias_data_params.tsv",
            "bias_model_params.tsv",
        ],
    )


def _write_bias_model(path):
    sequence = tf.keras.Input(shape=(20, 4), name="sequence")
    flattened = tf.keras.layers.Flatten()(sequence)
    profile = tf.keras.layers.Dense(
        10, name="logits_profile_predictions"
    )(flattened)
    logcounts = tf.keras.layers.Dense(1, name="logcounts")(flattened)
    model = tf.keras.Model(sequence, [profile, logcounts])
    model.save(path)


def test_main_hyperparameters_match_across_job_counts(tmp_path):
    data = _make_dataset(tmp_path)
    bias_model_path = tmp_path / "bias.h5"
    _write_bias_model(bias_model_path)
    sequential_dir = tmp_path / "main_sequential"
    parallel_dir = tmp_path / "main_parallel"
    sequential_dir.mkdir()
    parallel_dir.mkdir()

    common = {
        "genome": str(data["fasta"]),
        "bigwig": str(data["bigwig"]),
        "peaks": str(data["peaks"]),
        "nonpeaks": str(data["nonpeaks"]),
        "negative_sampling_ratio": 0.5,
        "outlier_threshold": 0.99,
        "max_jitter": 5,
        "chr_fold_path": str(data["folds"]),
        "inputlen": 20,
        "outputlen": 10,
        "filters": 8,
        "n_dilation_layers": 1,
        "bias_model_path": str(bias_model_path),
        "seed": 1234,
    }
    sequential_prefix = sequential_dir / "run."
    parallel_prefix = parallel_dir / "run."
    find_chrombpnet_hyperparams.main(
        SimpleNamespace(
            **common,
            jobs=1,
            output_prefix=str(sequential_prefix),
        )
    )
    find_chrombpnet_hyperparams.main(
        SimpleNamespace(
            **common,
            jobs=2,
            output_prefix=str(parallel_prefix),
        )
    )

    _assert_files_equal(
        sequential_prefix,
        parallel_prefix,
        [
            "filtered.peaks.bed",
            "filtered.nonpeaks.bed",
            "chrombpnet_data_params.tsv",
        ],
    )
    sequential_model_params = Path(
        str(sequential_prefix) + "chrombpnet_model_params.tsv"
    ).read_text()
    parallel_model_params = Path(
        str(parallel_prefix) + "chrombpnet_model_params.tsv"
    ).read_text()
    assert sequential_model_params.replace(
        str(sequential_prefix), "<output-prefix>"
    ) == parallel_model_params.replace(
        str(parallel_prefix), "<output-prefix>"
    )
    sequential_model = tf.keras.models.load_model(
        str(sequential_prefix) + "bias_model_scaled.h5",
        compile=False,
    )
    parallel_model = tf.keras.models.load_model(
        str(parallel_prefix) + "bias_model_scaled.h5",
        compile=False,
    )
    for sequential_weights, parallel_weights in zip(
        sequential_model.get_weights(), parallel_model.get_weights()
    ):
        np.testing.assert_array_equal(
            sequential_weights, parallel_weights
        )
