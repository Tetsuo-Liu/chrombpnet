# ChromBPNet preprocessing optimizations

This downstream build is based on upstream ChromBPNet commit
`09938fdb4397ec0006510e5251e48920a505d4de`. It keeps the upstream model
architectures, losses, data generators, augmentation, and training loop
unchanged. The changes add reusable input paths and accelerate preprocessing
without changing model or training semantics.

## Added interfaces

An existing bigWig can be supplied to either training command:

```bash
chrombpnet bias train \
  ... \
  --input-bigwig sample_unstranded.bw \
  --jobs 8

chrombpnet train \
  ... \
  --input-bigwig sample_unstranded.bw \
  --jobs 8
```

The bigWig must contain raw, unscaled, unstranded, enzyme-shifted insertions.
When it is provided, ChromBPNet does not run reads-to-bigWig preprocessing.

A genome-wide GC profile can be computed once:

```bash
chrombpnet prep genomewide-gc \
  --genome genome.fa \
  --output-prefix genome_gc_inputlen2114_stride1000 \
  --inputlen 2114 \
  --stride 1000
```

It can then be reused when generating nonpeaks:

```bash
chrombpnet prep nonpeaks \
  ... \
  --genomewide-gc-profile genome_gc_inputlen2114_stride1000.bed \
  --jobs 8
```

The reused profile must have been generated from the same reference genome,
input length, and stride supplied to `prep nonpeaks`.

## Parallel preprocessing semantics

`--jobs` parallelizes foreground-GC calculation and the bigWig count retrieval
used for bias and main-model hyperparameter selection. Results are restored to
the original BED row order before filtering and sampling. Main-model
hyperparameter nonpeak sampling uses `--seed` (default `1234`).

Hyperparameter count retrieval no longer constructs one-hot sequences when
only counts are used. Sequence extraction for bias-model depth adjustment is
performed only after the retained nonpeaks are known.

## Validation

The optimization tests compare one and multiple jobs on interleaved
chromosome input and require identical filtered BEDs, parameter tables, and
scaled bias-model weights.

On 100,000 real regions from an existing 4090 workflow:

| Operation | Upstream-style | Optimized, 1 job | Optimized, 8 jobs |
| --- | ---: | ---: | ---: |
| bigWig counts | 30.95 s | 8.30 s | 1.43 s |
| foreground GC | 4.09 s | 1.54 s | 0.37 s |

Count and GC outputs were identical. The upstream-style count path also
constructed an 845.6 MB one-hot sequence array that the optimized path avoids at this
stage. A two-epoch bias-model run followed by a two-epoch ChromBPNet run
completed on an RTX 4090 with TensorFlow 2.8 using the direct-bigWig path.
