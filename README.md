# SlideDP

## Scaling Host-Resident LLM Fine-Tuning Across Multiple GPUs

**Paper:** arXiv link pending.

[Current code release](#current-code-release) · [SlideFormer](https://github.com/RegiaYoung/SlideFormer)

SlideDP is a synchronous data-parallel runtime for full-parameter LLM
fine-tuning when multiple GPUs share host-resident model states and host-side
resources. Layer streaming makes large models trainable beyond GPU memory,
but adding GPU ranks also changes the balance between computation, host
memory traffic, and CPU updates.

SlideDP coordinates parameter delivery, gradient aggregation, and CPU updates
around a shared authoritative host state, while decoupling persistent state
ownership from communication routes. It supports replicated or sharded
parameter delivery and CPU- or GPU-side gradient reduction, with route choices
that can be configured according to workload and topology while preserving
synchronous data-parallel update semantics. A cross-rank pipeline overlaps
data movement and CPU updates with GPU computation.

An analytical step-time model characterizes shared-resource bottlenecks and
exposed pipeline work; runtime measurements guide communication, chunking, and
activation policies under a GPU memory budget. The paper and its full technical
description will be linked here when the arXiv preprint becomes available.

## Current code release

**This initial release provides a reference multi-GPU extension of SlideFormer
in [`baselines/slideformer_dp/`](baselines/slideformer_dp/).**

The optimized SlideDP runtime used for the paper's SlideDP results is not
included in this initial code release. We plan to release the full runtime in
a future update.

## Getting started

Install CUDA-enabled PyTorch and follow the SlideFormer multi-GPU extension's
[environment and training instructions](baselines/slideformer_dp/README.md).
From this repository:

```bash
cd baselines/slideformer_dp
pip install -r requirements.txt
torchrun --standalone --nproc_per_node=2 scripts/main_dp.py \
  --tiny --steps 3 --chunk-numel 4096 --cpu-threads 2
```

## Repository layout

```text
SlideDP/
├── README.md
├── LICENSE
├── NOTICE
└── baselines/
    └── slideformer_dp/   # SlideFormer multi-GPU extension
```

## Paper and citation

The arXiv identifier and BibTeX entry will be added after the preprint is
announced. The paper title is *SlideDP: Scaling Host-Resident LLM Fine-Tuning
Across Multiple GPUs*.

For the original single-GPU runtime, see the
[SlideFormer paper and citation](https://github.com/RegiaYoung/SlideFormer#citation).

## License

The released source code is distributed under the Apache License 2.0, with
third-party notices retained in the baseline directory. See [LICENSE](LICENSE)
and [NOTICE](NOTICE).
