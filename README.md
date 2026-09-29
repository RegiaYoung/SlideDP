# SlideDP

## Scaling Host-Resident LLM Fine-Tuning Across Multiple GPUs

**Paper:** [arXiv:2609.34162](https://arxiv.org/abs/2609.34162).

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
activation policies under a GPU memory budget. See the
[paper](https://arxiv.org/abs/2609.34162) for the full technical description.

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

## Supported Models

- [Qwen2, Qwen2.5, Qwen3, and Qwen3.6 families](https://huggingface.co/Qwen)
- [Llama 3, 3.1, 3.2, and 3.3 families](https://huggingface.co/meta-llama)
- [Mistral family](https://huggingface.co/mistralai)
- Gemma 4 family: [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) and [31B](https://huggingface.co/google/gemma-4-31B).

We support full-parameter fine-tuning across a wide range of [Hugging Face](https://huggingface.co/models) language models.
Support for multimodal models is under active development.

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

The paper, *SlideDP: Scaling Host-Resident LLM Fine-Tuning Across Multiple
GPUs*, is available on [arXiv](https://arxiv.org/abs/2609.34162).

```bibtex
@misc{yang2026slidedp,
  title={{SlideDP}: Scaling Host-Resident {LLM} Fine-Tuning Across Multiple {GPUs}},
  author={Ruijia Yang and Shiyuan Lin and Yulong Ao and Zhiyu Li and Yingli Zhao and Xianduo Li and Yonghua Lin and Zeyi Wen},
  year={2026},
  eprint={2609.34162},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2609.34162}
}
```

For the original single-GPU runtime, see the
[SlideFormer paper and citation](https://github.com/RegiaYoung/SlideFormer#citation).

## License

The released source code is distributed under the Apache License 2.0, with
third-party notices retained in the baseline directory. See [LICENSE](LICENSE)
and [NOTICE](NOTICE).
