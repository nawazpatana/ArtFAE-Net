# Exploratory matched resource measurements

| model | features_ms_per_image | complete_ms_per_image | head_ms_per_image | images_per_second | peak_CUDA_allocated_GiB | peak_CUDA_reserved_GiB | peak_process_RSS_GiB |
|---|---|---|---|---|---|---|---|
| R1 | 13.0543 | 16.0173 | 0.0384 | 62.4325 | 1.9206 | 2.0625 | 2.5383 |
| PROBE | 12.5331 | 15.5190 | 0.0040 | 64.4372 | 1.9178 | 2.0566 | 2.5341 |

Same RTX 5090, same 32 validation images, batch8, FP32, TF32 off, fresh independent workers, two warm-up batches and five measured passes. Each process loads the complete CLIP object, including unused probe text parameters. These are implementation observations, not minimum deployment memory or a 110× system-cost improvement. CUDA synchronized wall timing includes CPU decoding, real feature branches, head and output transfer; model loading is separate. File caches may be warm. RAM includes load/import peaks; CUDA allocated/reserved excludes driver/context/device-wide allocations. Batch median/p95 and per-batch component timings are in the worker JSON/CSV files. No training or holdout inference.
