# Latency benchmark

**Status: not run.** The configured embedding model and Hugging Face Datasets package are absent, and there is no indexed collection. No p50/p95/p99 values are fabricated. The full ML dependency installation stalled while unpacking PyTorch and was stopped after core fixture tests were installed.

Run `python -m adrosonic_retrieval.benchmark --queries 100` after indexing. The tool records every query to `artifacts/latency_log.csv` and summary statistics to `artifacts/benchmark_report.json`. Timed latency is an in-process service call including query embedding, Qdrant search, and Python result conversion; it excludes HTTP transport/JSON serialization. Model initialization and first-time model loading occur before timing. The benchmark labels its run warm; cold-start latency is not measured. Separate embedding and Qdrant measurements are included for profiling. p95 below 300 ms remains unverified.
