# pi-lcm-memory · perf bench

- git: `ba43cec` (dirty)
- timestamp: 2026-04-28T03:52:52.474Z
- model: `Xenova/bge-small-en-v1.5` (q8)
- node: v25.9.0
- cpu: Apple M5 × 10
- memory: 16 GB
- mode: FULL (sweep corpus: 1000)

| benchmark | value | unit |
|---|---:|---|
| worker_warmup_ms | 191.0 | ms |
| embed_throughput | 596 | embeds/sec |
| embed_latency_b1_ms | 3.4 | ms |
| embed_latency_b32_ms | 52.1 | ms |
| sweep_throughput | 262 | rows/sec |
| db_size_bytes_per_row | 7373 | bytes |
| recall_latency_ms | 2.6 | ms |
