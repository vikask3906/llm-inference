# Admission-Control Benchmark: SLO Protection Under Pressure

Mixed gold/silver/bronze workload (mix: 20% gold / 30% silver / 50% bronze) at increasing offered load,
compared with and without admission control. SLO = 500ms TTFT;
shed pressure = 0.75; fleet = 3 backends @ 16 max inflight each.

## Gold-tier SLO compliance

| load_mult | baseline gold OK | admission gold OK | baseline bronze served | admission bronze served |
|---|---|---|---|---|
| 0.50x | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.75x | 100.0% | 100.0% | 100.0% | 100.0% |
| 1.00x | 100.0% | 100.0% | 100.0% | 100.0% |
| 1.25x | 100.0% | 100.0% | 100.0% | 100.0% |
| 1.50x | 2.0% | 100.0% | 100.0% | 29.2% |
| 2.00x | 2.0% | 100.0% | 100.0% | 29.2% |
| 3.00x | 2.0% | 97.5% | 100.0% | 5.5% |

**Reading the table**: at light load (< 1x) baseline and admission are
identical -- nothing to protect against. As offered load climbs past
capacity, baseline's gold-SLO compliance collapses (every tier shares the
queue); admission's stays high because it sheds bronze first. The cost
is visible too: bronze's served-rate drops under admission, by design.

## Full per-tier metrics

| load_mult | controller | gold served | silver served | bronze served | gold mean TTFT (ms) | bronze mean TTFT (ms) | rejected | queued |
|---|---|---|---|---|---|---|---|---|
| 0.50x | baseline | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 0.50x | admission | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 0.75x | baseline | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 0.75x | admission | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 1.00x | baseline | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 1.00x | admission | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 1.25x | baseline | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 1.25x | admission | 100.0% | 100.0% | 100.0% | 272.8 | 268.7 | 0 | 0 |
| 1.50x | baseline | 100.0% | 100.0% | 100.0% | 519.0 | 512.3 | 0 | 0 |
| 1.50x | admission | 100.0% | 99.0% | 29.2% | 345.1 | 313.3 | 0 | 714 |
| 2.00x | baseline | 100.0% | 100.0% | 100.0% | 519.0 | 512.3 | 0 | 0 |
| 2.00x | admission | 100.0% | 99.0% | 29.2% | 345.1 | 313.3 | 0 | 714 |
| 3.00x | baseline | 100.0% | 100.0% | 100.0% | 761.1 | 752.2 | 0 | 0 |
| 3.00x | admission | 100.0% | 80.0% | 5.5% | 370.9 | 269.6 | 0 | 1065 |

## Caveat

TTFT here is the same prefill + queue-penalty proxy the admission gate's
cost function uses; not wall-clock. The benchmark measures the controller's
policy behaviour (who gets admitted at what fleet pressure), which is what
it actually decides. Absolute ms validation is the GPU job; same caveat
as `bench/matrix.py`, `bench/rag_bench.py`, `bench/dag_bench.py`.
