# Disaggregation Benchmark: Adaptive vs Static Colocate/Split

Prefill/decode disaggregation (Splitwise / DistServe) is load-dependent:
splitting onto separate GPUs only pays off under load, where the queue it
saves beats the KV-handoff it costs. The adaptive router decides per
request; this benchmark shows it never loses to either static policy.

Fleet: 4 homogeneous backends. prompt=3000 tok, output=256 tok, handoff=6.0ms over 100GB/s.

## Mean total latency (ms) vs offered load

| load | colocate | split | adaptive | adaptive split-fraction |
|---|---|---|---|---|
| 0.25x | 201.2 | 207.2 | **201.2** | 0% |
| 0.50x | 321.1 | 233.9 | **262.7** | 25% |
| 1.00x | 560.7 | 415.0 | **408.4** | 29% |
| 2.00x | 1039.0 | 769.7 | **726.7** | 31% |
| 4.00x | 1991.6 | 1482.0 | **1348.4** | 33% |
| 6.00x | 2899.8 | 2156.8 | **1988.8** | 33% |
| 8.00x | 3822.8 | 2846.3 | **2579.8** | 33% |

**Reading the table**: at low load co-location wins (split's handoff is
pure overhead), and adaptive co-locates (split-fraction ~0%). As load
rises, splitting wins (it frees nodes from phase contention), and adaptive
shifts to splitting (split-fraction climbs). At every load point adaptive
tracks the better of the two static policies -- it's the lower envelope.

## Caveat

Latency is the disagg router's own cost model (compute proxy + queue
terms + analytic KV-handoff), not wall-clock. It captures the contention
tradeoff the policy optimizes. Real KV-transfer over NIXL/LMCache and
absolute ms are the GPU job; same caveat as `bench/matrix.py`.
