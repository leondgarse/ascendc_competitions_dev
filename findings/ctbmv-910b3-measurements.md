# 910B3 measurements — the actual acceptance device

Chip confirmed by `npu-smi info`: **`910B3`** (the earlier session ran on a devenv
reporting `Ascend910_9362` = A3). Built natively `SOC_VERSION=ascend910b3`,
`NPU_ARCH=dav-2201`, CANN 9.1.0. Aicore freq 1800 MHz (cur 800 MHz).

Kernel: committed band-parallel version, `feat/aclblasCtbmv-arch22` @ `39fcf1d`.

## Correctness

**84/84 and 1872/1872 pass on 910B3** — the implementation is correct on the
acceptance device, not just on A3.

## Headline perf cases

| case | NPU 910B3 | target | over |
|---|---|---|---|
| 512/8 U N NON_UNIT | 60.96 us | 10.00 | 6.1x |
| 1024/16 U N NON_UNIT | 69.24 us | 16.51 | 4.2x |
| 2048/32 L T NON_UNIT | 100.00 us | 19.77 | 5.1x |

## Full 200-case acceptance sweep

`data/npu910b3_200cases.csv` — every case from `gpu_baseline.csv`, measured on
910B3 with the same protocol (20 warmup, 60 timed, one sync).

- **pass: 0/200** against `npu <= a100_us / 0.8`
- NPU/target percentiles: p0 **1.90x**, p25 4.73x, p50 **6.00x**, p75 7.97x, p100 18.24x
- closest case: n=603 k=206 UPPER/N/UNIT at 1.90x

The distribution matters: the *best* case is still 1.9x over, and the median is
6x. This is not a near miss on a few shapes — it is a uniform shortfall.

Interesting: the closest cases are all **large k with UNIT diagonal** (k~200),
where the per-band fixed costs amortise best. The worst are small-k shapes where
dispatch and setup dominate.

## Hardware constants (910B3)

| Constant | 910B3 | A3 (previous) |
|---|---|---|
| Host launch (no sync) | 4.8 us | 2.9 us |
| Minimum call (n=1, k=0) | **23.5 us** | 25.1 us |
| First (cold) call | 62.5 us | 50.4 us |
| Extra dispatch (`copy_kernel`) | 13.4-19.8 us | 14.4-20.9 us |

**The 23.5 us floor still exceeds the 10 us target for case 1.**

## Cost model refit on 910B3

```
bands = k+1 ; cores = min(bands, 40) ; bandsPerCore = ceil(bands/cores)
chunks = ceil(n/1024) ; serialDMA = bandsPerCore*chunks
kElems = serialDMA * min(n,1024) / 1000

t_us = 56.24 + 12.95 * serialDMA + 6.68 * kElems
```
mean error **9.4%**, max 32.3% over 32 points (A3 fit was 58.5 / 13.94 / 6.97).

## 910B3 vs A3 — same code, same build

median ratio **0.946**: 910B3 is ~5% *faster* than the A3 devenv on this kernel,
consistent across all 36 shared points (0.92-0.99). So the earlier A3 numbers
were a good proxy; nothing about the gap analysis changes.

## msprof kernel split (910B3)

| shape | compute | copy | total |
|---|---|---|---|
| 512/8 | 47.39 | 13.44 | 60.8 |
| 1024/16 | 53.74 | 15.34 | 69.1 |
| 2048/32 | 78.94 | 18.28 | 97.2 |
| 4096/128 | 336.89 | 19.83 | 356.7 |

## Core scaling (n=2048, cores = k+1)

```
1: 32.6   2: 104.0   4: 99.8   8: 94.8   16: 91.0
32: 94.9   40: 101.4   48: 145.8
```
Best at 16 cores; degrades past 40 (physical AIV count). DMA-bound, not
compute-bound — matches the A3 finding.
