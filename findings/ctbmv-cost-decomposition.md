# Where every microsecond goes — measured decomposition on 910B3

All figures: 910B3, CANN 9.1.0, n=512 complex64 (4 KB of x), 30 warmup +
200 timed calls, one stream sync. Runs 2 and 3 of the same binary agree to
within 0.5 us; run 1 is discarded as cold-start. Source: `data/floor.cpp`.

## The ladder

| what | us | what it adds |
|---|---|---|
| `aclrtMemcpyAsync` D2D, 4 KB, **no kernel** | **2.8** | runtime floor, no kernel involved |
| `asdBlasCcopy` (SIP, plan-based) | **17.2** | + kernel launch (~14.4) |
| `aclblasCtbmv` k=0 (ours) | **23.0** | + our kernel's own setup (~5.8) |
| `aclblasCcopy` (ops-blas, stock) | **28.1** | + their per-call malloc/memcpy/free (~5.1) |
| `aclblasCtbmv` k=8 (ours, full band path) | **60.4** | + the actual banded algorithm (~37) |

## Key measurements

**1. Kernel launch costs ~14.4 us and is irreducible.**
`asdBlasCcopy` (17.2) minus `aclrtMemcpyAsync` (2.8) = 14.4 us. SIP is the
best-case invocation model available on this chip — plan-based, tiling hoisted
out — and it still pays this.

**2. Our launch floor is size-independent.**

```
ctbmv n=1 k=0: 23.21 us     ctbmv n=4 k=0: 22.94 us
ctbmv n=2 k=0: 22.96 us     ctbmv n=8 k=0: 22.99 us
```

Flat to n=512 (23.0). This is pure fixed cost.

**3. We are already better than stock ops-blas.**
Our k=0 path (23.0) beats `aclblasCcopy` (28.1) because we hoist the gather
tables with `std::call_once` and pass tiling **by value** through `<<<>>>`
rather than via a per-call GM buffer. Stock ops-blas ops do
malloc + blocking H2D memcpy + free every call (~13.8 us for the memcpy alone).

**4. The remaining gap to SIP is ~5.8 us of our own kernel setup**
(TPipe construction, several `InitBuffer` calls, GM tensor binding, gather-table
`DataCopyPad` into UB). That is the only part of the fixed cost we still
control.

## What this means for the acceptance targets

| case | target | our measured | of which fixed | of which algorithm |
|---|---|---|---|---|
| 512/8 | 10.00 | 60.4 | ~23 | ~37 |
| 1024/16 | 16.51 | 69.2 | ~23 | ~46 |
| 2048/32 | 19.77 | 100.0 | ~23 | ~77 |

**Case 1's target (10 us) is below the 14.4 us launch floor that even SIP pays.**
It is not reachable by any kernel on this device through this invocation path,
regardless of algorithm.

Cases 2 and 3 (16.51 / 19.77 us) are above the launch floor but leave only
~2-5 us for the entire banded computation after a 14.4 us launch. Our algorithm
currently uses 46-77 us, so closing that would need roughly a 15-30x improvement
in the compute portion — not plausible.

## Honest reading

Two earlier claims of mine are now superseded:

- *"~23 us is a hardware floor"* — wrong. SIP reaches 17.2 us on the same chip;
  ~5.8 us of ours is our own kernel setup, which is addressable.
- *"the gap is entirely invocation overhead"* — also wrong. At case 1, fixed
  cost is 23 of 60 us; the other 37 us is genuinely our banded algorithm.

The accurate statement: **the fixed cost alone (23 us) already exceeds two of
the three targets, and the launch component (14.4 us) exceeds the first target
even for the best invocation model on the platform.**

## The question worth asking

`asdBlasCcopy` = 17.2 us and `aclrtMemcpyAsync` = 2.8 us on 910B3, so a kernel
launch through any supported path costs ~14.4 us. How is a 10 us target for
`aclblasCtbmv` (n=512, k=8) intended to be met? Were the targets measured on
arch35 (where the merged `Ctrmv` PRs report 9.47 us at n=512), or through a
path that avoids per-call kernel launch entirely?
