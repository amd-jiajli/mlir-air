# Multi-Core GEMV BF16: Design Notes

## Overview

The single-core GEMV computes `C[M] = A[M,K] @ B[K]` with bf16 precision.
Each launch instance processes `tile_m_l2` output rows on one AIE core.
The multi-core extension distributes rows across `num_cores` cores (up to 8
columns on NPU2/Strix), with each core independently computing `tile_m_l2`
rows per launch instance.

### Performance

| Config (M, K) | Cores | tile_m_l2 | Avg Time | Avg GFLOPS | Speedup |
|----------------|-------|-----------|----------|------------|---------|
| 2048, 8192 | 1 | 4 | 7983 us | 4.20 | 1.0x |
| 2048, 8192 | 8 | 2 | 2913 us | 11.52 | 2.74x |
| 8192, 2048 | 1 | 16 | 8013 us | 4.19 | 1.0x |
| 8192, 2048 | 8 | 4 | 5323 us | 6.30 | 1.50x |

Speedup is below the theoretical 8x due to external memory bandwidth
saturation — GEMV is memory-bound (O(MK) data, O(MK) compute).

---

## Architecture

```
L3 ──A──> L2 (per-column memtile, split by air-split-l2-memref)
                    │
                    ├── core 0: L2[0:tile_m_l2] ──> L1
                    ├── core 1: L2[tile_m_l2:2*tile_m_l2] ──> L1
                    └── ...

L3 ──B──> L1 (each core loads independently through own memtile)

L1 ──C──> L3 (each core writes directly at launch_offset_m + _tx * tile_m_l2)
```

Three data paths, all avoiding shared-memtile bottlenecks:
- **A:** L3 → L2 (single bulk transfer at segment level, compiler splits
  across memtiles) → L1 (per-core slice inside herd)
- **B:** L3 → L1 (per-core direct, compiler routes through each column's
  memtile automatically)
- **C:** L1 → L3 (per-core direct, each core writes at its computed global
  offset)

The design uses L2 only for A (where the large matrix tile benefits from L2
buffering) and eliminates L2 staging for B and C entirely.

---

## Problems Encountered During Scaling

### Problem 1: Broadcast Detection + DMA-to-Channel Interaction

**Symptom:** Dominance error during `aircc` pass pipeline — `operand #1 does
not dominate this use` in `air.wait_all`.

**Root Cause:** The B vector DMA from L2 to L1 is identical for all cores
(same source, same size, no per-core offset). The `air-broadcast-detection`
pass recognizes this and wraps the DMA in an `affine.if` with a
`broadcast_set` constraint, so only one core initiates the transfer while
others receive via hardware broadcast:

```mlir
affine.if #broadcast_set(%tx) {
  %token = air.dma_memcpy_nd(...)  // only core 0 executes
}
```

The subsequent `air-dma-to-channel` pass converts DMAs to channel operations
but cannot properly handle the async token `%token` defined inside the
`affine.if` region — it references it outside the region, violating SSA
dominance.

**Isolation method:** Run the aircc passes incrementally:
- `air-dependency` → OK
- `air-broadcast-detection` → OK
- `air-specialize-dma-broadcast` → OK
- `air-dma-to-channel` → **FAILS**

**Fix:** Set `omit_auto_broadcast=True` for multi-core configs. This skips
both `air-broadcast-detection` and `air-specialize-dma-broadcast` passes
entirely. Each core independently loads B, which is acceptable since B is
small and this also avoids the DMA channel exhaustion in Problem 3.

---

### Problem 2: DMA Buffer Descriptor 4-Byte Alignment

**Symptom:** `'aie.dma_bd' op transfer length must be multiple of 4` for
`memref<1xbf16>`.

**Root Cause:** AIE2P DMA buffer descriptors require transfer lengths
representing 4-byte-aligned addresses. With bf16 (2 bytes), transferring 1
element = 2 bytes violates the 4-byte minimum.

**Constraint:** `tile_m_l2 >= 2` for bf16 (2 elements * 2 bytes = 4 bytes).

**Impact:** The original plan targeted `tile_m_l2=1, num_cores=8` for
M=2048, K=8192. This is not possible. We use `tile_m_l2=2, num_cores=8`
instead (L2 A: 8*2*8192*2 = 256KB, fitting exactly in one memtile partition).

---

### Problem 3: MemTile DMA Channel Exhaustion

**Symptom:** Compilation succeeds, but core 0 produces all zeros while cores
1-7 produce correct results.

**Root Cause:** The initial design staged B in a single L2 buffer on one
memtile (`mem_tile_0_1`). Distributing B to 8 cores requires 8 MM2S DMA
channels on that memtile, but **memtiles have only 6 MM2S DMA channels**
(`XAIE_MEM_TILE_DMA_NUM_CH = 6`).

The compiler allocates channels 0-5 for the first 6 cores' B transfers.
For cores 6 and 7, it reuses DMA channel 0 via multi-destination flow. But
channel 0 is also needed for core 0's A data transfer. The resulting flows
conflict:

```mlir
aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)  // B to core 0
aie.flow(%mem_tile_0_1, DMA : 0, %tile_6_2, DMA : 0)  // B to core 6
aie.flow(%mem_tile_0_1, DMA : 0, %tile_7_2, DMA : 0)  // B to core 7
aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 1)  // A to core 0 !!
```

The BD chain for DMA channel 0 interleaves B and A transfers, causing core 0
to receive corrupted data.

**Diagnosis:** Inspected `aie.air.mlir` flow declarations and memtile DMA
configs. The mismatch pattern (only core 0 wrong) pointed to a routing issue
on the column hosting the shared B buffer.

**Fix:** Eliminated L2 B buffer. Each core loads B directly from L3 through
its own column's memtile. This distributes the load across 8 memtiles (1
channel each), well within the 6-channel limit per memtile.

**Trade-off:** 8x DDR bandwidth for B. Since B is small (K*2 bytes) relative
to A (tile_m_l2*K*2 bytes per core), the overhead is negligible.

---

### Problem 4: Missing L2→L3 C Writeback Dependency

**Symptom:** Compilation succeeds, all output values are zero (100% mismatch).

**Root Cause:** The initial design had per-core C results written to a shared
L2 buffer, then a single segment-level DMA transferred L2 C to L3. After
`air-dma-to-channel` conversion:

- Per-core L1→L2 writeback becomes `channel_4.put` (herd) / `channel_4.get`
  (segment, writes L2 C)
- L2→L3 writeback becomes `channel_5.put` (segment, reads L2 C)

The `channel_5.put` has **no dependency** on `channel_4.get`. The dependency
pass tracks memref data flow, but after channel conversion the two channel
operations share the same underlying memref through different channels — the
compiler treats them as independent.

Result: `channel_5.put` fires immediately and reads uninitialized L2 C data
before the cores finish writing.

**Why single-core worked:** With 1 core, all DMA operations route through one
memtile. Implicit serialization on that memtile ensures ordering. With
multiple columns, this implicit ordering breaks.

**Fix:** Eliminated L2 C buffer. Each core writes C directly from L1 to L3 at
its global offset (`launch_offset_m + _tx * tile_m_l2`). The per-core DMA
inside the herd body naturally depends on the compute loop completion via the
SSA dependency chain.

**Implementation detail:** `launch_offset_m` is computed at segment level
(via `affine.apply`) and passed as an explicit `index`-typed operand to the
herd, alongside the L3 C memref. Inside the herd:

```python
# Global C offset: launch_offset_m + _tx * tile_m_l2
c_global_offset_map = AffineMap.get(0, 2, [
    AffineExpr.get_add(
        AffineSymbolExpr.get(0),         # launch_offset_m
        AffineExpr.get_mul(
            AffineSymbolExpr.get(1),     # _tx
            AffineConstantExpr.get(tile_m_l2)
        )
    )
])
```

---

## Usage

```bash
# Single-core (default)
make run M=2048 K=8192 TILE_M_L2=4 M_INPUT=1

# Multi-core (8 cores)
make run M=2048 K=8192 TILE_M_L2=2 M_INPUT=1 NUM_CORES=8

# Convenience targets
make run1_multi     # M=2048 K=8192 8-core
make profile1_multi # M=2048 K=8192 8-core profiling
make run2_multi     # M=8192 K=2048 8-core
make profile2_multi # M=8192 K=2048 8-core profiling
```

### Constraints

- `M % (tile_m_l2 * num_cores) == 0`
- `tile_m_l2 >= 2` (DMA 4-byte alignment for bf16)
- `num_cores * tile_m_l2 * K * 2 <= 256KB` (L2 A buffer capacity)
- `num_cores <= 8` (NPU2 has 8 columns)
