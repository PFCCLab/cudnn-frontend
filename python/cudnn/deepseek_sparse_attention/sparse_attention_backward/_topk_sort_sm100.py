# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascending per-row top-k sort for the deterministic SM100 DSA backward.

Deterministic relay ranks wait far less when every top-k row is ascending.
One CTA sorts one row. Invalid slots (negative indices, and with
``topk_length`` every slot at or past the row length) sort as INT32_MAX and
are written back as -1, so they stay at the row tail. The output equals
``sort(where(idx >= 0, idx, INT32_MAX))`` followed by INT32_MAX -> -1 for every
input, duplicates included: a sorted multiset is unique.

A row of width ``T`` is split into a power-of-two head ``P <= T`` and a tail
padded to a power of two. Each part is sorted by a bitonic network whose
stages are grouped ``LOG_E`` at a time: a thread loads ``E = 2**LOG_E`` keys
that only differ in the group's index bits, runs those stages in registers,
and stores them back, so shared memory is touched once per group instead of
once per stage. Head and tail are then merged by binary-search ranks.
"""

from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cudnn._cutlass_compat import SmemAllocator
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options

_SENTINEL = 0x7FFFFFFF


class TopkRowSortSm100:
    """Sort each ``(rows, topk)`` int32 row ascending, invalid slots last as -1."""

    LOG_E = 4

    def __init__(self, topk: int):
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        self.topk = topk
        self.log_head = topk.bit_length() - 1
        self.head = 1 << self.log_head
        rest = topk - self.head
        self.log_tail = (rest - 1).bit_length() if rest else -1
        self.tail = (1 << self.log_tail) if rest else 0
        self.threads = max(32, self.head >> self.LOG_E)

    def _passes(self, log_n: int):
        """Register passes (lo, m, kk0, kk1) of a 2**log_n bitonic sort: keys differing in index bits
        [lo, lo+m) are loaded together and merge stages kk0..kk1-1 run on those bits in registers."""
        first = min(self.LOG_E, log_n)
        passes = [(0, first, 0, first)]
        for kk in range(first, log_n):
            hi = kk
            while hi >= 0:
                m = min(self.LOG_E, hi + 1)
                passes.append((hi - m + 1, m, kk, kk + 1))
                hi -= m
        return passes

    @staticmethod
    def _at(i):
        """Shared-memory slot of key i: one pad word per 32 keys breaks the strided bank conflicts."""
        return i + (i >> 5)

    @cute.jit
    def _sort_segment(self, keys: cute.Tensor, reg: cute.Tensor, tidx: Int32, base: cutlass.Constexpr, log_n: cutlass.Constexpr):
        """In-place ascending bitonic sort of keys[base : base + 2**log_n]."""
        passes = self._passes(log_n)
        for pi in cutlass.range_constexpr(len(passes)):
            lo, m, kk0, kk1 = passes[pi]
            e = 1 << m
            groups = (1 << log_n) >> m
            for it in cutlass.range_constexpr((groups + self.threads - 1) // self.threads):
                g = tidx + Int32(it * self.threads)
                if g < Int32(groups):
                    gbase = ((g >> lo) << (lo + m)) | (g & Int32((1 << lo) - 1))
                    for x in cutlass.range_constexpr(e):
                        reg[x] = keys[self._at(Int32(base) + (gbase | Int32(x << lo)))]
                    for kk in cutlass.range_constexpr(kk0, kk1):
                        upd = (gbase & Int32(2 << kk)) == Int32(0)
                        for jr in cutlass.range_constexpr(min(kk, lo + m - 1) - lo + 1):
                            b = min(kk, lo + m - 1) - lo - jr
                            for x in cutlass.range_constexpr(e):
                                if cutlass.const_expr(((x >> b) & 1) == 0):
                                    y = x | (1 << b)
                                    va = reg[x]
                                    vc = reg[y]
                                    lo_v = va if va < vc else vc
                                    hi_v = vc if va < vc else va
                                    if cutlass.const_expr(kk + 1 == log_n):
                                        reg[x] = lo_v
                                        reg[y] = hi_v
                                    elif cutlass.const_expr(kk + 1 < lo + m):
                                        # The direction bit is one of this group's own bits: a constant of x.
                                        if cutlass.const_expr(((x >> (kk + 1 - lo)) & 1) == 0):
                                            reg[x] = lo_v
                                            reg[y] = hi_v
                                        else:
                                            reg[x] = hi_v
                                            reg[y] = lo_v
                                    else:
                                        reg[x] = lo_v if upd else hi_v
                                        reg[y] = hi_v if upd else lo_v
                    for x in cutlass.range_constexpr(e):
                        keys[self._at(Int32(base) + (gbase | Int32(x << lo)))] = reg[x]
            cute.arch.sync_threads()

    @cute.jit
    def __call__(self, mInput: cute.Tensor, mOutput: cute.Tensor, mTopkLength: Optional[cute.Tensor], stream: cuda.CUstream):
        self.kernel(mInput, mOutput, mTopkLength).launch(grid=(mInput.shape[0], 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mInput: cute.Tensor, mOutput: cute.Tensor, mTopkLength: Optional[cute.Tensor]):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        sentinel = Int32(_SENTINEL)
        total = self.head + self.tail
        smem = SmemAllocator()
        keys = smem.allocate_tensor(Int32, cute.make_layout((total + total // 32 + 1,)), byte_alignment=128)
        reg = cute.make_rmem_tensor((1 << self.LOG_E,), Int32)

        length = Int32(self.topk)
        if cutlass.const_expr(mTopkLength is not None):
            length = mTopkLength[row]
        for item in cutlass.range_constexpr((total + self.threads - 1) // self.threads):
            pos = tidx + Int32(item * self.threads)
            if pos < Int32(total):
                value = sentinel
                if pos < length and pos < Int32(self.topk):
                    index = mInput[row, pos]
                    if index >= Int32(0):
                        value = index
                keys[self._at(pos)] = value
        cute.arch.sync_threads()

        self._sort_segment(keys, reg, tidx, 0, self.log_head)
        head_items = (self.head + self.threads - 1) // self.threads
        if cutlass.const_expr(self.tail == 0):
            for item in cutlass.range_constexpr(head_items):
                pos = tidx + Int32(item * self.threads)
                if pos < Int32(self.head):
                    value = keys[self._at(pos)]
                    mOutput[row, pos] = Int32(-1) if value == sentinel else value
        else:
            self._sort_segment(keys, reg, tidx, self.head, self.log_tail)
            # Merge: a head key goes after the tail keys smaller than it, a tail key after the head keys
            # not larger than it (head first on ties). Positions >= topk only hold padding sentinels.
            for item in cutlass.range_constexpr(head_items):
                hi_ = tidx + Int32(item * self.threads)
                if hi_ < Int32(self.head):
                    hv = keys[self._at(hi_)]
                    hpos = Int32(0)
                    for lb in cutlass.range_constexpr(self.log_tail - 1, -1, -1):
                        hprobe = keys[self._at(Int32(self.head - 1 + (1 << lb)) + hpos)]
                        hpos = hpos + (Int32(1 << lb) if hprobe < hv else Int32(0))
                    hpos = hpos + (Int32(1) if keys[self._at(Int32(self.head) + hpos)] < hv else Int32(0))
                    hdst = hi_ + hpos
                    if hdst < Int32(self.topk):
                        mOutput[row, hdst] = Int32(-1) if hv == sentinel else hv
            for item in cutlass.range_constexpr((self.tail + self.threads - 1) // self.threads):
                tj = tidx + Int32(item * self.threads)
                if tj < Int32(self.tail):
                    tv = keys[self._at(Int32(self.head) + tj)]
                    tpos = Int32(0)
                    for lb in cutlass.range_constexpr(self.log_head - 1, -1, -1):
                        tprobe = keys[self._at(Int32((1 << lb) - 1) + tpos)]
                        tpos = tpos + (Int32(1 << lb) if tprobe <= tv else Int32(0))
                    tpos = tpos + (Int32(1) if keys[self._at(tpos)] <= tv else Int32(0))
                    tdst = tj + tpos
                    if tdst < Int32(self.topk):
                        mOutput[row, tdst] = Int32(-1) if tv == sentinel else tv

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@lru_cache(maxsize=None)
def compiled_topk_row_sort(topk: int, has_topk_length: bool):
    """Compile once per (row width, topk_length presence); the row count is dynamic."""
    rows = cute.sym_int()
    fake = lambda shape: cute.runtime.make_fake_compact_tensor(cutlass.Int32, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=4)
    lengths = fake((rows,)) if has_topk_length else None
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(TopkRowSortSm100(topk), fake((rows, topk)), fake((rows, topk)), lengths, stream, options=compile_options())
