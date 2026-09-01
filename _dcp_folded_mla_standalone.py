#!/usr/bin/env python3
"""Drive the folded-MLA -> DCP decode path standalone: 1 GPU, no NCCL, no weights.

Why this exists
---------------
Every DCP failure so far was investigated by launching the full server, which
crosses five fragile systems that all present as the same idle-looking hang:

  * model loading    -- 1.5 TiB mmap against a 1.62 TiB GTT ceiling
  * ROCm/KFD         -- queue eviction under memory/userptr pressure
  * RCCL             -- collectives on this box cost ~208 ms each, so an 8-rank
                        init alone runs for minutes (measured, box-level: a bare
                        torchrun probe with no vLLM reproduces it, and raw HIP
                        peer copies are healthy at 55.7 GB/s)
  * process cleanup  -- a wedged worker cannot shut down, and the leftover
                        co-resident serve then slows the *next* run 5-10x
  * graph capture    -- persistent pointers replayed asynchronously

The most recent load wedge happened BEFORE the DCP operator ever executed, so
those launches were exercising model loading and KFD, not DCP.

The key structural fact that makes this driver possible: the aiter MLA backend
contains NO collective. ``get_dcp_group()`` appears in exactly one line
(rocm_aiter_mla.py:400, for ``_dcp_rank``) and is wrapped in try/except with a
fallback. There is no ``torch.distributed``, no all_gather, nothing. The query
all-gather across the DCP group happens in the *layer* (mla_attention.py:975),
strictly BEFORE ``forward_mqa``; the head fold, the round-robin cprr kernel call
and the un-fold are pure per-rank local arithmetic in which ``cp_world_size`` and
``cp_rank`` are just numbers feeding the causal-mask math.

So the whole folded-MLA -> DCP decode path can be driven on ONE GPU in ONE
process, with a rank simulated by passing its ``cp_rank``. That removes model
loading, RCCL, GTT pressure and multi-process teardown from the experiment in
one move, and leaves exactly the code under test.

What is real here and what is not
---------------------------------
REAL (this is the point): the production ``AiterMLAMetadataBuilder`` including
the DCP head-fold pseudo-batch construction, and the production
``AiterMLAImpl.forward_mqa`` -- the fold, the raw aiter ``mla_decode_fwd`` cprr
call, and the un-fold -- plus cudagraph capture and unsynced replay of it.

SYNTHETIC: the KV cache contents, q, and the batch shape. The impl is built with
``object.__new__`` and given exactly the attributes ``forward_mqa`` reads, so no
weight-dependent construction (``kv_b_proj`` and friends) is pulled in. The
cross-rank combine is NOT covered here -- ``k3_dcp_direct_hip/`` already tests it
standalone; this driver covers the producer of its inputs, which that test feeds
with random tensors.

Usage (inside the container)
----------------------------
    python3 /workspace/_dcp_folded_mla_standalone.py                 # rank 0
    python3 /workspace/_dcp_folded_mla_standalone.py --all-ranks     # cp_rank 0..7
    python3 /workspace/_dcp_folded_mla_standalone.py --graph         # + capture/replay

Always run under an external timeout so a fault terminates rather than traps the
GPU:

    timeout --signal=TERM --kill-after=10s 300s python3 ... _dcp_folded_mla_standalone.py
"""

import argparse
import glob
import os
import sys
import traceback

import torch


# --------------------------------------------------------------------------
# model paths (config only -- no weights are ever read)
# --------------------------------------------------------------------------
def _snapshot(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no model snapshot matched {pattern}")
    return hits[0].rstrip("/")


DEFAULT_TARGET = "/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/*/"
DEFAULT_DRAFT = "/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots/*/"


# --------------------------------------------------------------------------
# fake distributed groups: the backend needs .world_size / .rank_in_group and
# nothing else, so we never touch NCCL.
# --------------------------------------------------------------------------
class _FakeGroup:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.rank = rank
        self.local_rank = rank

    def all_gather(self, tensor, dim=0):  # only reached by weight prep, not here
        raise RuntimeError(
            "collective called in the standalone driver -- the path under test "
            "is supposed to be collective-free; this indicates a real change"
        )


def install_fake_groups(dcp_size: int, cp_rank: int, tp_size: int) -> None:
    """Point vLLM's parallel_state at rank-simulating stubs (no NCCL)."""
    import vllm.distributed.parallel_state as ps

    dcp = _FakeGroup(dcp_size, cp_rank)
    tp = _FakeGroup(tp_size, cp_rank)
    ps.get_dcp_group = lambda: dcp
    ps.get_tp_group = lambda: tp

    # The backend imported the symbol directly at module import time.
    import vllm.v1.attention.backends.mla.rocm_aiter_mla as backend

    backend.get_dcp_group = lambda: dcp


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def build_vllm_config(args):
    from vllm.engine.arg_utils import EngineArgs

    spec_cfg = {
        "model": args.draft,
        "num_speculative_tokens": args.num_spec,
        "method": "dspark",
        "attention_backend": "ROCM_AITER_MLA",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block",
    }
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        decode_context_parallel_size=args.dcp,
        cp_kv_cache_interleave_size=args.interleave,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.mnbt,
        kv_cache_dtype=args.kv_cache_dtype,
        speculative_config=spec_cfg,
        load_format="dummy",
        enforce_eager=False,
    )
    return engine_args.create_engine_config()


def build_kv_spec(vllm_config, head_size: int, non_causal: bool):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return MLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=head_size,
        dtype=vllm_config.model_config.dtype,
        non_causal_multi_token_decode=non_causal,
    )


# --------------------------------------------------------------------------
# synthetic decode batch
# --------------------------------------------------------------------------
def build_common_metadata(device, num_reqs: int, qlen: int, ctx: int, dcp: int):
    """A uniform spec-decode verify batch: num_reqs requests, qlen tokens each."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    num_tokens = num_reqs * qlen
    query_start_loc_cpu = torch.arange(
        0, num_tokens + 1, qlen, dtype=torch.int32
    )
    query_start_loc = query_start_loc_cpu.to(device)

    # seq_lens counts tokens scheduled this step, so context + the verify block.
    seq_lens_cpu = torch.full((num_reqs,), ctx + qlen, dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device)

    # block_size==1 in the flat aiter view: one page per token, pages laid out
    # contiguously per request so page ids stay in ascending position order.
    max_pages = ctx + qlen
    block_table_tensor = (
        torch.arange(num_reqs * max_pages, dtype=torch.int32, device=device)
        .reshape(num_reqs, max_pages)
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Under DCP each rank holds ~1/dcp of the context rows.
    dcp_local = torch.full(
        (num_reqs,), (ctx + qlen + dcp - 1) // dcp, dtype=torch.int32
    )

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=qlen,
        max_seq_len=int(seq_lens_cpu.max()),
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
        dcp_local_seq_lens=dcp_local.to(device),
        dcp_local_seq_lens_cpu=dcp_local,
    ), num_reqs * max_pages


# --------------------------------------------------------------------------
# impl shim: real forward_mqa, only the attributes it reads
# --------------------------------------------------------------------------
class _Layer:
    def __init__(self, device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)


class _PrefillBackendStub:
    """The base builder does ``attention_layer.prefill_backend.clone()``.

    prefill_backend is consulted only from the chunked-context prefill path
    (mla_attention.py:2696-2938), which a decode-only driver never enters.
    """

    def clone(self):
        return self


def make_ctx_layer(vllm_config, non_causal: bool):
    """Stand-in for the registered attention layer in static_forward_context.

    The base ``MLACommonMetadataBuilder.__init__`` reads the MLA dims, the
    replicated-draft flag (hunk B), ``dcp_manager`` and ``prefill_backend`` off
    this object. ``dcp_manager`` is the Direct-DCP symmetric-memory workspace and
    is used ONLY by the chunked-prefill KV all-gather -- it is constructed here
    without running its __init__ so no peer pointers are exchanged and no
    collective is issued. That is what keeps this driver single-process.
    """
    from vllm.v1.attention.ops.dcp import MLADCPManager

    hf = vllm_config.model_config.hf_text_config
    mgr = object.__new__(MLADCPManager)
    mgr.init_kv_gather = lambda *a, **k: None

    class _CtxLayer:
        pass

    layer = _CtxLayer()
    layer.non_causal_multi_token_decode = non_causal
    layer.q_lora_rank = getattr(hf, "q_lora_rank", None)
    layer.kv_lora_rank = getattr(hf, "kv_lora_rank", 512)
    layer.qk_nope_head_dim = getattr(hf, "qk_nope_head_dim", 128)
    layer.qk_rope_head_dim = getattr(hf, "qk_rope_head_dim", 64)
    layer.v_head_dim = getattr(hf, "v_head_dim", 128)
    layer.dcp_manager = mgr
    layer.prefill_backend = _PrefillBackendStub()
    return layer


def build_impl(vllm_config, num_heads: int, dcp: int, cp_rank: int, kv_dtype: str):
    """Real AiterMLAImpl.forward_mqa bound to a minimally-populated instance.

    Constructing the full impl would drag in kv_b_proj and the rest of the
    weight-dependent MLA setup, which is exactly the model-loading machinery this
    driver is separating out. forward_mqa's fold path reads only the attributes
    set below.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    hf = vllm_config.model_config.hf_text_config
    impl.kv_lora_rank = getattr(hf, "kv_lora_rank", 512)
    impl.qk_rope_head_dim = getattr(hf, "qk_rope_head_dim", 64)
    impl.num_heads = num_heads
    impl.dcp_world_size = dcp
    impl.dcp_rank = cp_rank
    impl.pcp_world_size = 1
    impl.kv_cache_dtype = kv_dtype
    impl.scale = (impl.kv_lora_rank + impl.qk_rope_head_dim) ** -0.5
    impl._decode_num_heads = num_heads * dcp
    return impl


# --------------------------------------------------------------------------
# one rank's run
# --------------------------------------------------------------------------
def torch_reference(q_f32, kv_f32, ctx, qlen, scale, positions):
    """Exact absorbed-MLA decode over an explicit set of global KV positions.

    q_f32: (num_reqs, qlen, H, D)   kv_f32: (num_reqs, L, D)
    positions: 1-D LongTensor of global positions this shard holds, ascending.
    Returns (o, lse_natural) with o: (num_reqs, qlen, H, kv_lora), lse: (...,H).
    A shard sees only `positions`; causality still uses the GLOBAL position, so
    token i (global position ctx+i) may attend to any p in positions with
    p <= ctx+i. Restricting `positions` to a round-robin residue class is
    exactly what DCP rank r sees.
    """
    nr, _, H, D = q_f32.shape
    kv_lora = kv_f32.shape[-1] - 64 if kv_f32.shape[-1] > 512 else 512
    kv_lora = 512
    o = torch.zeros(nr, qlen, H, kv_lora, dtype=torch.float32, device=q_f32.device)
    lse = torch.full(
        (nr, qlen, H), float("-inf"), dtype=torch.float32, device=q_f32.device
    )
    for r in range(nr):
        kv_sel = kv_f32[r].index_select(0, positions)          # (P, D)
        for i in range(qlen):
            valid = positions <= (ctx + i)
            if not bool(valid.any()):
                continue
            k = kv_sel[valid]                                   # (P', D)
            s = (q_f32[r, i] @ k.T) * scale                     # (H, P')
            m = s.max(dim=-1, keepdim=True).values
            p = torch.exp(s - m)
            denom = p.sum(dim=-1, keepdim=True)
            o[r, i] = (p / denom) @ k[:, :kv_lora]
            lse[r, i] = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))
    return o, lse


def merge_partials(os_, lses):
    """Standard cross-shard softmax merge (natural-log LSE)."""
    stack_lse = torch.stack(lses)                                # (R, ...)
    m = stack_lse.max(dim=0).values
    w = torch.exp(stack_lse - m)                                 # (R, ...)
    num = (w.unsqueeze(-1) * torch.stack(os_)).sum(dim=0)
    den = w.sum(dim=0).unsqueeze(-1)
    return num / den, (m + torch.log(w.sum(dim=0)))


def ensure_workspace_manager(device: torch.device) -> None:
    """Init the global workspace manager, which a serve-free driver has to do itself.

    In production `GPUModelRunner.__init__` does this before any attention
    metadata is built. `AiterMLAMetadataBuilder.__init__` now reserves the fp8
    prefill persistent-scheduling scratch through `current_workspace_manager()`
    (rocm_aiter_mla.py:647), so merely constructing a builder outside a serve
    trips its assert. That is what `--numerics` hit.

    Idempotent on purpose: `--all-ranks` builds one builder per cp_rank, and
    `run_rank()` is called in a loop, so this must be safe to call repeatedly.
    Deliberately does NOT lock the workspace -- production locks only after
    warmup, and these drivers keep allocating across shards.
    """
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if not is_workspace_manager_initialized():
        init_workspace_manager(device)


def run_numerics(args) -> bool:
    """Does the folded DCP decode + LSE actually reconstruct the answer?

    Runs all `dcp` shards sequentially on ONE GPU -- legitimate because cp_rank
    only selects which residue class of global positions the shard holds and how
    the kernel masks; there is no rank-to-rank traffic in forward_mqa. Each
    shard's asm (o, lse) is checked against an exact torch partial over the same
    positions, then the shards are merged and checked against the full-context
    torch reference. If the per-shard check passes and the merge check passes,
    the producer side of DCP is correct; the transport side is covered by
    k3_dcp_direct_hip/_test_a2a_serve_surface.py.
    """
    import aiter

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    ensure_workspace_manager(device)
    torch.manual_seed(0)

    install_fake_groups(args.dcp, 0, args.tp)
    vllm_config = build_vllm_config(args)
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata

    with set_current_vllm_config(vllm_config):
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
            AiterMLAMetadataBuilder,
        )

        head_size = args.kv_lora_rank + args.qk_rope_head_dim
        layer_name = "model.layers.0.self_attn.attn"
        vllm_config.compilation_config.static_forward_context[layer_name] = (
            make_ctx_layer(vllm_config, False)
        )
        spec = build_kv_spec(vllm_config, head_size, False)

        probe = AiterMLAMetadataBuilder(spec, [layer_name], vllm_config, device)
        num_heads = probe.num_heads
        dcp = probe.dcp_world_size
        qlen = args.qlen or probe._mtp_decode_qlen
        dcp_heads = num_heads * dcp
        nr, ctx = args.reqs, args.ctx
        L = ctx + qlen
        kv_dtype = aiter.dtypes.fp8

        # One GLOBAL kv cache: page (req*L + pos) holds request `req` position
        # `pos`. Shards select pages by residue class, so every shard reads the
        # same underlying bytes -- no re-quantisation noise between arms.
        kv_flat = (
            torch.randn(nr * L, 1, head_size, dtype=torch.float32, device=device)
            .to(kv_dtype)
        )
        q_flat = (
            torch.randn(
                nr * qlen, dcp_heads, head_size, dtype=torch.float32, device=device
            ).mul_(0.5).to(kv_dtype)
        )
        kv_f32 = kv_flat.float().view(nr, L, head_size)
        q_f32 = q_flat.float().view(nr, qlen, dcp_heads, head_size)

        scale = (args.qk_nope_head_dim + args.qk_rope_head_dim) ** -0.5
        layer = _Layer(device)

        all_pos = torch.arange(L, device=device)
        ref_o, ref_lse = torch_reference(q_f32, kv_f32, ctx, qlen, scale, all_pos)
        print(f"[numerics] reqs={nr} ctx={ctx} qlen={qlen} dcp={dcp} "
              f"heads={dcp_heads} L={L}", flush=True)

        shard_o, shard_lse = [], []
        max_shard_err = 0.0
        for r in range(dcp):
            pos = torch.arange(r, L, dcp, device=device)     # residue class r
            local_len = int(pos.numel())

            block_table = (
                torch.arange(nr, device=device).view(nr, 1) * L + pos.view(1, -1)
            ).to(torch.int32)
            seq_lens_cpu = torch.full((nr,), L, dtype=torch.int32)
            local_cpu = torch.full((nr,), local_len, dtype=torch.int32)
            common = CommonAttentionMetadata(
                query_start_loc=torch.arange(
                    0, nr * qlen + 1, qlen, dtype=torch.int32, device=device
                ),
                query_start_loc_cpu=torch.arange(
                    0, nr * qlen + 1, qlen, dtype=torch.int32
                ),
                seq_lens=seq_lens_cpu.to(device),
                num_reqs=nr,
                num_actual_tokens=nr * qlen,
                max_query_len=qlen,
                max_seq_len=L,
                block_table_tensor=block_table,
                slot_mapping=torch.arange(
                    nr * qlen, dtype=torch.int64, device=device
                ),
                causal=True,
                dcp_local_seq_lens=local_cpu.to(device),
                dcp_local_seq_lens_cpu=local_cpu,
            )

            # --sabotage tells the kernel it is rank r+1 while handing it rank
            # r's pages. Global positions then come out wrong by one, so the
            # causal mask and the round-robin reconstruction are both off. This
            # is the positive control: if the merged error does NOT blow up
            # under sabotage, the test has no power and a passing run means
            # nothing.
            kern_rank = (r + 1) % dcp if args.sabotage else r
            install_fake_groups(args.dcp, kern_rank, args.tp)
            builder = AiterMLAMetadataBuilder(
                spec, [layer_name], vllm_config, device
            )
            md = builder.build(0, common)
            impl = build_impl(
                vllm_config, num_heads, dcp, kern_rank, args.kv_cache_dtype
            )
            impl.scale = scale
            o, lse = impl.forward_mqa(q_flat, kv_flat, md, layer)
            torch.cuda.synchronize()

            o = o.float().view(nr, qlen, dcp_heads, args.kv_lora_rank)
            lse = lse.float().view(nr, qlen, dcp_heads)
            po, plse = torch_reference(q_f32, kv_f32, ctx, qlen, scale, pos)

            # Relative, because fp8-e4m3 carries 3 mantissa bits: the error
            # floor scales with the output magnitude, so an absolute gate would
            # just be a magnitude gate.
            denom_r = max(po.abs().max().item(), 1e-6)
            err = (o - po).abs().max().item() / denom_r
            max_shard_err = max(max_shard_err, err)
            print(
                f"[numerics] shard {r}: local_len={local_len} "
                f"fold={'yes' if md.decode.fold_qo_indptr is not None else 'no'} "
                f"rel|o-ref_partial|={err:.4e}",
                flush=True,
            )
            shard_o.append(o)
            shard_lse.append(lse)

        # LSE base is not documented; decide it from the data rather than assume.
        best = None
        for name, conv in (("natural", lambda x: x),
                           ("log2", lambda x: x * 0.6931471805599453)):
            mo, _ = merge_partials(shard_o, [conv(l) for l in shard_lse])
            e = (mo - ref_o).abs().max().item()
            rel = e / max(ref_o.abs().max().item(), 1e-6)
            print(f"[numerics] merge with LSE as {name:8s}: "
                  f"max|o-ref|={e:.4e}  rel={rel:.4e}", flush=True)
            if best is None or e < best[1]:
                best = (name, e, rel)

        print(f"[numerics] per-shard max REL err = {max_shard_err:.4e} "
              f"(fp8-e4m3 floor is ~6.2e-2: 3 mantissa bits)")
        print(f"[numerics] best merge          = {best[0]} "
              f"(abs {best[1]:.4e}, rel {best[2]:.4e})")
        ok = best[2] < args.rtol and max_shard_err < args.rtol_shard
        print(f"[numerics] gate merge_rel<{args.rtol} and "
              f"shard_rel<{args.rtol_shard}: {'PASS' if ok else 'FAIL'}")
        return ok


def run_rank(args, cp_rank: int) -> bool:
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    ensure_workspace_manager(device)

    install_fake_groups(args.dcp, cp_rank, args.tp)
    vllm_config = build_vllm_config(args)

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
            AiterMLAMetadataBuilder,
        )

        head_size = args.kv_lora_rank + args.qk_rope_head_dim
        spec = build_kv_spec(vllm_config, head_size, args.non_causal)

        # Hunk B reads non_causal_multi_token_decode off the LAYER via
        # static_forward_context, so register a stand-in for layer 0.
        layer_name = "model.layers.0.self_attn.attn"
        vllm_config.compilation_config.static_forward_context[layer_name] = (
            make_ctx_layer(vllm_config, args.non_causal)
        )

        builder = AiterMLAMetadataBuilder(
            spec, [layer_name], vllm_config, device
        )
        num_heads = builder.num_heads
        dcp_eff = builder.dcp_world_size
        # _mtp_decode_qlen is the production verify length. Under DSpark
        # (parallel_drafting) that is 1 + 2*num_spec = 5, not 1 + num_spec.
        # --qlen overrides it to probe the shorter shapes: qlen in {1,2} has no
        # cprr asm kernel and takes the gated unfolded branch instead.
        qlen = args.qlen or builder._mtp_decode_qlen

        print(
            f"[cp_rank={cp_rank}] num_heads={num_heads} dcp={dcp_eff} "
            f"decode_num_heads={builder._decode_num_heads} "
            f"fold_factor={builder._dcp_fold_factor} "
            f"fold_heads={builder._dcp_fold_heads} qlen={qlen}",
            flush=True,
        )

        common, num_pages = build_common_metadata(
            device, args.reqs, qlen, args.ctx, dcp_eff
        )
        md = builder.build(0, common)
        decode = md.decode
        if decode is None:
            print(f"[cp_rank={cp_rank}] FAIL: builder produced no decode metadata")
            return False
        print(
            f"[cp_rank={cp_rank}] decode: max_qo_len={decode.max_qo_len} "
            f"fold_qo_indptr={'set' if decode.fold_qo_indptr is not None else 'None'} "
            f"fold_num_reqs={decode.fold_num_reqs} "
            f"cp_world_size={decode.cp_world_size} cp_rank={decode.cp_rank} "
            f"persistent={decode.has_persistent_metadata}",
            flush=True,
        )
        if decode.fold_qo_indptr is None:
            print(
                f"[cp_rank={cp_rank}] NOTE: fold path not selected for this shape "
                f"-- forward_mqa will take the unfolded dcp branch"
            )

        # synthetic fp8 KV cache + q, sized from the metadata we just built
        import aiter

        kv_dtype = (
            aiter.dtypes.fp8
            if args.kv_cache_dtype.startswith("fp8")
            else vllm_config.model_config.dtype
        )
        kv_cache = torch.randn(
            num_pages, 1, head_size, dtype=torch.float32, device=device
        ).to(kv_dtype)

        dcp_heads = num_heads * dcp_eff
        num_tokens = args.reqs * qlen
        q = torch.randn(
            num_tokens, dcp_heads, head_size, dtype=torch.float32, device=device
        ).to(kv_dtype)

        impl = build_impl(
            vllm_config, num_heads, dcp_eff, cp_rank, args.kv_cache_dtype
        )
        layer = _Layer(device)

        # ---- eager ----
        out, lse = impl.forward_mqa(q, kv_cache, md, layer)
        torch.cuda.synchronize()
        finite_o = bool(torch.isfinite(out.float()).all())
        finite_l = lse is None or bool(torch.isfinite(lse.float()).all())
        print(
            f"[cp_rank={cp_rank}] eager OK  o={tuple(out.shape)} "
            f"lse={tuple(lse.shape) if lse is not None else None} "
            f"finite_o={finite_o} finite_lse={finite_l}",
            flush=True,
        )
        ok = finite_o and finite_l

        # ---- repeat unsynced (the async-fault surface) ----
        for _ in range(args.iters):
            impl.forward_mqa(q, kv_cache, md, layer)
        torch.cuda.synchronize()
        print(f"[cp_rank={cp_rank}] {args.iters} unsynced iters OK", flush=True)

        # ---- cudagraph capture + replay ----
        if args.graph:
            for _ in range(3):
                impl.forward_mqa(q, kv_cache, md, layer)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                g_out, g_lse = impl.forward_mqa(q, kv_cache, md, layer)
            for _ in range(args.graph_iters):
                graph.replay()
            torch.cuda.synchronize()
            g_ok = bool(torch.isfinite(g_out.float()).all())
            print(
                f"[cp_rank={cp_rank}] graph capture + {args.graph_iters} "
                f"replays OK finite={g_ok}",
                flush=True,
            )
            ok = ok and g_ok

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--dcp", type=int, default=8)
    ap.add_argument("--num-spec", type=int, default=2)
    ap.add_argument("--reqs", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--graph-iters", type=int, default=50)
    ap.add_argument("--all-ranks", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--non-causal", action="store_true",
                    help="simulate the replicated DSpark draft group (#51705)")
    ap.add_argument("--qlen", type=int, default=0,
                    help="override the verify length (0 = production value)")
    ap.add_argument("--interleave", type=int, default=1,
                    help="cp_kv_cache_interleave_size; !=1 drops "
                         "supports_dcp_with_varlen and forces qlen to 1")
    ap.add_argument("--numerics", action="store_true",
                    help="cross-shard equivalence vs an exact torch reference")
    ap.add_argument("--rtol", type=float, default=5e-2)
    ap.add_argument("--rtol-shard", type=float, default=1.0e-1)
    ap.add_argument("--sabotage", action="store_true",
                    help="positive control: feed shard r with cp_rank r+1")
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--kv-lora-rank", type=int, default=512)
    ap.add_argument("--qk-rope-head-dim", type=int, default=64)
    ap.add_argument("--qk-nope-head-dim", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--mnbt", type=int, default=4096)
    args = ap.parse_args()

    if args.model is None:
        args.model = _snapshot(DEFAULT_TARGET)
    if args.draft is None:
        args.draft = _snapshot(DEFAULT_DRAFT)

    if args.numerics:
        try:
            ok = run_numerics(args)
        except Exception:
            traceback.print_exc()
            ok = False
        print()
        print(f"VERDICT: {'PASS' if ok else 'FAIL'} (numerics)")
        return 0 if ok else 1

    ranks = range(args.dcp) if args.all_ranks else [args.rank]
    failures = []
    for cp_rank in ranks:
        try:
            if not run_rank(args, cp_rank):
                failures.append(cp_rank)
        except Exception:
            traceback.print_exc()
            failures.append(cp_rank)
            break

    print()
    if failures:
        print(f"VERDICT: FAIL (cp_ranks {failures})")
        return 1
    print(f"VERDICT: PASS (cp_ranks {list(ranks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
