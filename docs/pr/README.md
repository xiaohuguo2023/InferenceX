# Kimi-K3 fp8-ASM uncapped — PR artifacts

Prepared, ready-to-file artifacts for upstreaming the uncapped-fp8 K3 work. Master plan and
upstream-PR landscape: [`../kimik3_fp8_uncapped_pr_plan.md`](../kimik3_fp8_uncapped_pr_plan.md).

| file | PR | what |
|---|---|---|
| `vllm_pra_fp8_prefill_pad16.diff` | **PR-A** (vLLM) | Unified diff (L3+L4): FP8 ASM MLA prefill for non-divisor heads (pad-to-16) + PS metadata `num_head_k = max(16, num_heads)`. 7 hunks, round-trip-verified. |
| `vllm_pra_body.md` | **PR-A** (vLLM) | Paste-ready PR description. Stacks on #50578; composes with #48712. |
| `aiter_pr_notes.md` | **PR-B / PR-C** (aiter) | PR-B: land #4452 (our cherry-pick is byte-identical) + K3 validation comment. PR-C: bf16 tuned GEMM config (blocked on tuner run). |

## Status / adopt-vs-file

- **PR-A** — file (novel). Base commit was a local nightly (`5f76ae224`); rebase onto current
  `main` + #50578 before opening. Diff applies as-is against a tree that already has #50578.
- **#50578** (decode pad + mxfp4 MoE) — **adopt directly**, drop our local decode edit. It is
  the base for PR-A and also auto-sets `AITER_BF16_FP8_MOE_BOUND=0`.
- **#50618** (wvSplitK strided fix) — **adopt directly**; needed for full cudagraph capture.
- **PR-B / #4452** — validate & push; our branch `~/work/aiter @ 6fc5733b7` is the cherry-pick.
- **PR-C** — file only after the `k3_gemm_tune/` tuner produces the CSV on gfx950 with the
  matching aiter build.

## Not yet done

- Rebase container `rocm_aiter_mla.py` decode section onto #50578 so what we run == what we'd
  file (optional; PR-A diff is already isolated from the decode path).
- Run the GEMM tuner → `kimik3_bf16_tuned_gemm.csv` (PR-C).
- No PRs are filed yet — these are prepared artifacts only.
