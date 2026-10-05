"""Route the K3 latent-MoE tail through AITER allreduce_fusion_kernel_1stage.

The tail is ``allreduce(partial)`` then a plain RMSNorm. That is the contract
of ``fused_allreduce_rms_norm`` with a zero residual. ``AITER_AR_1STAGE=1``
skips the byte cap so the one-stage kernel is selected whenever the shape
contract (bf16/fp16, <=80 tokens, hidden packs <=1024) holds.
"""

from __future__ import annotations

import os
import pathlib
import sys


def _vllm_root() -> pathlib.Path:
    import vllm

    return pathlib.Path(vllm.__file__).resolve().parent


def _patch_latent(path: pathlib.Path) -> None:
    text = path.read_text()
    old = """        transform = self.routed_output_transform
        assert transform is not None

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)
"""
    new = """        transform = self.routed_output_transform
        assert transform is not None

        # allreduce + RMSNorm in one kernel (allreduce_fusion_kernel_1stage)
        # for decode-sized batches. Zero residual matches the previous plain
        # RMSNorm(allreduce(partial)). Prefill keeps allreduce then RMSNorm.
        if (
            transform.norm is not None
            and fused_output.dim() == 2
            and fused_output.shape[0] <= _AR1S_MAX_TOKENS
        ):
            from vllm.models.common.ops.fused_allreduce_rms_norm import (
                fused_allreduce_rms_norm,
            )

            latent, _ = fused_allreduce_rms_norm(
                fused_output, _ar1s_zero(fused_output), transform.norm
            )
        else:
            latent = tensor_model_parallel_all_reduce(fused_output)
            if transform.norm is not None:
                latent = transform.norm(latent)
"""
    helper = '''
# The 1-stage kernel takes at most 80 tokens. One zero residual per
# (dtype, device, hidden) is shared by every layer and sliced per batch.
_AR1S_MAX_TOKENS = 80
_AR1S_ZEROS: dict = {}


def _ar1s_zero(like: torch.Tensor) -> torch.Tensor:
    key = (like.dtype, like.device, like.shape[-1])
    buf = _AR1S_ZEROS.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "latent-MoE 1-stage zero residual was not allocated before capture"
            )
        buf = torch.zeros(
            (_AR1S_MAX_TOKENS, like.shape[-1]), dtype=like.dtype, device=like.device
        )
        _AR1S_ZEROS[key] = buf
    return buf[: like.shape[0]]

'''
    if new in text and helper in text:
        print(f"latent tail already fused: {path}")
        return
    if "_ar1s_zero" in text:
        raise SystemExit(
            f"{path} has an older 1-stage patch; restore the image file first"
        )
    if old not in text:
        raise SystemExit(f"latent tail site not found in {path}")
    needle = "\n\nclass ROCmLatentMoERunner(MoERunner):\n"
    if needle not in text:
        raise SystemExit(f"ROCmLatentMoERunner not found in {path}")
    text = text.replace(needle, "\n" + helper + "\nclass ROCmLatentMoERunner(MoERunner):\n", 1)
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"fused latent-MoE tail allreduce+RMSNorm: {path}")


def _patch_one_stage_gate(path: pathlib.Path) -> None:
    text = path.read_text()
    old = """        if hidden_dim % pack_size != 0 or hidden_dim // pack_size > 1024:
            return False
        ca = self._impl
"""
    new = """        if hidden_dim % pack_size != 0 or hidden_dim // pack_size > 1024:
            return False
        # AITER_AR_1STAGE=1 keeps the kernel shape contract and skips the
        # byte cap, matching aiter's fused_allreduce_rmsnorm override.
        override = os.environ.get("AITER_AR_1STAGE")
        ca = self._impl
        if override == "0":
            return False
        if override == "1":
            return ca.world_size == 2 or ca.fully_connected
        ca = self._impl
"""
    if "AITER_AR_1STAGE" in text and "skips the" in text:
        print(f"1-stage gate already honors AITER_AR_1STAGE: {path}")
        return
    if old not in text:
        raise SystemExit(f"1-stage gate site not found in {path}")
    if "import os" not in text:
        text = text.replace(
            "import torch\n",
            "import os\n\nimport torch\n",
            1,
        )
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"AITER_AR_1STAGE now forces allreduce_fusion_kernel_1stage: {path}")


def main() -> None:
    root = _vllm_root()
    _patch_latent(root / "models/kimi_k3/amd/latent_moe_runner.py")
    _patch_one_stage_gate(
        root / "distributed/device_communicators/aiter_custom_all_reduce.py"
    )
    # Import-check the edited modules.
    import ast

    for rel in (
        "models/kimi_k3/amd/latent_moe_runner.py",
        "distributed/device_communicators/aiter_custom_all_reduce.py",
    ):
        ast.parse((root / rel).read_text())
    print("ar1s patch ok", os.environ.get("AITER_AR_1STAGE", ""))


if __name__ == "__main__":
    sys.exit(main())
