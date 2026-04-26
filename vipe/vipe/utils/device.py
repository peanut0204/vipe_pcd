# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CudaCompatibility:
    cuda_available: bool
    device_index: int | None = None
    device_name: str | None = None
    device_capability: tuple[int, int] | None = None
    device_arch: str | None = None
    torch_arch_list: list[str] | None = None

    @property
    def is_supported(self) -> bool:
        if not self.cuda_available:
            return False
        if self.device_arch is None:
            return False
        if not self.torch_arch_list:
            # If arch list is unavailable, be conservative and assume CUDA works.
            return True
        return self.device_arch in self.torch_arch_list


def _torch_arch_list() -> list[str] | None:
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if get_arch_list is None:
        return None
    try:
        return list(get_arch_list())
    except Exception:
        return None


def get_cuda_compatibility() -> CudaCompatibility:
    if not torch.cuda.is_available():
        return CudaCompatibility(cuda_available=False)

    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        device_arch = f"sm_{capability[0]}{capability[1]}"
    except Exception:
        return CudaCompatibility(cuda_available=True)

    return CudaCompatibility(
        cuda_available=True,
        device_index=device_index,
        device_name=device_name,
        device_capability=capability,
        device_arch=device_arch,
        torch_arch_list=_torch_arch_list(),
    )


def ensure_cuda_supported_or_explain(*, env_var_allow: str = "VIPE_ALLOW_UNSUPPORTED_CUDA") -> None:
    """Fail fast with a clear message if the installed torch wheel cannot run on the detected GPU.

    This prevents confusing downstream errors like:
      RuntimeError: CUDA error: no kernel image is available for execution on the device

    The primary failure mode we target is using a modern PyTorch CUDA wheel compiled
    only for newer GPU architectures (e.g. sm_75+) on older GPUs (e.g. V100 sm_70).
    """

    if os.environ.get(env_var_allow, "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return

    compat = get_cuda_compatibility()
    if not compat.cuda_available:
        return

    if compat.is_supported:
        return

    # Note: `torch.__version__` includes the CUDA build tag (e.g. +cu128).
    torch_ver = getattr(torch, "__version__", "unknown")
    arch_list = ", ".join(compat.torch_arch_list or []) or "(unknown)"
    gpu = compat.device_name or "(unknown GPU)"
    arch = compat.device_arch or "(unknown arch)"

    raise RuntimeError(
        "Detected a CUDA GPU, but the installed PyTorch build cannot execute kernels on it.\n\n"
        f"- GPU: {gpu} ({arch})\n"
        f"- PyTorch: {torch_ver}\n"
        f"- PyTorch CUDA arch list: {arch_list}\n\n"
        "Fix options:\n"
        "1) Install a PyTorch CUDA wheel that includes your GPU architecture (e.g. V100 needs sm_70).\n"
        "   For many environments, installing a CUDA 11.8 (cu118) wheel is the simplest approach.\n"
        "   Example (adjust versions as needed):\n"
        "     pip uninstall -y torch torchvision\n"
        "     pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision\n\n"
        "2) Build PyTorch from source with `TORCH_CUDA_ARCH_LIST=7.0` (for V100).\n"
        "3) Run on a GPU with compute capability >= 7.5 (e.g. T4/A10/RTX 20+).\n\n"
        f"If you really want to proceed anyway, set `{env_var_allow}=1` (not recommended)."
    )
