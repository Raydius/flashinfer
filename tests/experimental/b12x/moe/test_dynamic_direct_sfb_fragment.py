"""Direct-SFB NVFP4 dynamic MoE kernels compile with the four-N-warp atom.

DenseGemmKernel._partition_fragment_SFB separates the N and K modes of the
(1, 4, 1)-atom NVFP4 scale fragment itself; the dynamic kernel's direct_sfb
paths must not flatten that fragment a second time, for the main FC1/FC2
fragment and for the swapped FC1 fragment a 32-aligned, non-128-aligned
intermediate width selects.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused_moe import _impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12,
    reason="compiles SM12x dynamic MoE kernels",
)


@pytest.mark.parametrize("n", [1024, 1056])
def test_four_warp_nvfp4_direct_sfb_kernel_compiles(n):
    """n=1024 keeps FC1 unswapped; n=1056 (32- but not 128-aligned) swaps FC1."""
    properties = torch.cuda.get_device_properties(0)
    compiled = _impl._get_dynamic_kernel(
        8,
        64,
        2048,
        n,
        2,
        128,
        topk_ids_dtype=torch.int32,
        fast_math=True,
        mac_override=properties.multi_processor_count,
        activation="silu",
        quant_mode="nvfp4",
        mma_warps=4,
        planned_tile_m=16,
    )
    assert compiled is not None
