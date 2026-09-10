from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from kda_ir import B300
from kda_ir.flashinfer_catalog import FrozenVariant
from kda_ir.frozen_source import parse_frozen_source, check_source_variant_contract
from kda_ir.toolchain import (
    ToolchainPlanError,
    check_compiled_resources,
    nvcc_compile_plan,
    parse_cuobjdump_resources,
)


SOURCE = b"""#define TMEM_NCOLS 256
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_TOTAL 227968
__global__ __launch_bounds__(1024) void kernel_flashkda_bf16_fused_m128() {}
// --- pipeline 'chunk_pipe' ---
// qk_full: 5 barriers, init_count=1
// qk_raw_full: 5 barriers, init_count=1
// smem_free: 5 barriers, init_count=4
// v_free: 5 barriers, init_count=4
// final_ready: 5 barriers, init_count=1
// --- pipeline 'checkpoint_pipe' ---
// checkpoint_ready: 2 barriers, init_count=4
// ---- Role: compute ----
if (warp <= 3) {
  mbarrier_wait(qk_full_addr, 0);
  mbarrier_wait(final_ready_addr, 0);
  if (elect_sync()) { mbarrier_arrive(smem_free_addr); }
}
// ---- Role: epilogue ----
} else if (warp >= 4 && warp <= 7) {}
// ---- Role: beta_prefetch ----
} else if (warp == 8) {}
// ---- Role: aux_mma ----
} else if (warp >= 10 && warp <= 11) {
  if (elect_sync()) { mbarrier_arrive(qk_full_addr); }
}
// ---- Role: mma ----
} else if (warp == 9) {
  mbarrier_wait(qk_full_addr, 0);
  if (elect_sync()) { mbarrier_arrive(final_ready_addr); }
}
// ---- Role: prep ----
} else if (warp >= 12 && warp <= 31) {
  mbarrier_wait(smem_free_addr, 0);
}
"""


class FrozenSourceTest(unittest.TestCase):
    def test_recovers_exact_export_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.cu"
            path.write_bytes(SOURCE)
            source = parse_frozen_source(path)
        self.assertEqual(source.launch_threads, 1024)
        self.assertEqual(source.chunk_tokens, 32)
        self.assertEqual(source.value_rows, 128)
        self.assertEqual(
            [(x.name, x.warps) for x in source.roles],
            [
                ("compute", 4),
                ("epilogue", 4),
                ("beta_prefetch", 1),
                ("aux_mma", 2),
                ("mma", 1),
                ("prep", 20),
            ],
        )
        self.assertEqual(sum(x.stages for x in source.barriers), 27)
        self.assertIn(
            ("compute", "smem_free", "arrive", True),
            {
                (use.role, use.barrier, use.action, use.elect_one)
                for use in source.barrier_uses
            },
        )

        variant = FrozenVariant(
            variant_id="sm_103a:test",
            arch="sm_103a",
            abi_family="direct_m128",
            state_mode="bf16",
            threads=1024,
            smem_bytes=227968,
            use_pdl=False,
            kernel_name="kernel_flashkda_bf16_fused_m128",
            body_relpath="csrc/kda/body.cu",
            body_sha256=hashlib.sha256(SOURCE).hexdigest(),
            launch_contract={"chunk_tokens": 32, "value_rows": 128},
            selector_keys=(),
        )
        self.assertEqual(check_source_variant_contract(source, variant), ())
        plan = nvcc_compile_plan(source, "sm100a", "body.cu", "body.o")
        self.assertIn("-gencode=arch=compute_100a,code=sm_100a", plan)
        with self.assertRaises(ToolchainPlanError):
            nvcc_compile_plan(source, "sm_100", "body.cu", "body.o")

        compiled = parse_cuobjdump_resources(
            """arch = sm_103a
Resource usage:
 Function kernel_flashkda_bf16_fused_m128:
  REG:64 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:1264
"""
        )
        self.assertEqual(compiled.static_shared_bytes, 1024)
        self.assertEqual(check_compiled_resources(source, B300, compiled), ())


if __name__ == "__main__":
    unittest.main()
