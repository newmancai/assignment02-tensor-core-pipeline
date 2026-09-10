from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kda_ir import B300, cake_bt16_prepare_chain
from kda_ir.flashinfer_catalog import (
    check_frozen_export_contract,
    closest_frozen_variants,
    load_flashinfer_catalog,
    summarize_catalog,
)


class FlashInferCatalogTest(unittest.TestCase):
    def test_real_schema_subset_is_consumed(self):
        payload = {
            "schema_version": 6,
            "physical_selector_schema_version": 1,
            "variants": [
                {
                    "variant_id": "sm_103a:test",
                    "arch": "sm_103a",
                    "abi_family": "bt16_chain",
                    "state_mode": "bf16",
                    "threads": 512,
                    "smem_bytes": 138240,
                    "use_pdl": False,
                    "kernel_name": "kernel_flashkda_bf16_bt16_chain_m64",
                    "body_relpath": "csrc/kda/body.cu",
                    "body_sha256": "deadbeef",
                    "launch_contract": {"chunk_tokens": 16, "value_rows": 64},
                    "physical_selectors": [
                        {"selector_key": {"route": "bt16_prepare_chain_m64"}}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(payload))
            catalog = load_flashinfer_catalog(path)
        summary = summarize_catalog(catalog, "sm103a")
        self.assertEqual(summary["variants"], 1)
        self.assertEqual(summary["routes"], {"bt16_prepare_chain_m64": 1})
        diagnostics = check_frozen_export_contract(
            cake_bt16_prepare_chain(), B300, catalog.variants[0]
        )
        self.assertEqual({item.code for item in diagnostics}, {"KIR502", "KIR503"})
        closest = closest_frozen_variants(
            cake_bt16_prepare_chain(), B300, catalog, limit=1
        )
        self.assertEqual(closest[0]["mismatch_codes"], ["KIR502", "KIR503"])
        self.assertFalse(closest[0]["exact_visible_contract"])


if __name__ == "__main__":
    unittest.main()
