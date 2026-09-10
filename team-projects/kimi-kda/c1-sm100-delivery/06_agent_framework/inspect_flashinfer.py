from __future__ import annotations

import argparse
import json

from kda_ir import B300, cake_bt16_prepare_chain, official_flashkda, value_sliced_v16
from kda_ir.flashinfer_catalog import (
    closest_frozen_variants,
    load_flashinfer_catalog,
    summarize_catalog,
)


SCHEDULES = {
    "official": official_flashkda,
    "value-sliced": value_sliced_v16,
    "cake-class": cake_bt16_prepare_chain,
}


parser = argparse.ArgumentParser()
parser.add_argument("metadata")
parser.add_argument("--arch")
parser.add_argument("--schedule", choices=sorted(SCHEDULES))
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()
catalog = load_flashinfer_catalog(args.metadata)
result = {"catalog": summarize_catalog(catalog, args.arch)}
if args.schedule:
    result["closest_visible_contracts"] = closest_frozen_variants(
        SCHEDULES[args.schedule](), B300, catalog, args.limit
    )
print(json.dumps(result, indent=2))
