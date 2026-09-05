# Matched audit: PASS

Source: `outputs/kda-zero-state-20260905/matched_19918.log`

| Case | Scope | None ms | Reused zero ms | Created zero ms | Nonzero ms | Reuse reduction | Create reduction | Create−reuse µs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | eager | 0.148144 | 0.127632 | 0.129792 | 0.127616 | 13.845988 | 12.387943 | 2.160005 |
| 0 | graph | 0.145024 | 0.122496 | 0.124576 | 0.124448 | 15.533981 | 14.099735 | 2.080001 |
| 0 | cache_perturbed | 0.151648 | 0.130048 | 0.134608 | 0.130112 | 14.243507 | 11.236552 | 4.559986 |
| 1 | eager | 0.148208 | 0.127616 | 0.129696 | 0.127616 | 13.893985 | 12.490556 | 2.079993 |
| 1 | graph | 0.145024 | 0.122528 | 0.124576 | 0.122528 | 15.511915 | 14.099735 | 2.048001 |
| 1 | cache_perturbed | 0.152480 | 0.130064 | 0.135120 | 0.130080 | 14.700946 | 11.385105 | 5.055994 |
| 2 | eager | 0.283280 | 0.238368 | 0.242416 | 0.239200 | 15.854277 | 14.425303 | 4.047997 |
| 2 | graph | 0.279952 | 0.237184 | 0.237248 | 0.237184 | 15.276905 | 15.254044 | 0.064000 |
| 2 | cache_perturbed | 0.283632 | 0.238720 | 0.243712 | 0.238736 | 15.834606 | 14.074581 | 4.991993 |
| 3 | eager | 0.283264 | 0.239552 | 0.242400 | 0.239616 | 15.431545 | 14.426122 | 2.847999 |
| 3 | graph | 0.280192 | 0.237152 | 0.237184 | 0.237184 | 15.360893 | 15.349470 | 0.032008 |
| 3 | cache_perturbed | 0.283552 | 0.239104 | 0.243680 | 0.239504 | 15.675432 | 14.061620 | 4.575998 |
| 4 | eager | 0.548320 | 0.459136 | 0.461696 | 0.459232 | 16.264952 | 15.798071 | 2.560005 |
| 4 | graph | 0.544384 | 0.456320 | 0.457280 | 0.456288 | 16.176819 | 16.000469 | 0.960022 |
| 4 | cache_perturbed | 0.549888 | 0.459856 | 0.464304 | 0.459776 | 16.372793 | 15.563899 | 4.448012 |
| 5 | eager | 0.548032 | 0.457616 | 0.462080 | 0.457552 | 16.498304 | 15.683753 | 4.464000 |
| 5 | graph | 0.544416 | 0.454304 | 0.456352 | 0.456320 | 16.552048 | 16.175868 | 2.047986 |
| 5 | cache_perturbed | 0.549936 | 0.459760 | 0.464752 | 0.459744 | 16.397542 | 15.489801 | 4.991993 |
| 6 | eager | 0.553552 | 0.462320 | 0.466096 | 0.462768 | 16.481198 | 15.799058 | 3.775999 |
| 6 | graph | 0.548512 | 0.456352 | 0.459408 | 0.456448 | 16.801818 | 16.244674 | 3.056005 |
| 6 | cache_perturbed | 0.555072 | 0.464848 | 0.468960 | 0.464880 | 16.254470 | 15.513667 | 4.111990 |
| 7 | eager | 0.150176 | 0.127616 | 0.129728 | 0.127648 | 15.022374 | 13.616023 | 2.112001 |
| 7 | graph | 0.145056 | 0.124576 | 0.126496 | 0.124576 | 14.118680 | 12.795054 | 1.920000 |
| 7 | cache_perturbed | 0.152592 | 0.130064 | 0.134080 | 0.130944 | 14.763552 | 12.131699 | 4.015997 |
| 8 | eager | 0.549344 | 0.459424 | 0.463488 | 0.459456 | 16.368617 | 15.628821 | 4.064023 |
| 8 | graph | 0.546352 | 0.456320 | 0.458368 | 0.456352 | 16.478756 | 16.103903 | 2.048016 |
| 8 | cache_perturbed | 0.549392 | 0.459920 | 0.464880 | 0.460752 | 16.285644 | 15.382827 | 4.960001 |
| 9 | eager | 0.548432 | 0.457440 | 0.461456 | 0.457520 | 16.591301 | 15.859030 | 4.016012 |
| 9 | graph | 0.546432 | 0.454304 | 0.456352 | 0.454304 | 16.859921 | 16.485129 | 2.047986 |
| 9 | cache_perturbed | 0.549824 | 0.458912 | 0.462912 | 0.459216 | 16.534746 | 15.807241 | 3.999993 |

Case definitions: `[{"case": 0, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 2048}}, {"case": 1, "shape": {"batch": 1, "fp32": false, "gate": -8.0, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 2048}}, {"case": 2, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 4096}}, {"case": 3, "shape": {"batch": 1, "fp32": false, "gate": -8.0, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 4096}}, {"case": 4, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 8192}}, {"case": 5, "shape": {"batch": 1, "fp32": false, "gate": -8.0, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 8192}}, {"case": 6, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": [8192], "state_mode": "both", "tokens": 8192}}, {"case": 7, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 2049}}, {"case": 8, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "both", "tokens": 8191}}, {"case": 9, "shape": {"batch": 1, "fp32": false, "gate": null, "heads": 12, "lengths": null, "state_mode": "none", "tokens": 8192}}]`

Correctness: `{"correctness": {"nonself_comparisons": 40, "rows": 50, "self_comparisons": 10, "status_counts": {"PASS": 50}}, "nonzero_correctness": {"nonself_comparisons": 10, "rows": 10, "self_comparisons": 0, "status_counts": {"PASS": 10}}, "post_timing_correctness": {"nonself_comparisons": 40, "rows": 50, "self_comparisons": 10, "status_counts": {"PASS": 50}}}`

- Main status audits the matched measurement schema, not promotion or every protocol recommendation.
- GPU Event latency is not CPU allocation wall time. Graph zero_each is captured zero fill plus forward, not Python allocation per replay.
- cache_perturbed zeros 256 MiB before the timed call; this is not cold-K2 proof.
- No per-row dispatch/kernel identity, zero-buffer immutability snapshots, synchronized host wall time, or post-timing nonzero check are logged.
- state_mode is input-case metadata: zero/nonzero arms intentionally have initial state even when metadata says none.
- 50 pre and 50 post zero-semantic checks each contain 10 legacy_none self-checks; nonzero has 10 separate old-V128 comparisons.
- Medians are medians of 3 round medians. p10/p90 and across-shape ranges are not confidence intervals.
