# Release audit: PASS_WITH_SKIPS

Source: `outputs/kda-mainline-20260905/release_19901.log`

Main complete: True; correctness: PASS (120/120); state chain: PASS; concurrent correctness: PASS; entry hardening: PASS_WITH_SKIPS.

| Shape | Timing | Old auto ms | Release auto ms | Release 128 ms | Reduction vs old | Reduction vs 128 | Worst paired vs old |
|---|---|---:|---:|---:|---:|---:|---:|
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":2048}` | eager (PASS) | 0.154176 | 0.127552 | 0.207392 | 17.269% | 38.497% | 17.269% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":2048}` | graph (PASS) | 0.149056 | 0.122432 | 0.202304 | 17.862% | 39.481% | 17.862% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":2048}` | cache_perturbed (PASS) | 0.156704 | 0.130192 | 0.210784 | 16.919% | 38.234% | 16.898% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":3072}` | eager (PASS) | 0.223776 | 0.182784 | 0.301600 | 18.318% | 39.395% | 18.303% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":3072}` | graph (PASS) | 0.220736 | 0.181792 | 0.298560 | 17.643% | 39.110% | 17.643% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":3072}` | cache_perturbed (PASS) | 0.225408 | 0.185504 | 0.306240 | 17.703% | 39.425% | 17.464% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | eager (PASS) | 0.294048 | 0.239888 | 0.399808 | 18.419% | 39.999% | 18.419% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | graph (PASS) | 0.290336 | 0.237072 | 0.394816 | 18.346% | 39.954% | 18.340% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | cache_perturbed (PASS) | 0.294112 | 0.239168 | 0.400464 | 18.681% | 40.277% | 18.664% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":6144}` | eager (PASS) | 0.432608 | 0.349712 | 0.592496 | 19.162% | 40.976% | 19.161% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":6144}` | graph (PASS) | 0.429632 | 0.345664 | 0.589376 | 19.544% | 41.351% | 19.544% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":6144}` | cache_perturbed (PASS) | 0.432192 | 0.349200 | 0.593040 | 19.203% | 41.117% | 19.201% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | eager (PASS) | 0.569712 | 0.459184 | 0.784816 | 19.401% | 41.492% | 19.372% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | graph (PASS) | 0.566816 | 0.454304 | 0.780960 | 19.850% | 41.827% | 19.843% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | cache_perturbed (PASS) | 0.571376 | 0.459808 | 0.785504 | 19.526% | 41.463% | 19.518% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":16384}` | eager (PASS) | 1.542944 | 1.542928 | 1.543824 | 0.001% | 0.058% | -0.055% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":16384}` | graph (PASS) | 1.540768 | 1.538720 | 1.540640 | 0.133% | 0.125% | 0.133% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":16384}` | cache_perturbed (PASS) | 1.545216 | 1.545200 | 1.545136 | 0.001% | -0.004% | -0.003% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[8192],"state_mode":"both","tokens":8192}` | eager (PASS) | 0.573472 | 0.463152 | 0.788848 | 19.237% | 41.288% | 19.223% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[8192],"state_mode":"both","tokens":8192}` | graph (PASS) | 0.567872 | 0.456352 | 0.783008 | 19.638% | 41.718% | 19.596% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[8192],"state_mode":"both","tokens":8192}` | cache_perturbed (PASS) | 0.575536 | 0.464880 | 0.790624 | 19.227% | 41.201% | 19.211% |
| `{"batch":2,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | eager (PASS) | 0.618944 | 0.618944 | 0.817744 | 0.000% | 24.311% | -0.003% |
| `{"batch":2,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | graph (PASS) | 0.614960 | 0.615968 | 0.813728 | -0.164% | 24.303% | -0.167% |
| `{"batch":2,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":8192}` | cache_perturbed (PASS) | 0.619536 | 0.619536 | 0.819232 | 0.000% | 24.376% | -0.010% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[1024,1024,1024,1024,1024,1024,1024,1024],"state_mode":"both","tokens":8192}` | eager (PASS) | 0.156064 | 0.154512 | 0.156032 | 0.994% | 0.974% | 0.041% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[1024,1024,1024,1024,1024,1024,1024,1024],"state_mode":"both","tokens":8192}` | graph (PASS) | 0.149152 | 0.149152 | 0.149152 | 0.000% | 0.000% | -0.054% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":[1024,1024,1024,1024,1024,1024,1024,1024],"state_mode":"both","tokens":8192}` | cache_perturbed (PASS) | 0.155712 | 0.155680 | 0.155712 | 0.021% | 0.021% | 0.000% |
| `{"batch":1,"fp32":true,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | eager (PASS) | 0.303472 | 0.303408 | 0.387872 | 0.021% | 21.776% | -0.392% |
| `{"batch":1,"fp32":true,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | graph (PASS) | 0.300608 | 0.299344 | 0.384656 | 0.420% | 22.179% | 0.346% |
| `{"batch":1,"fp32":true,"gate":null,"heads":12,"lengths":null,"state_mode":"both","tokens":4096}` | cache_perturbed (PASS) | 0.302656 | 0.302592 | 0.391088 | 0.021% | 22.628% | -0.005% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"out","tokens":8192}` | eager (PASS) | 0.601840 | 0.547568 | 0.789072 | 9.018% | 30.606% | 8.867% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"out","tokens":8192}` | graph (PASS) | 0.597664 | 0.544416 | 0.785968 | 8.909% | 30.733% | 8.909% |
| `{"batch":1,"fp32":false,"gate":null,"heads":12,"lengths":null,"state_mode":"out","tokens":8192}` | cache_perturbed (PASS) | 0.602320 | 0.549728 | 0.789616 | 8.732% | 30.380% | 8.719% |

## Concurrent two-request pair

Status: PASS; baseline_auto: 0.672208 ms; release_auto: 0.669040 ms.
Comparison (including all paired rounds): `{"latency_reduction_pct":0.4712807579967948,"paired_rounds":[{"latency_reduction_pct":0.50460292335055,"repeat":0},{"latency_reduction_pct":0.33125182160759215,"repeat":1},{"latency_reduction_pct":0.504434834740819,"repeat":2}],"speedup":1.0047351233049717,"status":"PASS","worst_paired_round_reduction_pct":0.33125182160759215,"worst_paired_round_regression_pct":0}`

## Instrumentation and entry skips

- sanitizer/memcheck: exit PASS, sidecar PASS; `{"records":[{"exit_code":0,"kind":"sanitizer_exit","tool":"memcheck"}],"sidecar":{"correctness":{"expected_rows":6,"issues":[],"markers":[{"comparison_rows":6,"kind":"correctness_complete","sanitizer":true}],"observed_rows":6,"status":"PASS","status_counts":{"PASS":6},"tensor_comparisons":12},"error_summary_counts":[0],"issues":[],"status":"PASS"},"status":"PASS"}`
- sanitizer/synccheck: exit PASS, sidecar PASS; `{"records":[{"exit_code":0,"kind":"sanitizer_exit","tool":"synccheck"}],"sidecar":{"correctness":{"expected_rows":6,"issues":[],"markers":[{"comparison_rows":6,"kind":"correctness_complete","sanitizer":true}],"observed_rows":6,"status":"PASS","status_counts":{"PASS":6},"tensor_comparisons":12},"error_summary_counts":[0],"issues":[],"status":"PASS"},"status":"PASS"}`
- profiles/baseline: exit PASS, sidecar PASS; `{"records":[{"exit_code":0,"kind":"profile_exit","variant":"baseline"}],"sidecar":{"issues":[],"markers":[{"kind":"profile_complete","mode":"baseline"}],"status":"PASS"},"status":"PASS"}`
- profiles/release: exit PASS, sidecar PASS; `{"records":[{"exit_code":0,"kind":"profile_exit","variant":"release"}],"sidecar":{"issues":[],"markers":[{"kind":"profile_complete","mode":"release"}],"status":"PASS"},"status":"PASS"}`
- Entry skips: `[{"case":"multi_gpu","status":"SKIP"},{"check":"alias_default","reason":"no --alias-extension supplied","status":"SKIP"},{"check":"multi_gpu","reason":"only one CUDA device visible","status":"SKIP"}]`

## Caveats

- Median of per-round medians; p10/p90 are within-round quantiles, not confidence intervals.
- Positive latency_reduction_pct means faster; paired rounds retain the worst observed regression.
- cache_perturbed zeros 256 MiB before the start event; K1 may still warm K2 workspace, so this is not cold-K2 proof.
- Concurrent pair is a joined two-request/two-stream latency, not per-request latency or serving throughput.
- release_auto is guarded; its name is not proof P4 ran for every shape. Keep dispatch decisions and inspect the guard.

All per-round median/p10/p90/count values and dispatch decisions are retained in JSON output.
