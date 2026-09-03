# Job 17965 NCU evidence manifest

This directory publishes the integrity manifest for the two targeted NCU
captures used by the C1 report:

- `05_official_v128_h12_t8192_17965.ncu-rep`
- `05_valueslice_v16_h12_t8192_17965.ncu-rep`

The binary reports are retained in the local experiment archive but are
intentionally excluded from the public repository by the project-level
`*.ncu-rep` ignore rule.  NCU report containers embed machine, account, driver,
and GPU metadata that is not needed to reproduce the numerical claims.

Public, inspectable evidence is available in [`../../data/raw/`](../../data/raw/):

- `05_targeted_ncu_17965.log`
- `05_targeted_ncu_metrics_17965.csv`
- `05_targeted_ncu_summary_17965.csv`

The SHA-256 values in [`SHA256SUMS`](SHA256SUMS) were checked against both the
local copies and the files on B300; they matched byte-for-byte.
