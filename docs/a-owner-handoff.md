# Assignment 02 handoff

## A status

- M0: complete (0.1 correct/mismatched ARCH experiment, 0.2 roofline, 0.3)
- M1: complete
  - 1.1 fragment map: PASS
  - 1.2 debug: fixed, PASS (before fix: 59/128 mismatches in D rows 8--15)
  - 1.3 FP8 MMA: five seeds, JUDGE PASS
  - 1.4 manual + ldmatrix: both PASS for all three seeds
  - 1.5 B300/Nsight experiment recorded in report
- M2: complete (2.1 written, 2.2/2.3 PASS)
- M6: complete; sm_90a selects WGMMA, while fixed TileLang 0.1.13
  falls back to mma.sync + ldmatrix for the tested sm_100a FP16 T.gemm.

Detailed A report: `作业二报告模板.md`; standalone derivations:
`M0_0.2_峰值与机器平衡点.md` and `M6_TileLang_lowering_对照.md`.

## A -> B: M2 shared-memory layout

- `m2_smem/02_descriptor`: PASS (3/3 scenarios)
- `m2_smem/03_swizzle`: PASS (128B / 64B / 32B)

### SM100 descriptor fields

| Field | Encoding |
|---|---|
| start address | `(saddr >> 4)` in bits `[0, 14)` |
| LBO | `(lbo >> 4)` in bits `[16, 30)` |
| SBO | `(sbo >> 4)` in bits `[32, 46)` |
| version | `1` in bits `[46, 48)` |
| layout type | `layout` in bits `[61, 64)` |

### Scenarios

| Scenario | LBO (B) | SBO (B) | layout |
|---|---:|---:|---:|
| K-major, no swizzle | 128 | 1024 | 0 |
| K-major, 128B swizzle | 0 | 1024 | 2 |
| MN-major, 128B swizzle | 0 | 1024 | 2 |

### 128B swizzle

For an 8-row x 128B atom:

```text
chunk         = colByte >> 4
byteInChunk   = colByte & 0xf
physicalChunk = chunk XOR (row & 7)
offset        = row * 128 + physicalChunk * 16 + byteInChunk
```

For rows outside the first atom, add `(row >> 3) * 1024` and use
`row & 7` as the row inside the atom.

## A -> C: roofline inputs

- Recommended report/roofline inputs (official dense figures):
  - B300 BF16 peak: `2250 TFLOPS`
  - B300 memory bandwidth: `8000 GB/s` (up to)
  - B300 BF16 balance point: `281.25 FLOP/byte`
- Actual allocated `NVIDIA B300 SXM6 AC` device:
  - compute capability `10.3`, `148 SM`, max application clock `2032 MHz`
  - `7680-bit` memory bus, memory clock `3996 MHz`, about `270 GiB`
  - architecture/max-clock BF16 upper bound: `2463.63 TFLOPS`
  - interface-derived bandwidth: `7672.32 GB/s`
  - corresponding balance point: `321.11 FLOP/byte`
- Use the official row for a conventional datasheet roofline. The device row is
  retained to explain why a max-clock derivation does not exactly equal the
  rounded, system-rated datasheet value.
