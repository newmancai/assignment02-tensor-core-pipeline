"""Assignment 02 M6: compare TileLang lowering for sm_90a and sm_100a.

This is compile/lowering only; no H100 or B300 kernel execution is required.
Generated TIR and CUDA files are written under the adjacent ``generated``
directory.
"""
from pathlib import Path

import tilelang
import tilelang.language as T
import tvm


M = N = K = 1024
BM, BN, BK = 128, 128, 64
THREADS, STAGES = 128, 3


def make_gemm():
    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((K, N), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM),
                      threads=THREADS) as (bx, by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            B_shared = T.alloc_shared((BK, BN), "float16")
            C_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BM, bx * BN])

    return main


def interesting_lines(source: str) -> list[str]:
    needles = (
        "wgmma", "tcgen", "mma", "tma", "cp.async", "cp_async",
        "descriptor", "desc", "swizzle", "mbarrier", "cuda::memcpy_async",
    )
    return [line.strip() for line in source.splitlines()
            if any(n in line.lower() for n in needles)]


def lower_one(label: str, target_spec: dict, out_dir: Path) -> None:
    target = tvm.target.Target(target_spec)
    # Some lowering passes query Target.current(); keep the explicit target
    # active as a context as well as passing it to tilelang.lower.
    with target:
        artifact = tilelang.lower(make_gemm(), target=target)
    (out_dir / f"{label}_lowered_tir.txt").write_text(
        artifact.device_mod.script(), encoding="utf-8")
    (out_dir / f"{label}_generated.cu").write_text(
        artifact.kernel_source, encoding="utf-8")
    lines = interesting_lines(artifact.kernel_source)
    (out_dir / f"{label}_key_lines.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{label}] target={target} "
          f"CUDA lines={len(artifact.kernel_source.splitlines())}, "
          f"key lines={len(lines)}")
    for line in lines[:24]:
        print("  " + line)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "generated"
    out_dir.mkdir(exist_ok=True)
    targets = (
        ("sm_90a", {"kind": "cuda", "arch": "sm_90a"}),
        ("sm_100a_literal", {"kind": "cuda", "arch": "sm_100a"}),
        # TileLang 0.1.13 documents family arch + architecture-specific code
        # as the preferred Blackwell target spelling.
        ("sm_100a", {"kind": "cuda", "arch": "sm_100f",
                      "code": ["sm_100a"]}),
    )
    for label, target_spec in targets:
        lower_one(label, target_spec, out_dir)


if __name__ == "__main__":
    main()
