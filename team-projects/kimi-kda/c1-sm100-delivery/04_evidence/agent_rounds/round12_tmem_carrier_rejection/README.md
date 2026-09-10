# Round 12: rejected direct TMEM carrier

This directory preserves two correctness failures explored after the accepted
P3-to-P4 shared-carrier result (Job 25377):

- Job 25382 rounded P3's FP32 D fragments, packed two BF16 values per register,
  and stored them with `tcgen05.st.unpack::16b` before using that region as P4's
  TMEM A operand.
- Job 25385 stored each rounded BF16 value in the low half of a separate TMEM
  word with `tcgen05.st.32x32b.x8`.

Both compiled without spills on B300, but both failed all 24,576 output values
for inner counts 1, 2, and 4 with the same mismatch pattern. The accepted shared
carrier and the second-issue diagnostic remained bitwise correct. This
localizes the failure to the physical conversion from the tcgen05 accumulator
load fragment into the dense A-TMEM layout; it is not a barrier, second-commit,
or arithmetic failure.

The result is a rejection, not a proof that a legal direct TMEM conversion does
not exist. A future reopening must supply an explicit CUTLASS/CuTe-derived
D-fragment-to-A layout mapping and pass exact validation before timing. These
files are evidence only and are deliberately excluded from the accepted
reproduction path.
