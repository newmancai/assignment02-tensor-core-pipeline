# MARPE compact paper source

The 2026-09-10 source follows the single-column CAKE manuscript layout published with arXiv:2608.12629: 11 pt `article`, Letter paper, 1-inch margins, T1 fonts, numeric compressed citations, and a small `unsrtnat` bibliography. The current build is eight Letter pages. It presents the FlashKDA SM80-HMMA-to-SM100 mainline, the H0/H1/T0/T1/T2 challenge, open proposals, program IR, scoped experience IR, execution identity, and the eight-round B300 dual-lane campaign.

The architecture section identifies MARPE's open-source basis and adaptation boundary. It attributes parallel branch search and candidate allocation to PIKE, and isolated GPU execution, profiling, Git episodes, and ABBA validation to Atrex Kernel Agent. The paper claims the HMMA/tcgen05 dual-branch Program IR, scoped Experience IR, matched-opportunity budget protocol, and proof-carrying candidate promotion as its original layer. The experiments are explicitly described as a narrow-width dual-branch closed loop and do not claim a multi-agent advantage over a single agent because no matched single-agent ablation has been run.

Figure 1 is designed to remain readable when captured for the defense presentation. The rebuilt PDF is `../../output/pdf/runtime-profile-evolution-mainline-20260910.pdf`. The scope and final stopping boundary are documented in `../../FRAMEWORK_SCOPE_CORRECTION_20260910.md` and `../../experiments/sm100_open_round1/dual_ir_agent_rounds/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md`.

Build with TeX Live:

```bash
make
```

Or build with Tectonic:

```bash
tectonic main.tex
```

Qualified results, scoped screens, and mechanism probes are labeled separately. The paper claims default-knowledge high-value proposal saturation only under the frozen B300 existing-carrier envelope; it does not claim a global theoretical optimum or completed production H1/T1/T2 verdict.
