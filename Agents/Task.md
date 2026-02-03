# Task: Debug and Fix JUWELS DDP Address-Family Error with Minimal Compute

## Problem
Multi-node JUWELS jobs intermittently show:
- `Address family not supported by protocol` during c10d socket setup,
- sometimes followed by rendezvous/TCPStore connectivity failures.

Current state: partially improved scripts reduce severe failures, but AF warnings still appear in 2-node 10-minute validations.

## Objective
Find the exact root cause, implement a robust launcher-level fix, and prove it with the smallest possible compute footprint.

## Deliverables
1. **Root-cause statement** (1-2 paragraphs, specific and test-backed).
2. **Final fixed SLURM launcher** for JUWELS scaling runs.
3. **Validation matrix** (job id -> config -> outcome -> warning/error counts).
4. **Evidence bundle**:
   - `.out/.err` logs,
   - `sacct` summary,
   - concise README section with conclusions.

## Work Items

### A. Repro & Measure
- [ ] Build/update matrix from existing jobs.
- [ ] Count AF warnings, TCPStore failures, No-route errors per run.

### B. Isolate Networking Path (No GPU if possible)
- [ ] Compare DNS results for master host variants (`jwb`, `jwb.juwels`, `jwb*i`).
- [ ] Verify deterministic IPv4 extraction method on JUWELS.

### C. Minimal Distributed Test Harness
- [ ] Add a tiny rendezvous-only script/job (2 nodes, <=5 min).
- [ ] Confirm whether warning occurs without training code.

### D. Fix Iteration (One Change per Run)
- [ ] Endpoint propagation strategy (`master_addr` vs `rdzv_endpoint`).
- [ ] Master address source hardening (strict IPv4 literal).
- [ ] Env consistency across launcher/container ranks.

### E. Confirm & Scale Carefully
- [ ] 2-node 10-min clean run (required gate).
- [ ] 4-node short run.
- [ ] 8-node short run only if 4-node clean.

### F. Finalize
- [ ] Write final “fixed vs not fixed” conclusion.
- [ ] Update README + task artifacts paths.
- [ ] Archive scripts/logs/sacct under `trash/...`.

## Acceptance Criteria
- 2-node validation run has **zero** `Address family not supported` lines.
- No TCPStore route/timeouts in short validation runs.
- Same fixed launcher works for at least 4-node short run.
- Documentation is complete and reproducible.

## Non-Goals
- Training quality/performance tuning.
- Model architecture or optimizer changes.
- Large-scale long-duration training during debug phase.
