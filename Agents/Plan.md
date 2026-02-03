# JUWELS DDP Address-Family Debug Plan (Low-Compute)

## Goal
Eliminate `Address family not supported by protocol` in multi-node DDP startup on JUWELS, then prove the fix with minimal GPU time.

## Constraints
- Minimize GPU usage and queue cost.
- Do not change model/training logic until network root cause is isolated.
- Use short walltimes (<=10 min), tiny workloads, and staged validation.

## Stages

1. **Baseline capture (no new assumptions)**
   - Gather current failing/partial-fix logs (`repro`, `v2`, `v3`, `v4`).
   - Build a table: script, master addr source, AF warning count, TCPStore failures.
   - Exit criterion: single comparison matrix checked in docs.

2. **Zero/near-zero compute isolation**
   - Validate hostname and address selection paths on JUWELS login node:
     - `getent ahosts`, `getent ahostsv4`, `hostname -I`, IB interface IP lookup.
   - Confirm whether `jwbXXXX`, `jwbXXXX.juwels`, and `jwbXXXXi.juwels` resolve differently.
   - Exit criterion: deterministic master-address selection rule (documented).

3. **Minimal distributed smoke test (2 nodes, no training)**
   - Run a tiny 2-node rendezvous-only script (torch TCPStore/DDP init + barrier + exit).
   - No model load, no dataset, no checkpoints.
   - Test only network/env combinations.
   - Exit criterion: clean startup with 0 AF warnings.

4. **Pinpoint offending resolver path**
   - Compare `--master_addr/--master_port` vs `--rdzv_endpoint` behavior.
   - Compare explicit IPv4 literal vs hostname path.
   - Validate whether any env var or fallback rewrites endpoint to hostname.
   - Exit criterion: one reproducible mechanism that triggers warning.

5. **Apply fix in production slurm launcher**
   - Implement smallest robust fix in a copy of production launcher:
     - stable IPv4 selection,
     - consistent endpoint propagation,
     - no hostname fallback in distributed init.
   - Exit criterion: 2-node 10-min run with warning-free init.

6. **Scale confidence checks (still cost-aware)**
   - 4-node short run.
   - 8-node short run only if 4-node is clean.
   - Exit criterion: no AF warnings / no TCPStore route failures in short checks.

7. **Closeout**
   - Document root cause, final patch, validation evidence, and rollback path.
   - Archive all logs and sacct summaries in `trash/.../ipv4_validation...`.

## GPU Budget Strategy
- Prefer CPU/login diagnostics first.
- Use 2-node short jobs for most experiments.
- Run exactly one variable change per test to keep attribution clean.
- Stop scaling tests immediately on first regression.
