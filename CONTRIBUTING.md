# Contributing

Thanks for your interest in this project.

## Reporting issues

If you run into a bug or unexpected result, please open an issue with:
- What you ran (command / notebook cell) and your environment (OS, GPU, package versions)
- What you expected vs. what happened
- Any relevant error output or log excerpt

## Proposing changes

1. Fork the repository and create a branch for your change.
2. Keep changes focused — one topic per pull request.
3. Match the existing code style (no framework-specific formatter is enforced,
   but keep functions small and avoid adding dependencies unless necessary).
4. If you change gate logic, calibration, or execution scoring, describe the
   behavioral difference in your PR description — these are the parts most
   sensitive to subtle regressions.
5. Run the standard baseline (`python run_experiment.py --standard-only`) on a
   small sample to confirm nothing broke before opening the PR, if you have
   GPU access.

## Scope

This repository implements one method (uncertainty-gated iterative
refinement for text-to-SQL) evaluated on BIRD-SQL with two open-weight
models. Contributions that extend it to new models, datasets, or gate
signals are welcome; contributions that change the core calibration/gate
semantics should include a clear rationale, since those numbers are reported
in an accompanying paper.

## Questions

Open an issue for anything not covered here.
