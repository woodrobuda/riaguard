# Full historical validation

The complete project validation used a recovered historical corpus containing
Khoros test suites, historical UIS/pane files, workspaces, and golden results.
That corpus is not included in the public GitHub bundle because its
redistribution rights have not been established.

If you have lawful access to the corpus, place it outside the repository and
set the paths expected by the validation scripts.  The Phase 22 report records
the exact historical evidence population used for the project closure.

The public tests in `tests/` are intentionally independent of that private
material.  They verify the core Python package after a clean GitHub checkout.
