# Khoros/Cantata Python Conversion

This project is a Python reimplementation of the parts of the historical Khoros/Cantata system that could be reconstructed from the recovered source code and test material.

## What you can do

- Read historical Cantata workspace files.
- Inspect the boxes and connections in a workspace.
- Run supported workspaces with Python-native operators.
- Read historical Khoros VIFF data.
- Parse historical Khoros pane/UI files.
- Run the recovered numerical regression checks.

## What you cannot assume

The original Khoros executable is not included. A Python-native operator is not automatically proof that it produces exactly the same result for every historical input. The final operator matrix records the level of evidence for each operator.

## Quick start

From the release directory:

```text
python -m khoros_cantata.cli inventory
python -m khoros_cantata.cli check
python -m khoros_cantata.cli inspect historical_workspaces/datamanip/examples/workspaces/kclip/clipout.megaclip.wksp
```

For a friendly explanation, read the User Manual. For developers, read the Programmer Guide.
