# Code Documentation Index

Every main Python module in the release has a plain-language module docstring. The code is written so a reader can follow it from the outside in.

| File | What to read first |
|---|---|
| `kernel.py` | `Workspace`, `Node`, `Port`, `Connection`, `DataObject` |
| `workspace_runtime.py` | `HistoricalWorkspaceParser`, `NativeOperatorRegistry`, `WorkspaceExecutor` |
| `phase21_numeric.py` | `khoros_fft`, `conv1d_khoros`, `khoros_sine`, `run_phase21_checks` |
| `viff.py` | `read_viff` |
| `kdbm_compat.py` | `KDBM` |
| `kdms_object.py` | logical data object and physical storage classes |
| `cms.py` | CMS/toolbox record readers |
| `kforms_compat.py` | `parse_uis`, `parse_file` |
| `operator_spec.py` | `OperatorSpec` |
| `property_panel.py` | `PropertyPanelModel` |
| `cli.py` | `main` and the four commands |

The programmer guide explains how these pieces fit together. The user manual explains how to use the release without needing to understand the internals.
