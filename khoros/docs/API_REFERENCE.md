# Programmer API Reference

This is a short map of the most useful public pieces. The source code contains the full details.

## Graph engine — `khoros_cantata.kernel`

- `Workspace` — holds the graph and runs it.
- `Node` — one processing box.
- `Port` — an input or output point on a node.
- `Connection` — connects an output port to an input port.
- `DataObject` — carries one piece of data through the graph.

## Historical workspaces — `khoros_cantata.workspace_runtime`

- `HistoricalWorkspaceParser.parse(path)` — reads a `.wksp` file.
- `NativeOperatorRegistry.register(toolbox, name, function)` — adds a Python operator.
- `NativeOperatorRegistry.resolve(toolbox, name)` — finds an operator.
- `WorkspaceExecutor.restore(workspace)` — changes a historical workspace into a Python graph.
- `WorkspaceExecutor.execute(workspace)` — restores and runs it.

## Numerical compatibility — `khoros_cantata.phase21_numeric`

- `khoros_fft(...)` — selected historical FFT scaling and centering behavior.
- `khoros_sine(...)` — source-derived multidimensional sine generation.
- `conv1d_khoros(...)` — source-derived 1-D convolution behavior.
- `stats(...)` — simple numerical statistics.
- `compare(...)` — compare two arrays with a tolerance.
- `run_phase21_checks()` — run the eleven Phase 21 checks.

## Historical data — `khoros_cantata.viff`

- `read_viff(path)` — read a VIFF file.
- `write_viff(...)` — write supported VIFF structures.

## Historical UI — `khoros_cantata.kforms_compat`

- `parse_uis(text)` — parse UIS text.
- `parse_file(path)` — parse a pane file.

## Command line

```text
python -m khoros_cantata.cli inventory
python -m khoros_cantata.cli check
python -m khoros_cantata.cli inspect <workspace.wksp>
python -m khoros_cantata.cli run <workspace.wksp>
```
