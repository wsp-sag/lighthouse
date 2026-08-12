# lighthouse

This is a light-weight, simplified ABM.

## Usage

This project uses uv as Python manager. To install uv, please visit https://docs.astral.sh/uv/getting-started/installation/

Once uv is installed on the machine, create a new Python environment for lighthouse and install dependencies.

```bash
uv sync --locked
```

If running multi-processor, then need to disable multi-threading (once per session)
```bash
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
```

To run the model with test data, use the following command:

```bash
uv run activitysim run -c model/configs_mp -c model/configs -d model/data -o model/output
```

## Contents

- `model`: ActivitySim inputs (configs, data) for the lighthouse model. Currently `model/data`
  contains test-scale data from CTPS.
- `notebooks`: Demo notebooks to test if the model still works.
- `src/lighthouse`: Python code used to implement this model. This may grow to include extensions to ActivitySim for things we want the lighthouse model to do.


## Development

To point to a local repo, add this to the pyproject.toml file with path to your ActivitySim repo
[tool.uv.sources]
activitysim = { path = "C:/projects/activitysim", editable = true }