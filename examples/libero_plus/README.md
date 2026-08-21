# LIBERO-Plus evaluation client

`main.py` is the simulator-side evaluator for LIBERO-Plus. It follows the same
WebSocket client/server design as OpenPI's `examples/libero/main.py`: the
SHIFTVLA environment runs the policy server, while this separate Python 3.8
environment runs MuJoCo and sends observations to that server.

The two environments intentionally remain separate:

- `SHIFTVLA/.venv` (Python 3.11): training and policy server.
- `SHIFTVLA/openpi/examples/libero_plus/.venv` (Python 3.8): LIBERO-Plus simulator client.

## Prerequisites

Make sure the LIBERO-Plus repository is downloaded at:

```text
SHIFTVLA/LIBERO-plus
```

The checkout alone is not enough. Download the LIBERO-Plus assets described in
`LIBERO-plus/README.md` and extract `assets.zip` so that this directory exists:

```text
SHIFTVLA/LIBERO-plus/libero/libero/assets
```

The expected checkout also contains:

```text
SHIFTVLA/LIBERO-plus/libero/libero/benchmark/task_classification.json
```

LIBERO-Plus lists several system libraries that cannot be installed by `uv`.
Install them once with administrator privileges (skip packages that are already
present on the machine):

```bash
sudo apt-get update
sudo apt-get install -y \
  libexpat1 \
  libfontconfig1-dev \
  libpython3-stdlib \
  libmagickwand-dev \
  cmake build-essential
```

`libmagickwand-dev` is required by the Python `wand` package. A uv virtual
environment cannot replace these operating-system shared libraries.

## Create the Python 3.8 client environment

Run all commands in this section from the vendored OpenPI directory:

```bash
cd /home/dataset-assist-1/VLA/SHIFTVLA/openpi
export SHIFTVLA_ROOT="$(cd .. && pwd)"

uv venv --python 3.8 examples/libero_plus/.venv
source examples/libero_plus/.venv/bin/activate

uv pip sync examples/libero_plus/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install -r "$SHIFTVLA_ROOT/LIBERO-plus/requirements.txt"
uv pip install -r "$SHIFTVLA_ROOT/LIBERO-plus/extra_requirements.txt"
uv pip install -e packages/openpi-client
uv pip install -e "$SHIFTVLA_ROOT/LIBERO-plus"
```

Use `uv pip`, not plain `pip`, for all Python package operations in this
environment.

`examples/libero_plus/requirements.txt` intentionally stays identical to
OpenPI's standard LIBERO client requirements. LIBERO-Plus keeps its own runtime
dependencies in `LIBERO-plus/requirements.txt` and its two additional packages
(`wand` and `scikit-image`) in `LIBERO-plus/extra_requirements.txt`.

Both projects pin `robosuite==1.4.1`, so installing the LIBERO-Plus dependencies
on top of the base environment does not create a version conflict. Use
`uv pip sync` only for the compiled OpenPI lock file; the two LIBERO-Plus files
are ordinary input files, so install them with `uv pip install -r` to include
their transitive dependencies. The editable installs remain separate because
LIBERO-Plus's `setup.py` has an empty `install_requires`.

When OpenPI's LIBERO client dependencies change, update the standard LIBERO
input and regenerate its lock file, then copy both files to this directory:

```bash
uv pip compile examples/libero/requirements.in \
  --output-file examples/libero/requirements.txt \
  --python-version 3.8 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

cp examples/libero/requirements.in examples/libero_plus/requirements.in
cp examples/libero/requirements.txt examples/libero_plus/requirements.txt
```

Then rerun the installation commands above. Do not hand-edit either generated
`requirements.txt`.

## Create the LIBERO path configuration

Keep the LIBERO-Plus path file with this example rather than writing it to the
default `~/.libero_plus` directory:

```bash
cd /home/dataset-assist-1/VLA/SHIFTVLA/openpi
source examples/libero_plus/.venv/bin/activate

export SHIFTVLA_ROOT="$(cd .. && pwd)"
export LIBERO_CONFIG_PATH="$PWD/examples/libero_plus/.libero_plus"
export PYTHONPATH="$SHIFTVLA_ROOT/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"

# The first import asks whether to choose a custom dataset directory. "n"
# creates config.yaml using the editable LIBERO-Plus checkout paths.
printf 'n\n' | python -c 'import libero.libero'
```

The resulting file must be:

```text
SHIFTVLA/openpi/examples/libero_plus/.libero_plus/config.yaml
```

An editable LIBERO-Plus install is normally sufficient for imports. The
explicit `PYTHONPATH` above is retained for parity with OpenPI's LIBERO example
and guarantees that this checkout wins if another package named `libero` is
installed elsewhere.

Verify the setup before starting a long evaluation:

```bash
test -d "$SHIFTVLA_ROOT/LIBERO-plus/libero/libero/assets"
test -f "$SHIFTVLA_ROOT/LIBERO-plus/libero/libero/benchmark/task_classification.json"
test -f "$LIBERO_CONFIG_PATH/config.yaml"

MUJOCO_GL=egl python -c \
  'from libero.libero import benchmark; print(sorted(benchmark.get_benchmark_dict()))'
python examples/libero_plus/main.py --help
```

If EGL initialization fails, test with `MUJOCO_GL=glx`, although EGL is the
normal choice for a headless GPU server.

## Run evaluation

The recommended entrypoint is the repository-level wrapper. It selects this
Python 3.8 environment directly, so the client venv does not need to be
activated in this terminal.

Terminal 1 — start the policy server with the SHIFTVLA Python 3.11 environment:

```bash
cd /home/dataset-assist-1/VLA/SHIFTVLA
bash scripts_sh/serve_libero_plus_policy.sh \
  --checkpoint /home/dataset-assist-1/VLA/models/pi05_libero_base \
  --config pi05_libero \
  --gpu 0 \
  --port 9000
```

Terminal 2 — run the LIBERO-Plus simulator client:

```bash
cd /home/dataset-assist-1/VLA/SHIFTVLA
bash scripts_sh/eval_libero_plus_client.sh \
  --host 127.0.0.1 \
  --port 9000 \
  --gpu 1
```

For a full-model/base checkpoint, the default automatic profile evaluates all
seven official perturbation categories. For an adapter checkpoint, it defaults
to the five training categories. Both profiles run all four suites by default.
Selections are evaluated sequentially, one perturbation and one suite per
`main.py` process.

Examples:

```bash
# Camera tasks from only the Spatial suite.
bash scripts_sh/eval_libero_plus_client.sh \
  --perturbations camera \
  --suites spatial \
  --no-video

# Five selected perturbations over all four suites.
bash scripts_sh/eval_libero_plus_client.sh \
  --perturbations background,camera,light,noise,language \
  --suites all

# All seven official named categories over all four suites.
bash scripts_sh/eval_libero_plus_client.sh \
  --perturbations official7 \
  --suites all
```

Use `bash scripts_sh/eval_libero_plus_client.sh --help` for every option. The
wrapper checks the policy server, creates separate output directories/logs,
rejects rollout exceptions, and verifies the expected number of completed
episodes.

## Direct `main.py` invocation

For debugging a single category/suite without the wrapper:

```bash
cd /home/dataset-assist-1/VLA/SHIFTVLA/openpi
source examples/libero_plus/.venv/bin/activate
export SHIFTVLA_ROOT="$(cd .. && pwd)"
export LIBERO_CONFIG_PATH="$PWD/examples/libero_plus/.libero_plus"
export PYTHONPATH="$PWD/packages/openpi-client/src:$SHIFTVLA_ROOT/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
python examples/libero_plus/main.py \
  --args.host 127.0.0.1 \
  --args.port 9000 \
  --args.task-suite-name libero_spatial \
  --args.perturbation "Camera Viewpoints" \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero_plus/camera_spatial
```

One direct `main.py` process accepts one perturbation category and one suite.
The shell wrapper provides multi-perturbation and multi-suite evaluation by
running those combinations sequentially against the same policy server.

## Data conversion

The mixed-RLDS LIBERO-Plus converter is:

```text
openpi/examples/libero_plus/convert_libero_plus_rlds_by_perturbation.py
```

`convert_libero_rlds_to_lerobot.py` in this directory is retained only as an
old standard-LIBERO reference and is not the LIBERO-Plus mixed-data converter.
