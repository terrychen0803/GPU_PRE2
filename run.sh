#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${GPU_PRE2_PYTHON:-}"
if [[ -z "$python_bin" && -f "$project_root/local-runtime.json" ]]; then
    python_bin="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$project_root/local-runtime.json")"
fi
if [[ -z "$python_bin" ]]; then
    if [[ -x "$project_root/.venv/bin/python" ]]; then
        python_bin="$project_root/.venv/bin/python"
    else
        python_bin=python3
    fi
fi
export PYTHONPATH="$project_root/src:$project_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export GPU_PRE2_DATA_ROOT="${GPU_PRE2_DATA_ROOT:-$project_root/../research_data}"
cd -- "$project_root"
exec "$python_bin" "$@"
