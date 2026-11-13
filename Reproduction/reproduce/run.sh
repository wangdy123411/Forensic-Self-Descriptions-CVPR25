#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FSD_PYTHON:-python}" -u "$here/../run.py" reproduce "$@"
