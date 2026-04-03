#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${SCRIPT_DIR}/bootstrap-vcpkg.sh"

cmake --workflow --preset workflow-dev-linux

echo "Build and tests completed successfully."
