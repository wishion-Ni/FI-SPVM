#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${SCRIPT_DIR}/bootstrap-vcpkg.sh"

cmake --preset dev-linux
cmake --build --preset build-dev-linux
ctest --preset test-dev-linux --output-on-failure

echo "Build and tests completed successfully."
