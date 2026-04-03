#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VCPKG_DIR="${1:-${REPO_ROOT}/vcpkg}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found in PATH." >&2
  exit 1
fi

if [ ! -d "${VCPKG_DIR}" ]; then
  echo "Cloning vcpkg to ${VCPKG_DIR} ..."
  git clone https://github.com/microsoft/vcpkg "${VCPKG_DIR}"
else
  echo "Using existing vcpkg at ${VCPKG_DIR}"
fi

if [ ! -x "${VCPKG_DIR}/vcpkg" ]; then
  if [ ! -f "${VCPKG_DIR}/bootstrap-vcpkg.sh" ]; then
    echo "bootstrap-vcpkg.sh not found under ${VCPKG_DIR}" >&2
    exit 1
  fi
  echo "Bootstrapping vcpkg ..."
  "${VCPKG_DIR}/bootstrap-vcpkg.sh"
fi

echo "vcpkg ready: ${VCPKG_DIR}"
