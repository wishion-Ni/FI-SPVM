#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ "${1:-}" != "" ]; then
  VCPKG_DIR="$1"
elif [ "${VCPKG_ROOT:-}" != "" ] && [ -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]; then
  VCPKG_DIR="${VCPKG_ROOT}"
elif command -v vcpkg >/dev/null 2>&1; then
  VCPKG_DIR="$(cd "$(dirname "$(command -v vcpkg)")" && pwd)"
else
  VCPKG_DIR="${REPO_ROOT}/vcpkg"
fi

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

if [ -z "${VCPKG_DOWNLOADS:-}" ]; then
  if [ -d "${VCPKG_DIR}/downloads" ]; then
    export VCPKG_DOWNLOADS="${VCPKG_DIR}/downloads"
  elif [ -d "${HOME}/.cache/vcpkg/downloads" ]; then
    export VCPKG_DOWNLOADS="${HOME}/.cache/vcpkg/downloads"
  fi
fi

export VCPKG_ROOT="${VCPKG_DIR}"
echo "vcpkg ready: ${VCPKG_DIR}"
if [ "${VCPKG_DOWNLOADS:-}" != "" ]; then
  echo "Using VCPKG_DOWNLOADS: ${VCPKG_DOWNLOADS}"
fi
