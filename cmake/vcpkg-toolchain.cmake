# Prefer an explicitly configured VCPKG_ROOT, then fall back to a repo-local clone.
set(_fi_spvm_env_vcpkg "$ENV{VCPKG_ROOT}")
set(_fi_spvm_repo_vcpkg "${CMAKE_CURRENT_LIST_DIR}/../vcpkg")

if(_fi_spvm_env_vcpkg AND EXISTS "${_fi_spvm_env_vcpkg}/scripts/buildsystems/vcpkg.cmake")
    include("${_fi_spvm_env_vcpkg}/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "${_fi_spvm_repo_vcpkg}/scripts/buildsystems/vcpkg.cmake")
    include("${_fi_spvm_repo_vcpkg}/scripts/buildsystems/vcpkg.cmake")
else()
    message(FATAL_ERROR
        "Unable to locate vcpkg toolchain. "
        "Set VCPKG_ROOT to an existing vcpkg installation or run scripts/bootstrap-vcpkg.")
endif()
