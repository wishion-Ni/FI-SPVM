[CmdletBinding()]
param(
    [string]$VcpkgDir = (Join-Path $PSScriptRoot "..\vcpkg")
)

$ErrorActionPreference = "Stop"

$resolvedVcpkgDir = [System.IO.Path]::GetFullPath($VcpkgDir)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required but was not found in PATH."
}

if (-not (Test-Path $resolvedVcpkgDir)) {
    Write-Host "Cloning vcpkg to $resolvedVcpkgDir ..."
    git clone https://github.com/microsoft/vcpkg $resolvedVcpkgDir
} else {
    Write-Host "Using existing vcpkg at $resolvedVcpkgDir"
}

$bootstrapBat = Join-Path $resolvedVcpkgDir "bootstrap-vcpkg.bat"
$vcpkgExe = Join-Path $resolvedVcpkgDir "vcpkg.exe"

if (-not (Test-Path $vcpkgExe)) {
    if (-not (Test-Path $bootstrapBat)) {
        throw "bootstrap-vcpkg.bat not found under $resolvedVcpkgDir"
    }
    Write-Host "Bootstrapping vcpkg ..."
    & $bootstrapBat
    if ($LASTEXITCODE -ne 0) {
        throw "vcpkg bootstrap failed with exit code $LASTEXITCODE"
    }
}

Write-Host "vcpkg ready: $resolvedVcpkgDir"
