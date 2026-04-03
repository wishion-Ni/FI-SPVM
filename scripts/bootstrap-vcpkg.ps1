[CmdletBinding()]
param(
    [string]$VcpkgDir
)

$ErrorActionPreference = "Stop"

if (-not $VcpkgDir) {
    if ($env:VCPKG_ROOT -and (Test-Path (Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake"))) {
        $VcpkgDir = $env:VCPKG_ROOT
    } elseif ($cmd = Get-Command vcpkg -ErrorAction SilentlyContinue) {
        $VcpkgDir = Split-Path -Parent $cmd.Source
    } else {
        $VcpkgDir = Join-Path $PSScriptRoot "..\vcpkg"
    }
}

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

if (-not $env:VCPKG_DOWNLOADS) {
    $downloadsCandidates = @(
        (Join-Path $resolvedVcpkgDir "downloads"),
        (Join-Path $env:LOCALAPPDATA "vcpkg\downloads")
    )
    foreach ($candidate in $downloadsCandidates) {
        if ($candidate -and (Test-Path $candidate)) {
            $env:VCPKG_DOWNLOADS = [System.IO.Path]::GetFullPath($candidate)
            break
        }
    }
}

$env:VCPKG_ROOT = $resolvedVcpkgDir
Write-Host "vcpkg ready: $resolvedVcpkgDir"
if ($env:VCPKG_DOWNLOADS) {
    Write-Host "Using VCPKG_DOWNLOADS: $($env:VCPKG_DOWNLOADS)"
}
