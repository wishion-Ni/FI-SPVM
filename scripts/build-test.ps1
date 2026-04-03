[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot "bootstrap-vcpkg.ps1")

    $isWindowsHost = ($env:OS -eq "Windows_NT")
    if (-not $isWindowsHost) {
        throw "This script is for Windows. Use scripts/build-test.sh on Linux."
    }

    $candidates = @(
        @{ Configure = "dev-windows"; Workflow = "workflow-dev-windows" },
        @{ Configure = "dev-windows-vs2022"; Workflow = "workflow-dev-windows-vs2022" },
        @{ Configure = "dev-windows-vs2019"; Workflow = "workflow-dev-windows-vs2019" },
        @{ Configure = "dev-windows-ninja"; Workflow = "workflow-dev-windows-ninja" }
    )

    $selected = $null
    foreach ($candidate in $candidates) {
        Write-Host "Trying configure preset: $($candidate.Configure)"
        & cmake --preset $candidate.Configure
        if ($LASTEXITCODE -eq 0) {
            $selected = $candidate
            break
        }
        Write-Warning "Preset '$($candidate.Configure)' failed. Trying next fallback."
    }

    if ($null -eq $selected) {
        throw "All Windows configure presets failed."
    }

    Write-Host "Using workflow preset: $($selected.Workflow)"
    & cmake --workflow --preset $selected.Workflow
    if ($LASTEXITCODE -ne 0) {
        throw "Workflow failed with preset '$($selected.Workflow)'."
    }

    Write-Host "Build and tests completed successfully."
}
finally {
    Pop-Location
}
