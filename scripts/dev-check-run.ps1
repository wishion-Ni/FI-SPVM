[CmdletBinding()]
param(
    [string]$ConfigPath = "config.json"
)

$ErrorActionPreference = "Stop"

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Unable to locate repository root via git."
}

Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot "build-test.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $resolvedConfig = $ConfigPath
    if (-not (Test-Path $resolvedConfig)) {
        $fixtureConfig = "tests/fixtures/smoke_config.json"
        if (Test-Path $fixtureConfig) {
            Write-Warning "Config '$ConfigPath' not found. Falling back to '$fixtureConfig'."
            $resolvedConfig = $fixtureConfig
        } else {
            throw "Config '$ConfigPath' not found and fixture config is unavailable."
        }
    }

    $cli = Get-ChildItem ".\out\build" -Recurse -Filter "trspv_cli.exe" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $cli) {
        throw "trspv_cli.exe not found under out/build."
    }

    & $cli --conf $resolvedConfig
    if ($LASTEXITCODE -ne 0) {
        $fixtureConfig = "tests/fixtures/smoke_config.json"
        $canFallbackAfterRunFail = (
            $resolvedConfig -eq "config.json" -and
            (Test-Path $fixtureConfig)
        )
        if ($canFallbackAfterRunFail) {
            Write-Warning "Run with '$resolvedConfig' failed. Retrying with '$fixtureConfig'."
            & $cli --conf $fixtureConfig
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "CLI run failed with exit code $LASTEXITCODE using config '$resolvedConfig'."
    }
}
finally {
    Pop-Location
}
