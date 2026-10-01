param(
    [string]$AppEnv   = 'uat',
    [switch]$SkipBuild,         # Pass -SkipBuild to skip frontend rebuild (Python-only changes)
    [switch]$Infra              # Pass -Infra to (re)deploy bundle resources via terraform
                                 # (app ACLs, schema/volume, jobs). Requires MANAGE on the
                                 # target catalog for the first deploy of a target. Only
                                 # needed when resources/permissions change.
)

$ErrorActionPreference = 'Stop'

# Deploy orchestration metadata only (bundle target / app name / CLI profile).
# App-facing env var overrides (LAKEBASE_DATABASE, TRANSLATE_VOLUME_PATH,
# SOFFICE_ARCHIVE_VOLUME_PATH) live in utils/deploy/target_env.json, rendered
# by render_target_config_env.py below. Don't hardcode them here too --
# that's exactly how a stale test database name could leak into a real-uat
# deploy.
$Targets = @{
    uat        = @{ Target = 'latlang-uat';      AppName = 'latlang';           Profile = 'UAT' }
    prod       = @{ Target = 'latlang-prod';     AppName = 'latlang';           Profile = 'qualibot-prod' }
    'uat-test' = @{ Target = 'latlang-uat-test'; AppName = 'latlang-uat-test';  Profile = 'UAT' }
}

if (-not $Targets.ContainsKey($AppEnv)) {
    Write-Host "Unknown environment '$AppEnv'. Use: uat | prod | uat-test" -ForegroundColor Red
    exit 1
}

$Target  = $Targets[$AppEnv].Target
$AppName = $Targets[$AppEnv].AppName
$Profile = $Targets[$AppEnv].Profile

Write-Host ""
Write-Host "=== LatLang deploy  env=$AppEnv  target=$Target  profile=$Profile ===" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $ProjectRoot

try {
    # 1. Build frontend (skip with -SkipBuild for Python-only changes)
    if ($SkipBuild) {
        Write-Host "[1/4] Frontend build skipped (-SkipBuild)" -ForegroundColor DarkGray
    } else {
        Write-Host "[1/4] Building frontend..." -ForegroundColor Yellow
        Push-Location client
        try {
            & bun install
            if ($LASTEXITCODE -ne 0) { throw "bun install failed (exit $LASTEXITCODE)" }
            & bun run build
            if ($LASTEXITCODE -ne 0) { throw "bun run build failed (exit $LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }

    # 2. Generate per-env config (from target_env.json) and push source files
    # to the workspace. target_config.env overrides app.yaml's defaults; it is
    # gitignored and uploaded separately -- not committed.
    Write-Host "[2/4] Writing target_config.env and uploading source files..." -ForegroundColor Yellow
    & python utils/deploy/render_target_config_env.py $AppEnv target_config.env
    if ($LASTEXITCODE -ne 0) { throw "render_target_config_env.py failed (exit $LASTEXITCODE)" }

    $SourceCodePath = "/Workspace/Shared/.bundle/latlang/$Target/files"

    if ($Infra) {
        # Full bundle deploy: (re)applies app ACLs + schema/volume + jobs via
        # terraform. First deploy of a target REQUIRES the deploying user to
        # have MANAGE on the target catalog. Reserve for the first deploy or
        # when resources/permissions actually change.
        Write-Host "      -Infra: deploying bundle resources via terraform (needs catalog MANAGE)..." -ForegroundColor Yellow
        & databricks bundle deploy --target $Target --profile $Profile
        if ($LASTEXITCODE -ne 0) { throw "databricks bundle deploy failed (exit $LASTEXITCODE)" }
    } else {
        # Routine code deploy: just upload the source files. No terraform, so
        # it never touches catalog/volume permissions and works without admin
        # rights. --include force-uploads target_config.env and client/out
        # (both gitignored, both required at runtime).
        & databricks sync . $SourceCodePath --profile $Profile `
            --include 'target_config.env' `
            --include 'client/out/**' `
            --exclude 'client/src/**' `
            --exclude 'client/public/**' `
            --exclude 'client/node_modules/**' `
            --exclude 'Translator/**' `
            --exclude 'lakebase_backups/**' `
            --exclude '**/*.ipynb' `
            --exclude '.databricks/**'
        if ($LASTEXITCODE -ne 0) { throw "databricks sync failed (exit $LASTEXITCODE)" }
    }

    # 3. Ensure the app compute is running before deploying. Deploying to a
    # stopped app fails outright ("Cannot deploy app ... as it is not in
    # RUNNING state"). latlang-uat-test must NEVER be auto-started by a
    # script (disposable test app -- being stopped is its normal/intended
    # state outside of an active manual test session). uat and prod are
    # fine to auto-start here.
    Write-Host "[3/4] Checking app compute state..." -ForegroundColor Yellow
    $appInfo = & databricks apps get $AppName --profile $Profile -o json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "databricks apps get failed (exit $LASTEXITCODE)" }
    $computeState = $appInfo.compute_status.state
    if ($computeState -in @('ACTIVE', 'STARTING')) {
        Write-Host "      $AppName compute is $computeState - no action needed." -ForegroundColor DarkGray
    } elseif ($AppEnv -eq 'uat-test') {
        throw "$AppName compute is $computeState. This app is never auto-started by a script - start it manually first: databricks apps start $AppName --profile $Profile"
    } else {
        Write-Host "      $AppName compute is $computeState - starting it (waits until active)..." -ForegroundColor Yellow
        & databricks apps start $AppName --profile $Profile
        if ($LASTEXITCODE -ne 0) { throw "databricks apps start failed (exit $LASTEXITCODE)" }
    }

    # Create a new app deployment so the app restarts and picks up the newly synced files.
    Write-Host "[4/4] Deploying app (waiting for restart to complete)..." -ForegroundColor Yellow
    & databricks apps deploy $AppName --source-code-path $SourceCodePath --auto-approve --profile $Profile
    if ($LASTEXITCODE -ne 0) { throw "databricks apps deploy failed (exit $LASTEXITCODE)" }

    Write-Host ""
    Write-Host "LatLang ($AppEnv / app=$AppName) deployed and running. See target_config.env for the env overrides applied." -ForegroundColor Green

} finally {
    Pop-Location
}
