# Weekly Refresh Orchestrator
# =============================================================================
# Runs the full unattended weekly cycle: scrape (all 3 crops) -> validate ->
# merge -> Script 09 (rebuild panels) -> sync into repo -> Script 23 (retrain)
# -> Script 44 (final sanity check, logged) -> local git commit.
#
# Safety gates:
#   - A crop's scrape/validate/merge failure only skips THAT crop -- others
#     still proceed, and the trusted file for a failed crop is left untouched.
#   - If zero crops merge successfully, the job stops before touching the
#     pipeline at all (Script 09/23 never run on unchanged data).
#   - If Script 09 or Script 23 itself fails (nonzero exit), the job stops
#     immediately -- no git commit happens on a broken run.
#   - Never auto-pushes to GitHub. Commits locally only.
#
# Register with Task Scheduler (run once, as the user, elevated not required):
#   schtasks /Create /SC WEEKLY /D TUE /ST 03:00 /TN "TOP_Digital_Twin_Weekly_Refresh" ^
#     /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\masro\Documents\TOP_Digital_Twin\scripts\weekly_refresh\run_weekly_refresh.ps1\"" ^
#     /RL LIMITED
#
# Or see register_task.ps1 in this folder for a scripted version of the same.
# =============================================================================

$RepoRoot = "C:\Users\masro\Documents\TOP_Digital_Twin"
$ScraperScript = "C:\Users\masro\Documents\Codex\2026-05-14\assuming-you-re-an-expert-provide\agmarknet_onion_prices.py"
$Downloads = "C:\Users\masro\Downloads"
$Staging = Join-Path $RepoRoot "scripts\weekly_refresh\staging"
$LogDir = Join-Path $RepoRoot "logs\weekly_refresh"
$Python = "python"

New-Item -ItemType Directory -Force -Path $Staging | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ts = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = Join-Path $LogDir "refresh_$ts.log"

# Opened ONCE, shared for read so the log stays tailable by another process
# while this script runs (Add-Content-per-line was found to hold the file
# exclusively locked for the run's full duration -- made live monitoring
# impossible during the 2026-08-27 incident below).
$LogStream = [System.IO.StreamWriter]::new(
    [System.IO.File]::Open($LogFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
)
$LogStream.AutoFlush = $true

function Log {
    param([string]$msg)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    $LogStream.WriteLine($line)
}

function RunLogged {
    # Runs a command, streams its output into the log, returns the exit code.
    # NOTE: the parameter is deliberately NOT named $Args -- that collides
    # with PowerShell's reserved automatic $args variable and silently
    # splats an EMPTY argument list to the child process (confirmed live,
    # 2026-08-27: this caused `python` to launch with zero arguments and
    # drop into an interactive REPL, which then hung for ~46 hours issuing
    # console errors in a non-interactive redirected context -- Task
    # Scheduler's own ExecutionTimeLimit, see register_task.ps1, is the
    # backstop for any future hang of this general shape, but this specific
    # cause is also now guarded directly below).
    param([string]$Exe, [string[]]$CmdArgs)
    if (-not $CmdArgs -or $CmdArgs.Count -eq 0) {
        Log "INTERNAL ERROR: RunLogged called with an empty argument list for $Exe -- refusing to run it (would launch interactively/REPL and hang). This means a caller passed a malformed argument array."
        return 1
    }
    Log ("RUN: {0} {1}" -f $Exe, ($CmdArgs -join ' '))
    # Write-Host, NOT Write-Output, for the per-line echo below: Write-Output
    # would add every line to THIS function's own output stream, and since
    # the caller captures `$code = RunLogged ...`, that turns $code into an
    # array of [every log line..., exit code] instead of a clean scalar --
    # `if ($code -ne 0)` on that array is then true almost regardless of the
    # real exit code (elementwise -ne against a bunch of non-empty strings).
    # Confirmed live, 2026-08-29: a genuinely successful 146,842-row tomato
    # scrape got reported as FAILED and silently skipped because of exactly
    # this. Write-Host bypasses the function's output stream entirely, so
    # $LASTEXITCODE alone reaches the caller via `return`.
    & $Exe @CmdArgs 2>&1 | ForEach-Object { $LogStream.WriteLine($_); Write-Host $_ }
    return $LASTEXITCODE
}

function FailAndExit {
    param([string]$reason)
    Log "=== WEEKLY REFRESH FAILED: $reason ==="
    $LogStream.Flush(); $LogStream.Close()
    $failedName = Join-Path $LogDir ("refresh_{0}_FAILED.log" -f $ts)
    Rename-Item -Path $LogFile -NewName (Split-Path $failedName -Leaf) -ErrorAction SilentlyContinue
    exit 1
}

Log "=== Weekly Refresh Started ==="
$year = (Get-Date).Year
$crops = @("Tomato", "Onion", "Potato")
$mergedCrops = @()

# Potato's production panel has only ever included West Bengal + Uttarakhand
# (the other-state balanced-panel/price-clip filters in Script 09 drop
# everything else regardless). A live all-India potato scrape confirmed
# 2026-08-29 that most of what gets filtered out isn't even comparable data:
# Tamil Nadu's "potato markets" are the Uzhavar Sandhai retail farmer-market
# scheme (median 0.22 tonnes/report vs West Bengal's 12 -- retail lots, not
# wholesale mandi trade), and Kerala is a genuine but thin, import-dependent
# consumption market. Restricting the scrape itself to the two states that
# actually survive into the panel avoids re-scraping/re-validating data that
# never reaches the model, at the cost of the trusted raw CSV no longer
# retaining all-India potato history going forward (it never fed anything
# downstream anyway). Tomato/onion stay all-India -- their panels use many
# more states.
$StateFilter = @{ Potato = "West Bengal,Uttarakhand" }

foreach ($crop in $crops) {
    $cropLower = $crop.ToLower()
    $trustedFile = Join-Path $Downloads "${cropLower}_all_india_apmcs_2000_2026.csv"
    $scrapeFile = Join-Path $Staging "${cropLower}_weekly_scrape.csv"

    $scrapeArgs = @($ScraperScript, "--commodity", $crop, "--start-year", $year, "--end-year", $year, "--out", $scrapeFile, "--log-level", "WARNING")
    $scopeLabel = "all states"
    if ($StateFilter.ContainsKey($crop)) {
        $scrapeArgs += @("--states", $StateFilter[$crop])
        $scopeLabel = $StateFilter[$crop]
    }

    Log "--- $crop : scraping (year $year, $scopeLabel) ---"
    $code = RunLogged $Python $scrapeArgs
    if ($code -ne 0) {
        Log "$crop : SCRAPE FAILED (exit $code) -- skipping this crop, trusted file untouched"
        continue
    }

    Log "--- $crop : validating ---"
    $code = RunLogged $Python @((Join-Path $RepoRoot "scripts\weekly_refresh\validate_scrape.py"), $cropLower, $trustedFile, $scrapeFile)
    if ($code -ne 0) {
        Log "$crop : VALIDATION FAILED -- skipping this crop, trusted file untouched"
        continue
    }

    Log "--- $crop : merging ---"
    $code = RunLogged $Python @((Join-Path $RepoRoot "scripts\weekly_refresh\merge_scrape.py"), $cropLower, $trustedFile, $scrapeFile)
    if ($code -ne 0) {
        Log "$crop : MERGE FAILED -- check for a .backup_* file next to the trusted CSV, it may need manual restoration"
        continue
    }

    $mergedCrops += $crop
    Log "$crop : refresh complete"
}

if ($mergedCrops.Count -eq 0) {
    FailAndExit "no crops merged successfully, stopping before touching the pipeline"
}
Log ("Crops merged this run: {0}" -f ($mergedCrops -join ', '))

Push-Location $RepoRoot

Log "--- Script 09: rebuilding weekly panels ---"
$code = RunLogged $Python @("scripts\09_Agmarknet_Weekly_Panel.py")
if ($code -ne 0) { Pop-Location; FailAndExit "Script 09 failed (exit $code)" }

Log "--- Syncing panels into repo (data/agmarknet_weekly/) ---"
# FIXED 2026-09-02 (audit finding, confirmed): this used to be four bare
# Copy-Item calls with no existence check or error handling -- a missing
# source file (e.g. Script 09 silently not writing one of the four) would
# print a red error to the console and then fall straight through to
# Script 23 training on a stale or partial panel copy, with no FailAndExit
# guard like every other step in this script already has. Now verifies
# each source file exists first, and stops the run (matching the rest of
# the script's own established safety pattern) if any sync fails.
$syncFiles = @("tomato_weekly_panel.csv", "onion_weekly_panel.csv", "potato_weekly_panel.csv", "top_weekly_panel.csv")
foreach ($f in $syncFiles) {
    $src = Join-Path $Downloads "Agmarknet_Weekly\$f"
    if (-not (Test-Path $src)) {
        Pop-Location
        FailAndExit "sync source file missing: $src -- Script 09 may not have written it, refusing to proceed with a stale/incomplete panel copy"
    }
    try {
        Copy-Item $src "data\agmarknet_weekly\" -Force -ErrorAction Stop
    } catch {
        Pop-Location
        FailAndExit "failed to copy $src into data\agmarknet_weekly\ : $($_.Exception.Message)"
    }
}

Log "--- Script 23: retraining production models ---"
$code = RunLogged $Python @("scripts\23_Train_Production_Models.py")
if ($code -ne 0) { Pop-Location; FailAndExit "Script 23 failed (exit $code) -- production models may be in an inconsistent state, check Model_Output/production_models/ before trusting the dashboard" }

# FIXED 2026-09-02 (audit finding, confirmed): this job retrained Script 23's
# models and committed Model_Output/production_models/*.joblib, but never
# re-synced the web dashboard's own bundled copy (web/data/production_models/)
# or regenerated its JS model translations (web/backend/src/models/*.js) --
# so the live public dashboard could silently keep serving the PREVIOUS
# training run indefinitely while this job kept reporting success. Follows
# the exact steps documented in web/README.md's "Regenerating the models"
# section. Placed after Script 23 (needs the freshly retrained models) and
# before the git commit (the new web/ files must be committed alongside the
# retrained Model_Output/ files, or the two would drift apart again).
Log "--- Syncing retrained models into web/ + regenerating JS models ---"
$webModelFiles = @("feature_columns.json", "feature_ranges.json", "model_uncertainty.json",
                    "macro_climate_staleness.json", "reference_rows.csv", "price_history.csv")
foreach ($f in $webModelFiles) {
    $src = Join-Path $RepoRoot "Model_Output\production_models\$f"
    if (-not (Test-Path $src)) {
        Pop-Location
        FailAndExit "web bundle sync source missing: $src -- Script 23 may not have written it"
    }
    try {
        Copy-Item $src (Join-Path $RepoRoot "web\data\production_models\") -Force -ErrorAction Stop
    } catch {
        Pop-Location
        FailAndExit "failed to copy $src into web\data\production_models\ : $($_.Exception.Message)"
    }
}

Push-Location (Join-Path $RepoRoot "web")
$code = RunLogged $Python @("generate_js_models.py")
if ($code -ne 0) {
    Pop-Location; Pop-Location
    FailAndExit "generate_js_models.py failed (exit $code) -- web dashboard's bundled JS models NOT updated, still serving the previous training run"
}
$code = RunLogged "node" @("backend\src\models\__fixtures__\verify.mjs")
if ($code -ne 0) {
    Pop-Location; Pop-Location
    FailAndExit "verify.mjs cross-language parity check FAILED (exit $code) -- newly generated JS models do not reproduce Python predictions, refusing to commit them; check the log above for which model(s) diverged"
}
Pop-Location

Log "--- Script 44: final sanity check (informational) ---"
$code = RunLogged $Python @("scripts\44_Pipeline_Sanity_Check.py")
if ($code -ne 0) {
    Log "Script 44 reported FAILures after a successful Script 09 + 23 run -- unexpected, needs manual review. NOT aborting the commit since the pipeline itself completed; investigate before trusting the dashboard."
} else {
    Log "Script 44: clean."
}

Log "--- Local git commit (no push) ---"
$filesToAdd = @(
    "data\agmarknet_weekly\tomato_weekly_panel.csv",
    "data\agmarknet_weekly\onion_weekly_panel.csv",
    "data\agmarknet_weekly\potato_weekly_panel.csv",
    "Model_Output\production_models\feature_ranges.json",
    "Model_Output\production_models\macro_climate_staleness.json",
    "Model_Output\production_models\model_uncertainty.json",
    "Model_Output\production_models\price_history.csv",
    "Model_Output\production_models\reference_rows.csv",
    "Model_Output\production_models\feature_columns.json",
    "Model_Output\production_models\tomato_1w.joblib", "Model_Output\production_models\tomato_4w.joblib",
    "Model_Output\production_models\tomato_13w.joblib", "Model_Output\production_models\tomato_26w.joblib",
    "Model_Output\production_models\onion_1w.joblib", "Model_Output\production_models\onion_4w.joblib",
    "Model_Output\production_models\onion_13w.joblib", "Model_Output\production_models\onion_26w.joblib",
    "Model_Output\production_models\potato_1w.joblib", "Model_Output\production_models\potato_4w.joblib",
    "Model_Output\production_models\potato_13w.joblib", "Model_Output\production_models\potato_26w.joblib",
    # web/ bundle synced + regenerated above -- must travel with the retrain
    # it was generated from, or the live dashboard's models drift out of
    # step with Model_Output/production_models/ again.
    "web\data\production_models\feature_columns.json", "web\data\production_models\feature_ranges.json",
    "web\data\production_models\model_uncertainty.json", "web\data\production_models\macro_climate_staleness.json",
    "web\data\production_models\reference_rows.csv", "web\data\production_models\price_history.csv",
    "web\backend\src\models\feature_columns.json",
    "web\backend\src\models\tomato_1w.js", "web\backend\src\models\tomato_4w.js",
    "web\backend\src\models\tomato_13w.js", "web\backend\src\models\tomato_26w.js",
    "web\backend\src\models\onion_1w.js", "web\backend\src\models\onion_4w.js",
    "web\backend\src\models\onion_13w.js", "web\backend\src\models\onion_26w.js",
    "web\backend\src\models\potato_1w.js", "web\backend\src\models\potato_4w.js",
    "web\backend\src\models\potato_13w.js", "web\backend\src\models\potato_26w.js"
)
# FIXED 2026-09-02 (audit finding, confirmed): neither git call's exit code
# was checked -- a failed `git add` (e.g. a path typo) or failed `git commit`
# (e.g. a pre-commit hook rejection, or an index lock held by a concurrent
# git process) would still fall through to "Committed locally" and
# "Weekly Refresh Completed OK" (exit 0), silently reporting success on a
# run whose retrained models were never actually committed.
& git add $filesToAdd 2>&1 | ForEach-Object { $LogStream.WriteLine($_) }
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    FailAndExit "git add failed (exit $LASTEXITCODE) -- see log for git's own error output"
}
$commitMsg = "Automated weekly data refresh ($ts) -- crops: $($mergedCrops -join ', ')"
& git commit -m $commitMsg 2>&1 | ForEach-Object { $LogStream.WriteLine($_) }
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    FailAndExit "git commit failed (exit $LASTEXITCODE) -- changes are staged but NOT committed; see log for git's own error output (e.g. a pre-commit hook rejection or an index lock from a concurrent git process)"
}
Log "Committed locally (not pushed): $commitMsg"

Pop-Location

Log "=== Weekly Refresh Completed OK ==="
$LogStream.Flush(); $LogStream.Close()
exit 0
