# One-time setup: registers the weekly refresh as a Windows Task Scheduler job.
# Run this ONCE from an ordinary PowerShell window (no admin rights needed for
# a per-user task). Re-run it any time to update the schedule -- it deletes
# and recreates the task rather than erroring if it already exists.
#
# Schedule: every Tuesday at 03:00 local time.
#
# LogonType note: S4U (run whether logged on or not) needs the "Log on as a
# batch job" right, which a standard account often doesn't have -- confirmed
# on this machine 2026-08-29 (Register-ScheduledTask failed with "Access is
# denied" under S4U). Defaults to Interactive here instead: the task runs
# reliably as long as you're logged in (even locked/idle is fine), which
# covers a weekly 03:00 run on a machine that's normally left on. If you
# want true logged-off execution, grant yourself the batch-logon right first
# (secpol.msc -> Local Policies -> User Rights Assignment -> "Log on as a
# batch job" -> add your account) and change LogonType to S4U below.

$TaskName = "TOP_Digital_Twin_Weekly_Refresh"
$ScriptPath = "C:\Users\masro\Documents\TOP_Digital_Twin\scripts\weekly_refresh\run_weekly_refresh.ps1"

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""

$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 3:00AM

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -WakeToRun

$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
        -Settings $Settings -Principal $Principal `
        -Description "TOP Digital Twin: weekly Agmarknet data pull + panel rebuild + model retrain (tomato/onion/potato)." `
        -ErrorAction Stop | Out-Null
} catch {
    Write-Host "REGISTRATION FAILED: $_"
    Write-Host "The task was NOT created. Common cause: insufficient rights for the chosen -LogonType."
    exit 1
}

# Verify it actually exists rather than trusting the cmdlet didn't throw.
$verify = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $verify) {
    Write-Host "REGISTRATION FAILED: Register-ScheduledTask did not throw, but the task is not present afterward."
    exit 1
}

Write-Host "Registered task '$TaskName' -- runs every Tuesday at 03:00 (LogonType: Interactive)."
Write-Host "View/edit it any time in Task Scheduler (taskschd.msc), or run it immediately with:"
Write-Host "  Start-ScheduledTask -TaskName `"$TaskName`""
