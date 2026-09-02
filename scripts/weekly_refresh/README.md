# Weekly Automated Refresh

Pulls fresh Agmarknet data for tomato/onion/potato, validates it, merges it
into the trusted raw files, rebuilds the weekly panels, retrains the 12
production models, and commits the result locally — fully unattended, once
a week.

## Pieces

| File | What it does |
|---|---|
| `run_weekly_refresh.ps1` | The orchestrator. This is what Task Scheduler runs. |
| `validate_scrape.py` | Gate before merging: schema, price sanity, market_id conflicts, staleness, internal coverage continuity, row density, overlap-vs-trusted-file comparison. Exits 1 on any failure. |
| `merge_scrape.py` | Replaces the trusted file's current-year portion with the new scrape, backing up first. Only run after `validate_scrape.py` passes. Aborts (unless `--force`) if the new rows are under half the size of what they'd replace. |
| `register_task.ps1` | One-time setup — registers the Tuesday 03:00 Task Scheduler entry. |
| `staging/` | Scratch folder for each week's raw scrape output. Gitignored. |
| `../../logs/weekly_refresh/` | One timestamped log per run; failed runs get `_FAILED` in the filename. Gitignored. |

**Fixed 2026-09-02** (audit finding, confirmed live risk): `validate_scrape.py` originally defined a `MAX_SCRAPE_AGE_DAYS` staleness tolerance but never actually checked it, and had no check for a silent gap or thin-density stretch *inside* the scrape's own claimed date range. Since `merge_scrape.py` replaces every trusted row from the new file's own minimum date onward, a scrape that stalled or was cut off mid-run (network drop, pagination bug) could pass every original check and then delete months of real trusted data on merge — while logging "refresh complete." Added 3 new checks to `validate_scrape.py` (staleness, internal gap continuity, row density vs. the trusted file's recent history) plus an independent row-count safety net directly in `merge_scrape.py` as defense in depth. All new checks tested against synthetic clean/truncated/stale/thin-density scenarios before trusting them.

**Fixed 2026-09-02, second pass** (audit finding, confirmed): `run_weekly_refresh.ps1` had two spots inconsistent with its own established safety pattern (every other step already stops the run via `FailAndExit` on failure) — the panel-file sync step (`Copy-Item` x4) had no existence check or error handling, and the final `git add`/`git commit` calls never checked their exit codes. A missing sync source or a failed commit (pre-commit hook rejection, index lock from a concurrent git process — a real scenario, hit once this same session) would have silently fallen through to "Weekly Refresh Completed OK." Both now stop the run with a clear reason via the same `FailAndExit` path everything else uses. Verified against an isolated sandbox test (not the real repo/Downloads): missing-file detection, clean-case no-false-positive, and git-failure detection all confirmed working before trusting this.

**Known gap, NOT fixed** (flagged by the same audit, deliberately left for a separate decision rather than silently expanding this automation's scope): this script retrains and commits Script 23's output but never re-syncs the live dashboard's bundled models — it doesn't run `web/generate_js_models.py`, `web/backend/src/models/__fixtures__/verify.mjs`, or copy the refreshed reference data into `web/data/` (see `web/README.md`'s "Updating the bundled data" section for the manual steps this would need to replicate). The public dashboard can silently drift stale relative to the latest local retrain until someone does this by hand. Not added here because it's a genuine scope expansion of an unattended production job (new Node/npm dependency at runtime, added runtime, a new class of failure to handle) rather than a bug fix restoring already-intended behavior — worth a deliberate decision, not a silent addition.

## Data source

`C:\Users\masro\Documents\Codex\2026-05-14\assuming-you-re-an-expert-provide\agmarknet_onion_prices.py`
— despite the name, it's commodity-parameterized (`--commodity Tomato` /
`Onion` / `Potato`) and hits `api.agmarknet.gov.in` directly. No login, no
captcha. Confirmed 2026-08-27 to reach real data through the current day,
and its `market_id` scheme matches what's already baked into the trusted
files exactly (same AGMARKNET 2.0 source as every manual pull this project
has used).

## Scrape scope

Tomato and onion are scraped all-India (their panels draw on 22+ states).
**Potato is restricted to `--states "West Bengal,Uttarakhand"`** — the only
two states that have ever survived into the potato production panel (Script
09's 8-year balanced-panel + price-clip filters drop everything else
regardless). Confirmed 2026-08-29 via a live all-India potato scrape that
most of what gets filtered isn't even comparable data: Tamil Nadu's "potato
markets" are the Uzhavar Sandhai retail farmer-market scheme (median 0.22
tonnes/report vs West Bengal's 12 — retail lots, not wholesale mandi trade),
and Kerala is a genuine but thin, import-dependent consumption market. This
also means the trusted `potato_all_india_apmcs_2000_2026.csv` no longer
gains all-India history each week going forward — acceptable since none of
that ever reached the model, but worth knowing if it's ever wanted for a
separate regional-price study (a one-off unrestricted scrape can always be
run by hand for that).

## Safety model

- Per-crop failures (scrape/validate/merge) skip only that crop — the other
  two still proceed, and a failed crop's trusted file is left untouched.
- If all three crops fail, the job stops before Script 09/23 ever run.
- If Script 09 or Script 23 itself fails, the job stops immediately —
  nothing gets committed on a broken run.
- Script 44 runs at the end for visibility/logging but does not block the
  commit — after a successful Script 09 → 23 sequence it's expected to pass
  clean (confirmed 2026-08-29, 41/41 checks); if it doesn't, that's flagged
  clearly in the log as needing manual review, not silently ignored.
- **Never auto-pushes to GitHub.** Commits land in the local history only;
  review and `git push` manually when ready.
- Each merge keeps a rolling backup (`<file>.backup_YYYYMMDD_HHMMSS`,
  last 6 kept) next to the trusted CSV, so a bad week can be rolled back by
  hand.

## Setup

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\masro\Documents\TOP_Digital_Twin\scripts\weekly_refresh\register_task.ps1"
```

Registers a per-user Task Scheduler job (`TOP_Digital_Twin_Weekly_Refresh`),
Tuesdays at 03:00. Defaults to `-LogonType Interactive` (runs reliably as
long as you're logged in, even locked/idle — fine for a machine normally
left on). `-LogonType S4U` (runs even fully logged off) needs the "Log on
as a batch job" right, which this account didn't have by default — confirmed
2026-08-29, `Register-ScheduledTask` failed with Access Denied under S4U.
Grant yourself that right first (`secpol.msc` → Local Policies → User
Rights Assignment → "Log on as a batch job") if you want true logged-off
execution, then switch the LogonType in `register_task.ps1`.

To run it manually / test it on demand instead of waiting for Tuesday:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\masro\Documents\TOP_Digital_Twin\scripts\weekly_refresh\run_weekly_refresh.ps1"
```

## Why Tuesday, why local, why no auto-push

- **Tuesday** gives weekend mandi reporting a day to settle before pulling.
- **Local Task Scheduler, not a cloud cron** — everything here (Downloads,
  the repo, the trained models, the dashboard's local data) lives on this
  machine. A cloud-scheduled job would have no access to any of it.
- **No auto-push** — keeps GitHub (and the live Streamlit redeploy it
  triggers) under manual review before anything public changes.
