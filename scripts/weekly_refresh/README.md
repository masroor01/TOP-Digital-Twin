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

~~**Known gap, NOT fixed**~~ — **closed 2026-09-02** (commit `e52de53`, "Harden weekly-refresh automation: strict asserts, web model resync, marketIds cap"). This paragraph previously said the job retrained Script 23's output but never re-synced the live dashboard's bundled models; that's no longer true. `run_weekly_refresh.ps1` now, immediately after a successful Script 23 run and before the git commit: copies the 6 refreshed production-model files into `web/data/production_models/`, runs `web/generate_js_models.py` to regenerate the 12 JS models, then runs `web/backend/src/models/__fixtures__/verify.mjs` and **refuses to commit anything if the cross-language parity check fails**. The `web/` files are added to the exact same commit as the `Model_Output/production_models/*` files they were generated from, so the two can no longer drift apart. This corrects a real doc-staleness bug: the fix landed in the same day's audit pass as the two entries above, but this specific paragraph was never updated to match — caught 2026-09-14 by reading the actual script instead of trusting this file's prose. **Caveat, not yet resolved**: this code path has never actually executed successfully in a live run — every scheduled run since 2026-09-02 has failed earlier in the pipeline (see `logs/weekly_refresh/`), so it's structurally sound on read-through but unverified in practice. Treat it as "should work" until a real run's log shows the "Syncing retrained models into web/" section completing with a clean `verify.mjs` pass.

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

## Local (.ps1) vs CI (.py): known divergences

Two independent orchestrators now exist and they are **not kept in sync automatically**.
The cloud path (`.github/workflows/weekly_refresh.yml` -> `run_weekly_refresh_ci.py`) is the
one the live dashboard depends on; the local Task Scheduler path (`run_weekly_refresh.ps1`)
is the older fallback. Differences to know about before trusting either one:

| Aspect | Local `.ps1` | CI `.py` (GitHub Actions) |
|---|---|---|
| Satellite/climate top-up (GEE scripts 01-05) and Script 14 | **Not run.** ERA5/CHIRPS/S2/MODIS features stay as of the last manual or CI update, so the dashboard's climate/satellite feeds go stale | Run every cycle (S2 NDVI top-up is the ~1.6 h bottleneck of a ~2.5 h run); Script 14 is skipped, with a log line, if any GEE step is incomplete |
| Raw trusted CSVs | Fixed local Downloads folder | `TOP_DOWNLOADS_DIR`, populated by the workflow (`rclone copy`) before the script runs |
| Scraper pacing | `--sleep`/`--retries` tuned up 2026-09-17 | `--sleep 1.5 --retries 6` from the start |
| After the run | Commits to local history; never pushes | Commits on the runner; the workflow opens a **pull request** (Hostinger only redeploys on a push to master, so nothing public changes until a human merges) |
| Safety gates (strict merge asserts, parity `verify.mjs`, sanity suite Script 44) | Present | Present (same intent) |

Consequences: a local-only run refreshes prices and models but **not** the satellite/climate
layers, so retrained models then use stale climate/satellite rows (carried forward and flagged
as stale). Prefer the CI path for production refreshes. If the local path is ever relied on
again, add the GEE steps (or run `scripts/gee_auto/*` and Script 14 manually first). Any change
to one orchestrator must be mirrored by hand in the other.

## Captcha block (from 2026-10-27): the automated price scrape no longer works

On 2026-10-27 the scheduled refresh ran 4 h 48 min on the tomato scrape and was cancelled at the job time limit. Cause: the portal's data endpoint
(`api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity`) now answers **HTTP 400 `TOKEN_OR_CAPTCHA_REQUIRED`** ("Captcha key and
captcha value are required") to every request; the scraper retried each of about 400 requests six times. A captcha cannot and must not be solved or
bypassed by this scraper, so:

- `vendor/agmarknet_onion_prices.py` now raises `CaptchaRequired` on that response and exits with code 3 within seconds;
- `run_weekly_refresh_ci.py` stops the scrape for all crops on code 3 and fails the run with the message "AGMARKNET REQUIRES A CAPTCHA", so a blocked
  week costs about a minute, not five hours; nothing is merged and the trusted files are untouched;
- the incremental window (`--since-month`, `--until-month`; three weeks of overlap) is kept for when scraping is possible again, and a per-crop time
  budget (`SCRAPE_TIMEOUT_MIN`, default 75) stops any single crop from using the whole job.

### Preferred input: the combined "Daily Price Arrival Report" (tested 2026-10-28)
The portal can also export ONE file per commodity with price and arrival quantity on every row ("Daily Price Arrival Report-DD-MM-YYYY to DD-MM-YYYY.csv").
`portal_reports.py` accepts it directly (put that single file in `portal_reports/<crop>/`; if a combined file and a price/arrival pair are both present the
newest is used). Rows are variety/grade rows: prices are averaged and arrival quantities summed per (state, district, market, date).
Tested on the tomato file Nov 2025 to 28 Oct 2026 (214,023 rows, 28 states, 1,282 markets): against the trusted tomato file for the same weeks, prices agree
within 1 % in 100 % of the matched (market, date) rows and arrival totals within 1 % in 99.1 %; all validate_scrape.py gates pass; a dry-run merge on a copy
advances the trusted file's newest date from 2026-10-06 to 2026-10-28. 22 markets in the file are new, 69 trusted markets are absent from it (markets that did
not report in the window or are spelled differently). Download only the weeks since the last refresh plus about three weeks of overlap (the merge replaces
everything from the file's first date onward, and replaced rows become one averaged row per market and day); one such file is needed per crop (the tomato
file above has no onion or potato rows).

### Weekly procedure now (portal reports; implemented 2026-10-28, `portal_reports.py`)
1. On agmarknet.gov.in -> Price & Arrivals, for EACH crop (Tomato, Onion, Potato) download the **Daily Price Report** and the **Daily Arrival Report**
   as CSV for the period since the last refresh (the newest data in the trusted files is the week of 2026-10-05; a period of several weeks is fine, the merge
   replaces everything from the file's first date onward). The portal asks for a captcha; a person solves it.
2. Put the two files for each crop on Drive in `TOP_Digital_Twin_Raw_Data/portal_reports/<crop>/` (crop in lower case: tomato, onion, potato). Any file names
   work as long as they contain "Price" or "Arrival"; the newest pair by modified time is used.
3. The weekly workflow pulls that folder, converts each pair to the scraper's 20-column schema (`portal_reports.build_scrape`: variety rows averaged,
   market_id by normalised state and market name, new ids for new markets), then runs the same gates as before (`validate_scrape.py`, `merge_scrape.py`)
   and the rest of the pipeline. A crop without reports is skipped (or tried through the scraper, which stops at the captcha). Locally:
   `python scripts/weekly_refresh/portal_reports.py --crop onion --price "..." --arrival "..." --trusted <trusted csv> --out <staging csv>`.
Test: the onion reports of Jan-Jul 2026 reproduce the rows already in the trusted file (120,502 of 120,834 (market, date) rows overlap; prices within 1 % in 100 %,
arrivals within 1 % in 99.7 %).

**Known data issue found while building this:** in the trusted onion file the market_ids 4846 and 4848-4852 are each shared by two different markets in different
states (for example 4846 = Varanasi(F&V) APMC, Uttar Pradesh, to Jan 2026, and Karimangalam(Uzhavar Sandhai), Tamil Nadu, from Feb 2026), a collision between ids
assigned by Script 09b and ids from the scraper's directory. Script 09 merges each pair into one market series. A repair (unique ids for the later-seen market of
each pair, applied in Script 09) and a panel rebuild are still to be done; `portal_reports.py` reports the collision as a warning.

**data.gov.in feed (tested 2026-10-28 with the project's free API key, which is not stored in the repository):** the resource "Current Daily Price of Various
Commodities from Various Markets (Mandi)" answers, but it holds only the CURRENT day (657 onion records, all dated 27/10/2026) and has no arrival quantity
(fields: state, district, market, commodity, variety, grade, arrival_date, min/max/modal price). The pipeline needs arrivals (rows without arrivals are dropped and
arrival features are built from them) and history, so it cannot replace the portal reports; at most a daily collector could keep a price-only archive.

Ways forward (none implemented yet): (1) download the portal's "Daily Price Report" and "Daily Arrival Report" by hand each week (a person solves the
captcha) and merge them into the trusted files, as was done for onion 2026 (`09b_Merge_Onion_2026_Update.py`, to be generalised to all crops);
(2) an official open-data feed (data.gov.in "Current Daily Price of Various Commodities from Various Markets (Mandi)" needs a free API key); (3) ask
the Directorate of Marketing and Inspection for bulk access.
