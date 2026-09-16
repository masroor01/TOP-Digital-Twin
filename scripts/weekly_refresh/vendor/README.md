# Vendored Agmarknet scraper

`agmarknet_onion_prices.py` (+ `requirements.txt`) is a synced copy of
`C:\Users\masro\Documents\Codex\2026-05-14\assuming-you-re-an-expert-provide\agmarknet_onion_prices.py`
-- an external, non-versioned script outside this repo, despite its name it's
commodity-parameterized (`--commodity Tomato`/`Onion`/`Potato`).

Vendored here 2026-09-16 so the GitHub Actions weekly-refresh workflow can
check it out along with the rest of the repo (the runner has no access to
the local Codex folder). The **local** Windows Task Scheduler automation
(`scripts/weekly_refresh/run_weekly_refresh.ps1`) still points at the
original Codex-folder copy, unchanged -- these two copies are independent,
not synced automatically. If the original scraper gets a real fix, re-copy
it here by hand; there's no shared source of truth between the two.
