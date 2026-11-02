# Track C: public supply-side indicators for onion (inventory, 2026-11-02)

Purpose: find series that could lead the price (production, area, stocks, buffer, exports) before writing any test protocol. Checked by page fetch on 2026-11-02; nothing was downloaded or analysed, and no test was run.

## Result
No onion production, stock or buffer-stock series covering 2017-2026 was reachable in usable form. A test protocol is therefore not written yet; with about 12 spike episodes, a series with fewer than ten usable years would not give a meaningful test anyway.

| Source | Reached | Onion content | Verdict |
|---|---|---|---|
| NHB state area and production (nhb.gov.in/Statistics.aspx) | yes | state files 2011-12 to 2018-19 only; online query 2000-01 to 2011-12 | stops before the study window |
| NHB Price and Arrivals reports | listed, not opened | unverified | prices and arrivals only, which the twin already has |
| DES / Agriculture Ministry advance estimates (desagri.gov.in, agriwelfare.gov.in) | no (TLS error; stub page) | not seen | unverified; try other mirrors or the user's copies |
| PIB / Dept. of Consumer Affairs buffer stock | no (403, DNS failure) | not seen | unverified; likely scattered press notes |
| DGCIS TradeStat exports (tradestat.commerce.gov.in) | yes | fiscal years 2018-19 to 2025-26; HS 070310 expected | candidate; monthly option not confirmed; HS/unit changes from April 2026 |
| APEDA AgriExchange | yes (form) | product list did not render | unverified |
| NHRDF | no (403) | not seen | unverified |
| Maharashtra crop sowing reports (krishi.maharashtra.gov.in) | yes | weekly PDFs, 2026 season seen; onion not confirmed; archive depth unknown | candidate for sowing area, Maharashtra only |

## Open
- Export quantities are already known to be policy-driven (export controls), so they are an outcome of the same policy as the price, not an independent early signal; the Track B2 and onion-policy results should be kept in mind.
- Anything further needs data the user can supply (advance-estimate PDFs, buffer-stock quantities, Maharashtra sowing archive) or manual download.
