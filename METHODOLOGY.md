# Methodology note 

Scope: historical news-to-price event study for NVDA with SPY as benchmark, horizons of 60 and 390
regular-session trading minutes. The output describes what prices did *after* news became
available. It is a benchmark-adjusted association, not a causal estimate of a news effect and not
evidence of predictive value or tradable profit.

## 1. Equations

For an instrument *i* over `[t0, t1]` (prices are bar closes):

- Stock return: `R_i = (P_i,t1 − P_i,t0) / P_i,t0`
- Benchmark return: `R_m = (P_m,t1 − P_m,t0) / P_m,t0` (SPY, same two timestamps)
- Abnormal return: `AR = R_i − R_m`, reported for `h ∈ {60, 390}`

All returns are fractions (0.01 = 1.0%). Prices that are missing, non-finite or ≤ 0 yield NaN, never
0 and never ±inf. Worked example from the brief (NVDA 100→101→102, SPY 500→502→504) gives
AR60 = 0.006 and AR390 = 0.012; this is a unit test.

Summary statistics per horizon (`summarize_event_impacts`): count, mean `AR̄ = (1/N)ΣAR`, median,
`s = sqrt(Σ(AR−AR̄)²/(N−1))`, mean absolute abnormal return `|AR|̄ = (1/N)Σ|AR|`, fraction positive
(strictly > 0), quantiles (5/25/75/95%).

Uncertainty for a mean `m` (applied to both `AR` and `|AR|`):

- Naive: `m ± t_{0.975,N−1} · s/√N`. Valid only for independent observations.
- Cluster-robust: `Var(m̂) = G/(G−1) · Σ_g (Σ_{i∈g}(x_i − m̂))² / N²`, t with `G−1` df, where clusters
  are overlap clusters (below). With one event per cluster it equals the naive interval. Few
  clusters (G small) make it unreliable; the count `n_clusters` is reported next to it.

Group difference (`compare_groups`): `Δ = mean(AR_a) − mean(AR_b)` with a Welch interval, and the same
for `|AR|`. Treat as descriptive; it assumes independence.

## 2. Timing policy

1. Timestamps are stored and compared in UTC. Naive input timestamps are assumed to be UTC;
   tz-aware ones are converted. New York time is used only to decide session boundaries (DST-aware).
2. **Event availability** = `first_seen_at_utc` when present, else `published_at_utc`
   (`availability_source` records which; rows using publisher time carry `publisher_time_only`).
3. **Entry**: the first complete regular-session bar whose `bar_start_utc` ≥ availability.
   `P_t0` = that bar's close, observed at the bar's end; `entry_timestamp_utc` = that bar end.
   Example: news at 10:00:30 → bar 10:01–10:02 → entry 10:02 (the 10:00–10:01 bar mixes pre- and
   post-news trading). News at exactly 10:00:00 may use the 10:00–10:01 bar.
4. **Off-hours**: pre-open, after-close, weekend and holiday news enter at the first bar of the next
   regular session (the 09:30–09:31 bar, close at 09:31). The overnight gap is not part of any
   return, and the first minute of the session is not credited either.
5. **Horizon**: `h` regular-session minutes along the exchange calendar's bar-end grid, so overnight,
   weekends, holidays and after-hours do not count. A 390-minute window from 10:02 ends the next
   session at 10:02; across a 13:00 early close it ends correspondingly later. Horizons are exact
   for 1-minute bars; for coarser bars they must be a multiple of the bar length (validated, never
   silently rounded). Both NVDA and SPY are read at the identical entry and end bars.
6. **Calendar**: built-in rule-based NYSE calendar (09:30–16:00 ET; 13:00 early closes the day after
   Thanksgiving, Jul 3 and Dec 24 when Mon–Thu; Juneteenth from 2022; a short list of known
   unscheduled closures). It is not guaranteed complete: add `extra_holidays` / `extra_early_closes`
   and run `check_calendar_against_prices`. A session the calendar believes exists but the data lack
   produces NaN plus a flag, not a wrong number.

No entry or window is chosen using realized returns; the rule is fixed and deterministic.

## 3. Missing data and quality flags

Missing returns are preserved as NaN. `data_quality_flag` is `"ok"` or `"; "`-joined tokens:

| Token | Meaning |
|---|---|
| `missing_timestamp` | no usable availability time; row kept, all returns NaN |
| `publisher_time_only` | no `first_seen_at_utc`; publisher clock is the only timestamp |
| `delayed_availability` | `first_seen − published` > threshold (default 60 min) |
| `first_seen_before_published` | clock anomaly; first_seen still used |
| `off_hours_release` / `late_session_rollover` | released outside a session / in the last bar |
| `early_close_in_window` | an early-close session lies in the longest window |
| `missing_bar_{entry,end_Hm}_{stock,market}` | bar slot exists but no usable price |
| `beyond_price_data_*`, `before_price_data_*`, `no_price_data_*` | price file ends/starts/lacks ticker |
| `shared_entry_bar` | other articles for the ticker use the same entry bar |
| `duplicate_article_id` | `article_id` not unique (rows kept) |
| `suspected_unadjusted_split_Hm_{stock,market}` | consecutive closes jump by ~a split ratio inside the window; that return set to NaN |
| `corporate_action_in_window_Hm_*` | window spans an action from the optional `corporate_actions` table (informational) |

## 4. Overlap and duplicate coverage

Return intervals `(p, p+H]` of two events (entry positions `p` on the trading-minute grid, `H` = longest
horizon in bars) intersect iff `|p_i − p_j| < H`. `overlapping_event_flag` is true when another article
for the same ticker satisfies that (this includes identical entry bars). `n_overlapping_events` counts
them, and `overlap_cluster_id` chains overlaps into clusters for grouped/resampled inference.
Windows that only touch (`|Δ| = H`) are not overlapping. News-dense periods can put most events in a few
clusters, so effective sample size can be far below `N`. Run sensitivity checks with
`exclude_overlapping=True`.

## 5. Corporate actions

Prices must already be split-consistent. As a safeguard, a jump between consecutive regular-session
closes with `|r−1| ≥ 35%` that lies within 5% of a split ratio (2, 3, 4, 5, 10, 20, 25, 50 or their
inverses) is treated as an unadjusted split. This heuristic can in principle misfire on a genuine ~50%
crash; tune `split_tolerance`/`split_min_move` or disable with `detect_splits=False`. It does not
catch dividends or small adjustments.

## 6. Assumptions and limitations

- Timestamps from vendors may lag or lead true availability; `first_seen` quality is unknown until
  data arrive.
- Entry at the bar close after availability is conservative: it forgoes the first 0–2 minutes of
  reaction, so very fast reactions are under-measured. A 1-minute bar close is also not a guaranteed fill.
- Results depend on the bar convention; the 5-minute fallback shifts entry by up to 5 minutes.
- SPY is a simple benchmark (beta fixed at 1). NVDA's market beta ≠ 1, so AR contains market-timing
  noise; a market-model AR is a possible later extension, not in scope.
- Events cluster around earnings, product events and macro releases; pooled statistics mix regimes.
- Absence of statistical significance with a small sample is a result, not a failure.
- Nothing here estimates causality. Prediction is evaluated separately (Person 2), chronologically
  out of sample.

## 7. Decisions to confirm with Person 2

1. `entry_timestamp_utc` is the bar **end** (when `P_t0` is observed); `entry_bar_start_utc` is also in the file.
2. A missing entry bar gives NaN rather than slipping to the next available bar.
3. Overlap is defined at the longest horizon (390 trading minutes).
4. Extra (non-required) columns are present: prices, exit timestamps, `usable_Hm`, `n_overlapping_events`,
   `overlap_cluster_id`, `release_session_state`, `availability_source`, `entry_session_date`.
