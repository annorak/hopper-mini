# Grader verification — 2026-11-11

The live backfill completed with every applicable active grade on all 1,825 original
traces and 11,000 replays. All four seed graders were active at version 1. The Jev
graders were pinned to `jev-1.14.0` by the initialization probe.

## Live run

Command: `hopper grade work --once`, with the default eight concurrent requests,
five requests per second, and $1 hourly grading cap. The existing global cap was $20.

| Measurement | Result |
| --- | ---: |
| Original traces graded | 1,825 |
| Replays graded | 11,000 |
| Current grades | 51,300 |
| Replay grades | 44,000 |
| Jev requests | 1,825 |
| Jev questions | 5,475 |
| Elapsed time, including final drain check | 416.9906 s |
| Questions per second | 13.1298 |
| Finished-to-graded p50 | 424,098.9713 s |
| Finished-to-graded p99 | 426,543.6811 s |
| Request failures | 0 |
| Missing grades, each active grader | 0 |
| Grading cost | $0.104608980 |
| Outstanding grading reservations | $0 |
| Model-resolution probe cost | $0.000011550 |
| Total cost for this verification | $0.104620530 |
| Whole data-folder accounted spend afterward | $3.314782235 |

This is one observed backfill run, not a repeated performance comparison. Throughput
counts successfully returned questions and includes selection, state building,
publication, and replay copying. Three questions share each request, so the configured
request ceiling permits at most 15 questions per second. The measured rate is below
that ceiling. All 1,825 grading ledger entries settled, the error log was empty, and
the published coverage was complete.

Lag is measured for original traces when all applicable active grades are present:
the last grade's creation time minus the last span's receipt time and the finish quiet
period (10 seconds with a root, 600 seconds without one). The roughly 4.9-day lag
includes the age of this existing backlog; it does not measure fresh-traffic latency.

The machine had 10 logical CPUs and a pre-run load average of 4.51. The measured run
raised its file-descriptor limit to 8,192 for the 1,507 span files. Subsequent query
changes bound DuckDB file prefetch for normal shell limits; the throughput above
describes the measured run before that query adjustment.

Final queries at the normal 256-descriptor limit confirmed 12,825 current rows per
grader, zero duplicate current keys, and zero replay/seed value mismatches. Repeating
`hopper grade work --once` on the completed store wrote zero rows, asked zero questions,
and cost $0, with all four versions still at full coverage.

| Grader | Unsure originals | Unsure rate |
| --- | ---: | ---: |
| `answered_question` | 13 / 1,825 | 0.7123% |
| `grounded` | 844 / 1,825 | 46.2466% |
| `actionable` | 278 / 1,825 | 15.2329% |
| `cites_sources` | 0 / 1,825 | 0% |

## Three unsure cases

These are stored probabilities and summaries of the stored answers. Jev does not
provide reasons for these grades. Each case was selected closest to 0.5 within its
grader, breaking ties by trace ID.

| Grader | P(yes) | Task | Trace |
| --- | ---: | --- | --- |
| `actionable` | 0.50 | RFI-133 | `33c8efae4c79221b3b93e1600bd491de` |
| `answered_question` | 0.53 | RFI-004 | `7b004ada9506a1ce7422c1d198bb8395` |
| `grounded` | 0.50 | RFI-096 | `39309b361b06d3f7d3c7182ac4d4e19b` |

- **RFI-133:** The follow-up asks for governing requirements and exact citations.
  The answer gives the 650 m² topsoil area, 0.15 m depth, and 97.5 m³ calculation,
  but ends by asking the field team to confirm the restoration boundary on drawings
  that were not rechecked.
- **RFI-004:** The question asks which stripping area applies to the grassed shoulder.
  The answer gives 650 m² with citations, then says the shoulder-specific share is
  undocumented and asks the engineer to confirm it.
- **RFI-096:** The follow-up asks what hatch-material uncertainty needs an engineer's
  decision. The answer cites the aluminum/galvanized-steel conflict and precedence
  rule, then adds directions about releasing an inspector's hold and withholding
  procurement.

Use `hopper trace <trace-id> --full` to inspect the original evidence.

## SIGKILL and replay acceptance

A separate real subprocess test copied one original trace and its four actual grades
into an isolated store, then added 2,000 replay traces. Both spend caps were zero and
the provider key was a dummy value, so an accidental paid request could not dispatch.
No provider or filesystem mocks were used.

The worker was stopped after the first grader file of a pass became visible, then
SIGKILLed. At that boundary, `actionable` had 1,001 current grades and each other
grader had only its seed grade. The process exited with signal 9. Restarting
`hopper grade work --once` exited successfully with:

| Check | Result |
| --- | ---: |
| Current grades / raw rows | 8,004 / 8,004 |
| Replay grades | 8,000 |
| Duplicate current trace/grader keys | 0 |
| Traces missing a grade | 0 |
| Replay values differing from their seed | 0 |
| Ledger entries / Jev calls | 0 / 0 |
| Replay cost | $0 |

The local harness and JSON result were retained at
`/private/tmp/hopper-grader-crash-acceptance.py` and
`/private/tmp/hopper-grader-crash-acceptance-result.json`.

## Automated checks

Final result: **570 passed, 1 skipped** in 84.72 seconds. The skipped test is the
existing opt-in million-record WAL test. Ruff lint, formatting of all changed Python
files, and whitespace checks passed. Both the wheel and source distribution built.

Unit tests replace Jev and Anthropic at their client boundaries. Grader tests cover
model pinning, response validation, token-limit retry, concurrency/rate limits,
budget reservations, lifecycle, state extraction, citations, historical retries,
exact-version replay copying, late spans, and publication interruption. Stack tests
exercise real child processes and worker ownership. Shutdown tests inject a filesystem
failure during publication and verify that cancellation preserves both the lock and
the original error until publication terminates.
An SDK-boundary regression also verifies that the client and worker lock remain open
until every concurrent request finishes cancellation; the test fails with the former
`gather` implementation and passes with structured task cleanup.

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache src tests
git diff --check
```
