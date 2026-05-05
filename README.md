# Prosa — Code and Data Release

Prosa is a Brazilian Portuguese benchmark of 1,000 real user multi-turn conversations sourced from WildChat. Candidate model responses are scored against per-question binary rubrics (pass/fail) by an LLM judge, with a multi-judge post-hoc filter that removes low-quality rubrics before the final scoring.

This top-level file is a guide; full instructions live in the per-directory `README.md` files.

## Layout

The release is split in two self-contained directories, mirroring the structure of the paper.

| Directory | Paper section | Purpose |
|---|---|---|
| [`Prosa-construction/`](Prosa-construction/README.md) | §3 *Prosa Construction* | Filters WildChat-4.8M down to the 1,000 conversations that form the prompt set of Prosa. |
| [`Prosa-benchmark/`](Prosa-benchmark/README.md) | §4 *Evaluation Protocol*, §5 *Results* | Generates rubrics, scores candidate models, applies the multi-judge filter, produces the leaderboard. |

## Quick paths

### To verify how the 1,000 conversations were curated (§3)

- Pipeline overview and per-step counts: [`Prosa-construction/README.md`](Prosa-construction/README.md)
- Step-by-step scripts: `Prosa-construction/filter_step1_base.py` … `filter_step7_sample_1000.py`
- Frozen output of step 6 (before random sampling, 1,355 rows): `Prosa-construction/06_nonsensical_filtered/`
- Frozen output of step 7 (final sample, 1,000 rows): `Prosa-construction/07_sample_1000/`

### To verify how rubrics were generated and used (§4)

- Rubric generation, RRD F.1 prompt (§4.1): `Prosa-benchmark/prosa/gen_rubrics.py`
- Scoring formula and judge prompt, RRD F.3 (§4.2): `Prosa-benchmark/prosa/gen_score.py`
- Multi-judge rubric filter (§4.3): `Prosa-benchmark/prosa/filter_rubrics.py`
- Frozen 1,000 questions + 12,920 raw rubrics + filter outcomes: `Prosa-benchmark/prosa/data/prosa/question.jsonl`

### To verify the empirical results (§5)

- Holistic baseline scoring (§5.1): `Prosa-benchmark/prosa/gen_score_holistic.py`
- Final leaderboard script (Table 5, §5.2): `Prosa-benchmark/prosa/show_score.py`
- Frozen verdicts for the 16 candidates × 3 judges: `Prosa-benchmark/prosa/data/prosa/score_rubric/`, `score_holistic/`
- Three independent runs used for the *unstable* filter (§4.3): `Prosa-benchmark/prosa/data/prosa/score_rubric/`, `score_rubric_run2/`, `score_rubric_run3/`

## Reproducing the paper's leaderboard without API calls

All artefacts are frozen. Reproducing Table 5 from local data takes two commands:

```bash
cd Prosa-benchmark
pip install -r requirements.txt
python -m prosa.show_score --filtered    # rubric leaderboard (Table 5)
python -m prosa.show_score_holistic      # holistic baseline (§5.1)
```

## Evaluating a new candidate

A new candidate can be scored end-to-end with a single judge (Gemini 3 Flash recommended for cost). See [`Prosa-benchmark/README.md`](Prosa-benchmark/README.md) § *Evaluating a new candidate* for the three-step recipe (`gen_answer` → `gen_score` → `show_score --filtered`).

## Re-running the rubric filter

The filter outputs in `question.jsonl` (`checklist_filtered_indices`, `question_excluded`) come from `prosa.filter_rubrics`. Re-running the filter with the canonical configuration described in §4.3 reproduces them. See [`Prosa-benchmark/README.md`](Prosa-benchmark/README.md) § *Re-running the rubric filter* for usage and flags.
