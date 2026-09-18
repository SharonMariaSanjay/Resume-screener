# AI Resume Screener

A tool that scores a batch of resumes against a job description using NLP
(TF-IDF + cosine similarity) and keyword analysis, and returns a ranked
shortlist with matched skills, gaps, and a fit summary for each candidate.

## Why I built this

HR teams spend a lot of time on first-pass resume screening. This project
tests whether a simple, explainable NLP approach can do a consistent first
pass — ranking candidates against a JD and showing *why*, not just producing
a black-box score. It's scoped around a real HR Operations Executive job
description.

## How it works

1. Load a job description (`sample_data/job_description.txt`) and a folder
   of resumes (`sample_data/resumes/*.txt`).
2. Convert the JD and every resume into TF-IDF vectors (a standard way of
   representing text based on word importance) and compute cosine similarity
   between each resume and the JD — this gives the 0-100 match score.
3. Separately, scan for a curated list of role-relevant keywords (HRIS, SQL,
   wellbeing, recruitment, etc.) to produce a human-readable "matched
   skills" / "gaps" breakdown.
4. Rank all candidates by score, print a table, and save full results to
   `results.json`.

## Setup

```bash
pip install -r requirements.txt
python screen.py
```

## Sample data

`sample_data/` includes one job description and 5 synthetic resumes with
deliberately varying fit (strong / moderate / weak) so the ranking behavior
is easy to sanity-check. No real candidate data is used.

## Actual sample output

```
resume_4_divya  —  Score: 31.7/100
  Summary: Partial overlap — some relevant experience, several gaps.
  Matched: ai tools, engagement, hris, psychology, recruitment,
           recruitment coordination, sop, sql, wellbeing, wellbeing program
  Gaps:    data, performance management, process documentation,
           sentiment analysis, written communication

resume_1_priya  —  Score: 26.7/100
  Summary: Partial overlap — some relevant experience, several gaps.
  Matched: data, engagement, hris, psychology, recruitment, sop,
           wellbeing, written communication

resume_5_neha  —  Score: 25.6/100
  Summary: Partial overlap — some relevant experience, several gaps.
  Matched: data, hris, performance management, psychology

resume_2_arjun  —  Score: 10.2/100
  Summary: Low overlap — likely not a fit for this role.
  Matched: recruitment

resume_3_karan  —  Score: 7.3/100
  Summary: Low overlap — likely not a fit for this role.
  Matched: None
```

## Design Decisions 

- **TF-IDF + cosine similarity** instead of a paid LLM API — keeps the tool
  free and fully offline, and the scoring logic is transparent and
  explainable rather than a black box, which matters for a hiring-adjacent
  use case.
- **Separate keyword layer for matched skills/gaps** — TF-IDF alone gives a
  similarity number but no interpretable "why." Layering a curated keyword
  list on top makes the output actionable for a recruiter.
- **Known limitation (and a good thing to raise proactively):** plain
  keyword matching can't detect negation — e.g. "no SQL experience" would
  still match "SQL" as present. I caught this in testing and rewrote the
  sample resumes to avoid it, but a production version would need smarter
  phrase-level or embedding-based matching to handle this properly.
- **Synthetic sample data** — lets anyone run this end-to-end without real
  candidate data, and makes the ranking logic easy to verify.
