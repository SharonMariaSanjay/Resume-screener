"""
AI Resume Screener (Free / Local Version)
------------------------------------------
Scores a batch of resumes against a job description using TF-IDF vectors and
cosine similarity — a standard NLP technique. Runs entirely on your machine.
No API key, no signup, no cost.

Usage:
    pip install -r requirements.txt
    python screen.py

Folder layout expected:
    sample_data/job_description.txt
    sample_data/resumes/*.txt
"""

import os
import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JD_PATH = "sample_data/job_description.txt"
RESUME_DIR = "sample_data/resumes"

# Keywords pulled from what matters for this role. Edit this list to match
# whatever job description you're screening against.
SKILL_KEYWORDS = [
    "hris", "recruitment", "recruitment coordination", "wellbeing",
    "wellbeing program", "psychology", "sql", "excel", "performance management",
    "onboarding", "offboarding", "engagement", "employee engagement",
    "written communication", "sop", "process documentation", "ai tools",
    "sentiment analysis", "data",
]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_keywords(text: str) -> set:
    text_lower = text.lower()
    return {kw for kw in SKILL_KEYWORDS if kw in text_lower}


def score_resumes(jd_text: str, resumes: dict) -> list:
    """
    resumes: {name: text}
    Returns a ranked list of dicts with score, matched_skills, gaps, summary.
    """
    names = list(resumes.keys())
    documents = [jd_text] + [resumes[n] for n in names]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    jd_keywords = find_keywords(jd_text)

    results = []
    for i, name in enumerate(names):
        raw_score = similarities[i]
        score = round(float(raw_score) * 100, 1)

        resume_keywords = find_keywords(resumes[name])
        matched = sorted(jd_keywords & resume_keywords)
        gaps = sorted(jd_keywords - resume_keywords)

        if score >= 40:
            summary = "Strong lexical overlap with the job description."
        elif score >= 20:
            summary = "Partial overlap — some relevant experience, several gaps."
        else:
            summary = "Low overlap — likely not a fit for this role."

        results.append({
            "name": name,
            "score": score,
            "matched_skills": matched,
            "gaps": gaps,
            "summary": summary,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def main():
    jd_text = load_text(JD_PATH)

    resume_paths = sorted(glob.glob(os.path.join(RESUME_DIR, "*.txt")))
    if not resume_paths:
        raise SystemExit(f"No resumes found in {RESUME_DIR}")

    resumes = {}
    for path in resume_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        resumes[name] = load_text(path)

    results = score_resumes(jd_text, resumes)

    print("\n" + "=" * 70)
    print("RANKED RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\n{r['name']}  —  Score: {r['score']}/100")
        print(f"  Summary: {r['summary']}")
        print(f"  Matched: {', '.join(r['matched_skills']) or 'None'}")
        print(f"  Gaps:    {', '.join(r['gaps']) or 'None'}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to results.json")


if __name__ == "__main__":
    main()
