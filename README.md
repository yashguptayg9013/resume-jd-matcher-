# Resume ↔ Job Description Matcher

A production-minded Python CLI that uses an LLM to:

1. Score how well a resume matches a job description (0–100).
2. Break down scoring by skills, experience, domain fit, and impact.
3. Generate concrete resume improvement suggestions.

## What’s improved

- Enforces **JSON-only** responses from the model.
- Validates all output fields and score ranges.
- Fails with clear errors for bad/empty input files or invalid model output.
- Supports saving parsed JSON output to disk via `--output-file`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your key:

```bash
export OPENAI_API_KEY="your_key_here"
```

## Usage

```bash
python matcher.py \
  --resume-file examples/resume.txt \
  --jd-file examples/jd.txt
```

Print parsed JSON:

```bash
python matcher.py \
  --resume-file examples/resume.txt \
  --jd-file examples/jd.txt \
  --raw
```

Save parsed JSON:

```bash
python matcher.py \
  --resume-file examples/resume.txt \
  --jd-file examples/jd.txt \
  --output-file output.json
```

## Notes

- This is a decision-support tool, not a hiring decision engine.
- For production: add PII scrubbing, eval datasets, and prompt/version observability.
