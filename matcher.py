import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """
You are an expert recruiting analyst.
Assess how well a resume matches a job description.
Return valid JSON only.
""".strip()


USER_PROMPT_TEMPLATE = """
Score this resume against this job description.

Return JSON with exactly this schema:
{
  "overall_score": <int 0-100>,
  "category_scores": {
    "skills": <int 0-100>,
    "experience": <int 0-100>,
    "domain_fit": <int 0-100>,
    "impact": <int 0-100>
  },
  "strengths": ["..."],
  "gaps": ["..."],
  "missing_keywords": ["..."],
  "suggestions": [
    {
      "priority": "high|medium|low",
      "action": "...",
      "example_resume_line": "..."
    }
  ]
}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
""".strip()

REQUIRED_CATEGORIES = ("skills", "experience", "domain_fit", "impact")
VALID_PRIORITIES = {"high", "medium", "low"}


@dataclass
class MatchResult:
    overall_score: int
    category_scores: dict[str, int]
    strengths: list[str]
    gaps: list[str]
    missing_keywords: list[str]
    suggestions: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume ↔ JD matcher using an LLM")
    parser.add_argument("--resume-file", required=True, help="Path to resume text file")
    parser.add_argument("--jd-file", required=True, help="Path to JD text file")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--raw", action="store_true", help="Print parsed JSON output")
    parser.add_argument("--output-file", help="Optional path to write parsed JSON")
    return parser.parse_args()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    if not content:
        raise ValueError(f"Input file is empty: {path}")
    return content


def _validate_score(name: str, value: Any) -> int:
    if not isinstance(value, int) or not (0 <= value <= 100):
        raise ValueError(f"{name} must be an integer in [0, 100]. Got: {value!r}")
    return value


def _validate_result_shape(parsed: dict[str, Any]) -> MatchResult:
    overall_score = _validate_score("overall_score", parsed.get("overall_score"))

    category_scores = parsed.get("category_scores")
    if not isinstance(category_scores, dict):
        raise ValueError("category_scores must be an object")

    normalized_category_scores: dict[str, int] = {}
    for category in REQUIRED_CATEGORIES:
        normalized_category_scores[category] = _validate_score(
            f"category_scores.{category}", category_scores.get(category)
        )

    def ensure_string_list(name: str) -> list[str]:
        value = parsed.get(name)
        if not isinstance(value, list) or not all(isinstance(i, str) and i.strip() for i in value):
            raise ValueError(f"{name} must be a non-empty string list")
        return value

    strengths = ensure_string_list("strengths")
    gaps = ensure_string_list("gaps")
    missing_keywords = ensure_string_list("missing_keywords")

    suggestions_raw = parsed.get("suggestions")
    if not isinstance(suggestions_raw, list):
        raise ValueError("suggestions must be a list")

    suggestions: list[dict[str, str]] = []
    for idx, suggestion in enumerate(suggestions_raw):
        if not isinstance(suggestion, dict):
            raise ValueError(f"suggestions[{idx}] must be an object")
        priority = suggestion.get("priority")
        action = suggestion.get("action")
        example_line = suggestion.get("example_resume_line")
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"suggestions[{idx}].priority must be one of {sorted(VALID_PRIORITIES)}")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"suggestions[{idx}].action must be a non-empty string")
        if not isinstance(example_line, str) or not example_line.strip():
            raise ValueError(f"suggestions[{idx}].example_resume_line must be a non-empty string")
        suggestions.append(
            {
                "priority": priority,
                "action": action,
                "example_resume_line": example_line,
            }
        )

    return MatchResult(
        overall_score=overall_score,
        category_scores=normalized_category_scores,
        strengths=strengths,
        gaps=gaps,
        missing_keywords=missing_keywords,
        suggestions=suggestions,
    )


def run_match(resume: str, jd: str, model: str) -> MatchResult:
    client = OpenAI()
    prompt = USER_PROMPT_TEMPLATE.format(resume=resume, jd=jd)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "json_object"}},
    )

    text_output = response.output_text.strip()
    try:
        parsed = json.loads(text_output)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model returned non-JSON output: {text_output[:200]}") from error

    return _validate_result_shape(parsed)


def print_result(result: MatchResult) -> None:
    print(f"Overall match score: {result.overall_score}/100")
    print("\nCategory scores:")
    for key, value in result.category_scores.items():
        print(f"- {key}: {value}")

    print("\nStrengths:")
    for item in result.strengths:
        print(f"- {item}")

    print("\nGaps:")
    for item in result.gaps:
        print(f"- {item}")

    print("\nMissing keywords:")
    for item in result.missing_keywords:
        print(f"- {item}")

    print("\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"- [{suggestion['priority']}] {suggestion['action']}")
        print(f"  Example: {suggestion['example_resume_line']}")


def main() -> None:
    args = parse_args()
    try:
        resume = read_text(args.resume_file)
        jd = read_text(args.jd_file)
        result = run_match(resume=resume, jd=jd, model=args.model)
    except Exception as error:
        raise SystemExit(f"Error: {error}") from error

    result_json = json.dumps(asdict(result), indent=2)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as file:
            file.write(result_json + "\n")

    if args.raw:
        print(result_json)
    else:
        print_result(result)


if __name__ == "__main__":
    main()
