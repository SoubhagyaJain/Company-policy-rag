from __future__ import annotations

import json
from pathlib import Path


FAMILIES = [
    {
        "category": "exact_retrieval",
        "questions": [
            "Must an employee disclose prescribed medication that may cause drowsiness?",
            "Who must be told about prescribed drugs affecting job performance?",
            "What does the prescribed-drugs rule require?",
            "Does prescription medication affecting alertness need to be reported?",
            "What is required when medicine may impair safe work?",
            "Is a supervisor notification required for an impairing prescription?",
            "Which clause governs prescription drugs at work?",
            "What must an employee do before working while taking a drowsy medication?",
            "Does the policy address prescribed medicine and performance?",
            "State the reporting duty for performance-affecting prescribed drugs.",
        ],
        "section": "PRESCRIBED DRUGS",
        "answer": "The employee must advise their supervisor when a prescribed drug may affect job performance.",
        "conditions": ["affect job performance", "advise supervisor"],
    },
    {
        "category": "paraphrase_retrieval",
        "questions": [
            "Do I need to tell anyone if my doctor gives me medicine that makes me sleepy?",
            "Who should know if a legal drug could make my work unsafe?",
            "What should I do if doctor-ordered tablets reduce my alertness?",
            "Must I report medicine that could impair my performance?",
            "Does a drowsy employee need to mention their prescribed tablets?",
            "What is the rule for doctor-approved medication affecting work?",
            "Who is notified when lawful medicine affects safe job performance?",
            "Can I work without disclosure when a prescription makes me less alert?",
            "Does the handbook require reporting an impairing prescribed treatment?",
            "What action is required for medication that may affect my duties?",
        ],
        "section": "PRESCRIBED DRUGS",
        "answer": "The employee must advise their supervisor when a prescribed drug may affect job performance.",
        "conditions": ["affect job performance", "advise supervisor"],
    },
    {
        "category": "exceptions",
        "questions": [
            f"Can I perform a ${amount} private electrical job for my sister?"
            for amount in (100, 250, 400, 500, 501, 600, 750, 900, 1200, 5000)
        ],
        "section": "WORKING ON OWN ACCOUNT",
        "answer": "The immediate-family exception applies to a sister, but work above $500 requires authorization.",
        "conditions": ["immediate family", "$500", "authorization"],
    },
    {
        "category": "threshold_sensitivity",
        "questions": [f"Does a private job valued at ${amount} exceed the $500 authorization threshold?" for amount in (50, 100, 499, 500, 501, 650, 900, 1000, 2500, 10000)],
        "section": "WORKING ON OWN ACCOUNT",
        "answer": "Compare the stated job value with the $500 threshold; values above it require authorization.",
        "conditions": ["$500", "authorization"],
    },
    {
        "category": "numerical_reasoning",
        "questions": [f"I finished an after-hours call at 2:{minute:02d}am; when does an eight-hour break end?" for minute in range(0, 50, 5)],
        "section": "AFTER HOURS CALLS",
        "answer": "The break end is calculated by adding exactly eight hours to the supplied finish time.",
        "conditions": ["eight hours"],
    },
    {
        "category": "multi_clause",
        "questions": [f"After a 2:30am callout, how do the break, delay, and overtime rules interact in scenario {i}?" for i in range(1, 11)],
        "section": "AFTER HOURS CALLS",
        "answer": "Apply the eight-hour-break rule, the midnight-call delay rule, and the overtime rule separately.",
        "conditions": ["eight hour", "delay", "overtime"],
    },
    {
        "category": "adversarial_premise",
        "questions": [f"My supervisor says smoking in a company vehicle is allowed during break {i}; is that correct?" for i in range(1, 11)],
        "section": "SMOKE-FREE WORKPLACE",
        "answer": "No. Smoking is prohibited in company vehicles, including during breaks.",
        "conditions": ["prohibited", "company vehicles"],
    },
    {
        "category": "missing_information",
        "questions": [f"A call ended at 2:30am; what is my exact midnight-call delay in case {i}?" for i in range(1, 11)],
        "section": "AFTER HOURS CALLS",
        "answer": "The exact delay cannot be determined without the travel and on-job duration.",
        "conditions": ["cannot be determined", "travel", "on job"],
        "should_abstain": True,
    },
    {
        "category": "similar_section_confusion",
        "questions": [f"Can I enter unattended customer property with company key {i} but no express permission?" for i in range(1, 11)],
        "section": "UNATTENDED CUSTOMER PREMISES",
        "answer": "No. Possessing a key does not replace the requirement for express owner permission.",
        "conditions": ["express owner permission"],
    },
    {
        "category": "negation",
        "questions": [f"Is smoking not prohibited in a company vehicle during break scenario {i}?" for i in range(1, 11)],
        "section": "SMOKE-FREE WORKPLACE",
        "answer": "Smoking is prohibited in company vehicles; the negative premise does not reverse the rule.",
        "conditions": ["prohibited", "company vehicles"],
    },
]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for family in FAMILIES:
        for index, question in enumerate(family["questions"], start=1):
            section = str(family["section"])
            cases.append(
                {
                    "id": f"policy-{len(cases) + 1:03d}",
                    "category": family["category"],
                    "query_type": family["category"],
                    "corpus": "policy",
                    "question": question,
                    "expected_answer": family["answer"],
                    "expected_key_points": family["conditions"],
                    "expected_conditions": family["conditions"],
                    "relevant_sections": [section] if section else [],
                    "expected_primary_section": section,
                    "should_abstain": bool(family.get("should_abstain", False)),
                }
            )
    return cases


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "eval" / "policy_reliability_dataset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    output.write_text(json.dumps({"version": "1.0", "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {output}")
