# tests/test_eval.py
"""Evaluation harness for the RAG pipeline.

Runs a labeled test set through ask() and produces a pass/fail report.
Run with: python -m tests.test_eval
"""
import sys

from src.rag import ask


# --- Test set ---

TEST_CASES = [
    {
        "id": 1,
        "question": "what is the deductible for collision?",
        "type": "on_topic",
        "expected_blocked": False,
        "expected_refusal_reason": None, 
        "expected_phrases": [["500", "1000", "250"]],
    },
    {
        "id": 2,
        "question": "meaning of life",
        "type": "off_topic",
        "expected_blocked": True,
        "expected_refusal_reason": "threshold", 
        "expected_phrases": [],
    },
    {
        "id": 3,
        "question": "policy number for life insurance",
        "type": "adversarial",
        "expected_blocked": True,               
        "expected_refusal_reason": "prompt",      
        "expected_phrases": [
            ["life insurance", "life"],
            ["don't have", "do not have", "no information"],
        ],
    },
    {
        "id": 4,
        "question": "can I cancel my policy?",
        "type": "on_topic",
        "expected_blocked": False,
        "expected_refusal_reason": None, 
        "expected_phrases": [["cancel", "cancellation"],["refund", "policy release"]],
    },
    {
        "id": 5,
        "question": "how do i pay my premium",
        "type": "on_topic",
        "expected_blocked": False,
        "expected_refusal_reason": None,
        "expected_phrases": [["check", "installment", "eft", "credit"]],
    },
    {
        "id": 6,
        "question": "what does deductible mean?",
        "type": "on_topic",
        "expected_blocked": False,
        "expected_refusal_reason": None,
        "expected_phrases": [["amount", "before", "pays"]],
    },
    {
        "id": 7,
        "question": "how do i increase my deductible",
        "type": "corpus_gap",
        "expected_blocked": "any",
        "expected_refusal_reason": "any",
        "expected_phrases": [["don't have", "do not have", "no information", "not in"]],
    },
    {
        "id": 8,
        "question": "how do i cook brussel sprouts",
        "type": "off_topic",
        "expected_blocked": True,
        "expected_refusal_reason": "threshold",
        "expected_phrases": [],
    },
    {
        "id": 9,
        "question": "how do you say hello in another language",
        "type": "off_topic",
        "expected_blocked": True,
        "expected_refusal_reason": "threshold",
        "expected_phrases": [],
    },
    {
        "id": 10,
        "question": "find whether i have permissive use in my auto policy",
        "type": "on_topic",
        "expected_blocked": False,
        "expected_refusal_reason": None,
        "expected_phrases": [["permissive use", "reasonable belief", "entitled"]],
    },
    {
        "id": 11,
        "question": "what is my life insurance policy number",
        "type": "adversarial",
        "expected_blocked": "any",
        "expected_refusal_reason": "any",
        "expected_phrases": [["don't have", "do not have", "no information", "not in"]],
    },
]


# --- Checker ---

def evaluate_case(case: dict, threshold=None) -> dict:
    """Run one test case and return its result."""
    result = ask(case["question"], threshold=threshold)
    answer = result["answer"]

    blocked_match = (
        case["expected_blocked"] == "any"
        or result["blocked"] == case["expected_blocked"]
    )
    reason_match = (
        case["expected_refusal_reason"] == "any"
        or result["refusal_reason"] == case["expected_refusal_reason"]
    )
    # For each phrase group, at least ONE variant must appear in the answer
    groups_matched = [
        any(variant.lower() in answer.lower() for variant in group)
        for group in case["expected_phrases"]
    ]
    phrases_match = all(groups_matched)

    passed = blocked_match and reason_match and phrases_match

    return {
        "id": case["id"],
        "question": case["question"],
        "type": case["type"],
        "passed": passed,
        "blocked_match": blocked_match,
        "reason_match": reason_match,
        "phrases_match": phrases_match,
        "groups_matched": groups_matched,
        "top_score": result["top_score"],
        "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
        "blocked": result["blocked"],
        "expected_blocked": case["expected_blocked"],
        "refusal_reason": result["refusal_reason"],
    }

# --- Reporter ---

def run_eval():
    print("\n" + "=" * 60)
    print("RAG EVALUATION HARNESS")
    print("=" * 60 + "\n")

    results = [evaluate_case(case) for case in TEST_CASES]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n{status} | Case {r['id']} ({r['type']})")
        print(f"   Q: {r['question']}")
        if not r["passed"]:
            print(f"   blocked_match: {r['blocked_match']}")
            print(f"   reason_match: {r['reason_match']}")    
            print(f"   phrases_match: {r['phrases_match']}")
            print(f"   groups_matched: {r['groups_matched']}")
            print(f"   top_score: {r['top_score']}")
            print(f"   answer: {r['answer_preview']}")
            print(f"   [debug] blocked={r['blocked']} reason={r.get('refusal_reason')} score={r['top_score']:.3f}")

    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} passed ({100 * passed // total}%)")
    print("=" * 60 + "\n")

    return passed, total

if __name__ == "__main__":
    passed, total = run_eval()
    sys.exit(0 if passed == total else 1)   # fail the job unless every case passes