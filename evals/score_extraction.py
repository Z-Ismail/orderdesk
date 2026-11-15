"""Score extraction outputs against the golden answers.

Usage:  uv run python evals/score_extraction.py [--outputs scratch/out] [--threshold 0.95]

Both sides go through ExtractedOrder first, so they get the same clean-up rules
(units normalised, 'None' -> None). Then each field is compared with the rule that
matches what "correct" means for that field.
"""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from orderdesk.models import ExtractedOrder, OrderLine

ROOT = Path(__file__).resolve().parents[1]


def _words(value: object) -> str:
    """Lower-case, punctuation removed, single spaces: 'Gloves, Large' == 'gloves large'."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(value).lower()).split())


def exact(expected: object, got: object) -> bool:
    return expected == got


def same_words(expected: object, got: object) -> bool:
    if expected is None or got is None:
        return expected is got
    return _words(expected) == _words(got)


def contains(expected: object, got: object) -> bool:
    """Right if the answer contains the expected text (an extra company name is fine)."""
    if expected is None or got is None:
        return expected is got
    return _words(expected) in _words(got)


def presence(expected: object, got: object) -> bool:
    """Free text: only check that it's there when it should be, and absent when it shouldn't."""
    return (expected is None) == (got is None)


# The definition of "correct", field by field. This table IS the evaluation design.
HEADER_RULES: dict[str, Callable[[object, object], bool]] = {
    "customer_name": same_words,
    "customer_email": exact,
    "customer_po_number": exact,
    "order_date": exact,
    "requested_delivery_date": exact,
    "delivery_address": contains,
    "currency": exact,
    "notes": presence,
    "is_amendment": exact,
}
LINE_RULES: dict[str, Callable[[object, object], bool]] = {
    "customer_sku": exact,
    "description": same_words,
    "quantity": exact,
    "unit_of_measure": exact,
    "unit_price": exact,
}


def score_document(gold: ExtractedOrder, got: ExtractedOrder) -> tuple[int, int, list[str]]:
    correct, total, failures = 0, 0, []
    for field, rule in HEADER_RULES.items():
        total += 1
        e, g = getattr(gold, field), getattr(got, field)
        if rule(e, g):
            correct += 1
        else:
            failures.append(f"{field}: expected {e!r}, got {g!r}")

    total += 1
    if len(gold.lines) == len(got.lines):
        correct += 1
    else:
        failures.append(f"line count: expected {len(gold.lines)}, got {len(got.lines)}")

    got_by_number: dict[int, OrderLine] = {line.line_number: line for line in got.lines}
    for gline in gold.lines:
        oline = got_by_number.get(gline.line_number)
        for field, rule in LINE_RULES.items():
            total += 1
            e = getattr(gline, field)
            g = getattr(oline, field) if oline else "<missing line>"
            if rule(e, g):
                correct += 1
            else:
                failures.append(f"line {gline.line_number} {field}: expected {e!r}, got {g!r}")
    return correct, total, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "scratch" / "out")
    parser.add_argument("--golden", type=Path, default=ROOT / "sample_data" / "golden")
    parser.add_argument("--threshold", type=float, default=0.95)
    # Documents we expect to have no text output yet (scans wait for the vision model).
    parser.add_argument("--allow-missing", nargs="*", default=[])
    args = parser.parse_args()

    grand_correct = grand_total = 0
    for gold_path in sorted(args.golden.glob("*.json")):
        gold = ExtractedOrder.model_validate_json(gold_path.read_text())
        out_path = args.outputs / gold_path.name
        if not out_path.exists():
            if gold_path.stem in args.allow_missing:
                print(f"    --  {gold_path.stem:<40} no output (allowed to be missing)")
                continue
            # A missing result counts as every field wrong: silence is not success.
            missing_total = len(HEADER_RULES) + 1 + len(gold.lines) * len(LINE_RULES)
            grand_total += missing_total
            print(f"  0.0%  {gold_path.stem:<40} 0/{missing_total}  MISSING OUTPUT")
            continue
        got = ExtractedOrder.model_validate_json(out_path.read_text())
        correct, total, failures = score_document(gold, got)
        grand_correct += correct
        grand_total += total
        print(f"{correct / total:6.1%}  {gold_path.stem:<40} {correct}/{total}")
        for failure in failures:
            print(f"          x {failure}")

    accuracy = grand_correct / grand_total if grand_total else 0.0
    print(
        f"\nFIELD ACCURACY: {accuracy:.1%}  ({grand_correct}/{grand_total})  "
        f"threshold {args.threshold:.0%}  -> {'PASS' if accuracy >= args.threshold else 'FAIL'}"
    )
    return 0 if accuracy >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
