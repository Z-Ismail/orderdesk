"""Show similarity and margin for every golden line, so you can choose confidence thresholds
from evidence instead of guessing.

    uv run python evals/match_confidence_report.py

Good thresholds pass (almost) every correct match and catch every wrong one. Wrong matches
that sail through are the expensive mistake; a few extra reviews are the cheap one.
"""

import csv
from pathlib import Path

from orderdesk.config import get_settings
from orderdesk.customers import identify_customer
from orderdesk.db import connect, migrate
from orderdesk.embeddings import FastEmbedEmbedder
from orderdesk.matching import match_line
from orderdesk.models import ExtractedOrder
from orderdesk.pipeline import Thresholds, is_confident

GOLDEN = Path(__file__).resolve().parents[1] / "sample_data" / "golden"


def main() -> None:
    settings = get_settings()
    thresholds = Thresholds(settings.match_min_similarity, settings.match_min_margin)
    embedder = FastEmbedEmbedder(settings.embedding_model)
    with (GOLDEN / "expected_matches.csv").open() as handle:
        expected = {
            (r["document"], int(r["line_number"])): r["expected_sku"]
            for r in csv.DictReader(handle)
        }

    print(f"{'source':<10}{'sim':>6}{'margin':>8}  {'ok?':<5}{'trusted?':<10} description")
    wrong_but_trusted = right_but_flagged = 0
    with connect(settings.database_url) as conn:
        migrate(conn)
        for path in sorted(GOLDEN.glob("*.json")):
            order = ExtractedOrder.model_validate_json(path.read_text())
            customer = identify_customer(conn, order.customer_email)
            for line in order.lines:
                match = match_line(conn, embedder, line, customer.customer_id if customer else None)
                best = match.best
                correct = best is not None and best.sku == expected[(path.stem, line.line_number)]
                trusted = is_confident(match, thresholds)
                wrong_but_trusted += (not correct) and trusted
                right_but_flagged += correct and not trusted
                sim = f"{best.similarity:.3f}" if best and best.similarity is not None else "  -"
                margin = f"{match.margin:.3f}" if match.margin is not None else "   -"
                print(
                    f"{best.source if best else '-':<10}{sim:>6}{margin:>8}  "
                    f"{'yes' if correct else 'NO':<5}{'yes' if trusted else 'review':<10} "
                    f"{line.description}"
                )

    print(
        f"\nThresholds: similarity >= {thresholds.min_similarity}, "
        f"margin >= {thresholds.min_margin}"
    )
    print(f"Wrong matches trusted (dangerous): {wrong_but_trusted}")
    print(f"Right matches sent to review (cost: reviewer time): {right_but_flagged}")


if __name__ == "__main__":
    main()
