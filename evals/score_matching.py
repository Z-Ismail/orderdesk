"""Score catalogue matching against sample_data/golden/expected_matches.csv.

    uv run python evals/score_matching.py                 # real embedding model
    uv run python evals/score_matching.py --embedder hashing   # offline stand-in

Uses the GOLDEN line descriptions, not live extraction output, so we measure matching on its
own. If extraction is wrong too, that's extraction's eval's job to catch.
"""

import argparse
import csv
import sys
from pathlib import Path

from orderdesk.config import get_settings
from orderdesk.customers import identify_customer
from orderdesk.db import connect, migrate
from orderdesk.embeddings import FastEmbedEmbedder, HashingEmbedder
from orderdesk.matching import match_line
from orderdesk.models import ExtractedOrder

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "sample_data" / "golden"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--recall5-threshold", type=float, default=0.98)
    args = parser.parse_args()

    settings = get_settings()
    embedder = (
        HashingEmbedder()
        if args.embedder == "hashing"
        else FastEmbedEmbedder(settings.embedding_model)
    )
    with (GOLDEN / "expected_matches.csv").open() as handle:
        expected = {
            (r["document"], int(r["line_number"])): r["expected_sku"]
            for r in csv.DictReader(handle)
        }

    top1 = recall5 = total = 0
    with connect(settings.database_url) as conn:
        migrate(conn)
        for path in sorted(GOLDEN.glob("*.json")):
            order = ExtractedOrder.model_validate_json(path.read_text())
            customer = identify_customer(conn, order.customer_email)
            for order_line in order.lines:
                want = expected[(path.stem, order_line.line_number)]
                result = match_line(
                    conn, embedder, order_line, customer.customer_id if customer else None
                )
                skus = [c.sku for c in result.candidates]
                total += 1
                top1 += skus[:1] == [want]
                recall5 += want in skus[:5]
                if skus[:1] != [want]:
                    rank = skus.index(want) + 1 if want in skus else "not in top 5"
                    print(
                        f"  x {path.stem} line {order_line.line_number}: "
                        f"'{order_line.description}' -> {skus[:3]}  (want {want}, rank {rank})"
                    )

    t1, r5 = top1 / total, recall5 / total
    ok = t1 >= args.top1_threshold and r5 >= args.recall5_threshold
    print(f"\nTOP-1 ACCURACY: {t1:.1%} ({top1}/{total})   threshold {args.top1_threshold:.0%}")
    print(f"RECALL@5:       {r5:.1%} ({recall5}/{total})   threshold {args.recall5_threshold:.0%}")
    print(f"-> {'PASS' if ok else 'FAIL'}  (embedder: {args.embedder})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
