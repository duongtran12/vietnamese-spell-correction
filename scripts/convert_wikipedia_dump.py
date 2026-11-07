"""Convert an official Vietnamese Wikipedia XML dump into clean project JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.io import write_json
from src.data.wikipedia import WikipediaConversionConfig, convert_wikipedia_dump


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/interim/viwiki.jsonl"))
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/interim/viwiki_provenance.json"),
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--dump-date", required=True, help="Wikimedia dump date YYYYMMDD")
    parser.add_argument("--minimum-tokens", type=int, default=5)
    parser.add_argument("--maximum-tokens", type=int, default=80)
    parser.add_argument("--minimum-letter-ratio", type=float, default=0.6)
    parser.add_argument("--maximum-pages", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = convert_wikipedia_dump(
        args.input,
        args.output,
        source_url=args.source_url,
        dump_date=args.dump_date,
        config=WikipediaConversionConfig(
            minimum_tokens=args.minimum_tokens,
            maximum_tokens=args.maximum_tokens,
            minimum_letter_ratio=args.minimum_letter_ratio,
            maximum_pages=args.maximum_pages,
        ),
    )
    write_json(args.provenance, report)

    print(f"Pages scanned: {report['pages_scanned']}")
    print(f"Main namespace pages: {report['main_namespace_pages']}")
    print(f"Sentences written: {report['sentences_written']}")
    print(f"Output: {args.output}")
    print(f"Provenance: {args.provenance}")


if __name__ == "__main__":
    main()

