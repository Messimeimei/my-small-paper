#!/usr/bin/env python3
"""Validate the basic properties of an anonymous ACL draft PDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from pypdf import PdfReader


A4_WIDTH_PT = 595.276
A4_HEIGHT_PT = 841.890
PAGE_TOLERANCE_PT = 2.0
PAPER_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PAPER_WORKSPACE_ROOT.parent
POPPLER_ROOT = PROJECT_ROOT / ".tools" / "poppler" / "usr"
PDFFONTS = POPPLER_ROOT / "bin" / "pdffonts"
POPPLER_LIB = POPPLER_ROOT / "lib" / "x86_64-linux-gnu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    return parser.parse_args()


def find_unembedded_fonts(pdf: Path) -> tuple[int, list[str]]:
    if not PDFFONTS.is_file():
        raise RuntimeError(f"pdffonts is unavailable: {PDFFONTS}")

    environment = os.environ.copy()
    existing_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(POPPLER_LIB)
    if existing_library_path:
        environment["LD_LIBRARY_PATH"] += f":{existing_library_path}"

    result = subprocess.run(
        [str(PDFFONTS), str(pdf.resolve())],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    font_rows = [
        line
        for line in result.stdout.splitlines()[2:]
        if line.strip() and not line.startswith("-")
    ]
    unembedded: list[str] = []
    for row in font_rows:
        fields = row.split()
        if len(fields) < 6:
            raise RuntimeError(f"could not parse pdffonts row: {row}")
        if fields[-5].lower() != "yes":
            unembedded.append(fields[0])
    return len(font_rows), unembedded


def main() -> int:
    args = parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF does not exist: {args.pdf}")

    reader = PdfReader(args.pdf)
    failures: list[str] = []
    if not reader.pages:
        failures.append("PDF has no pages")

    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        portrait_a4 = (
            abs(width - A4_WIDTH_PT) <= PAGE_TOLERANCE_PT
            and abs(height - A4_HEIGHT_PT) <= PAGE_TOLERANCE_PT
        )
        if not portrait_a4:
            failures.append(
                f"page {index} is {width:.2f} x {height:.2f} pt, expected A4"
            )

    metadata = reader.metadata or {}
    author = str(metadata.get("/Author", "")).strip()
    if author and author.lower() not in {"anonymous", "anonymous acl submission"}:
        failures.append(f"non-anonymous PDF Author metadata: {author!r}")

    try:
        font_count, unembedded_fonts = find_unembedded_fonts(args.pdf)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        failures.append(f"font embedding check failed: {error}")
        font_count = 0
        unembedded_fonts = []
    if unembedded_fonts:
        failures.append(f"unembedded fonts: {', '.join(unembedded_fonts)}")

    print(f"pages={len(reader.pages)}")
    print(f"author_metadata={author or '<empty>'}")
    print(f"fonts_checked={font_count}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("PASS: PDF is nonempty, A4, anonymous, and has embedded fonts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
