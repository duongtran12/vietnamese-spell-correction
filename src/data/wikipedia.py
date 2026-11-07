"""Streaming conversion of an official Wikimedia XML dump to project JSONL."""

from __future__ import annotations

import bz2
from dataclasses import asdict, dataclass
import hashlib
from html import unescape
from importlib.metadata import version
from pathlib import Path
import re
from typing import BinaryIO, Iterator
import xml.etree.ElementTree as ElementTree

import mwparserfromhell

from src.data.io import write_jsonl
from src.data.normalize import normalize_text, tokenize
from src.data.schema import CleanSentence


WIKITABLE_PATTERN = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)
COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
REMOVABLE_TAGS = {"ref", "references", "gallery", "math", "code", "syntaxhighlight"}
REMOVABLE_LINK_PREFIXES = ("file:", "image:", "category:", "tập tin:", "thể loại:")


@dataclass(frozen=True)
class WikipediaPage:
    page_id: str
    title: str
    namespace: int
    text: str
    is_redirect: bool


@dataclass(frozen=True)
class WikipediaConversionConfig:
    minimum_tokens: int = 5
    maximum_tokens: int = 80
    minimum_letter_ratio: float = 0.6
    maximum_pages: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_tokens < 1 or self.maximum_tokens < self.minimum_tokens:
            raise ValueError("invalid sentence token limits")
        if not 0.0 <= self.minimum_letter_ratio <= 1.0:
            raise ValueError("minimum_letter_ratio must be between 0 and 1")
        if self.maximum_pages is not None and self.maximum_pages < 1:
            raise ValueError("maximum_pages must be positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_dump(path: Path) -> BinaryIO:
    if path.suffix.casefold() == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def iter_wikipedia_pages(path: str | Path) -> Iterator[WikipediaPage]:
    """Yield pages from a standard Wikimedia pages-articles XML or XML.BZ2 dump."""

    dump_path = Path(path)
    with _open_dump(dump_path) as handle:
        for _, element in ElementTree.iterparse(handle, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            title_element = _direct_child(element, "title")
            namespace_element = _direct_child(element, "ns")
            page_id_element = _direct_child(element, "id")
            revision_element = _direct_child(element, "revision")
            text_element = (
                _direct_child(revision_element, "text")
                if revision_element is not None
                else None
            )
            redirect_element = _direct_child(element, "redirect")
            yield WikipediaPage(
                page_id=(page_id_element.text or "").strip() if page_id_element is not None else "",
                title=(title_element.text or "").strip() if title_element is not None else "",
                namespace=int(namespace_element.text or 0) if namespace_element is not None else 0,
                text=text_element.text or "" if text_element is not None else "",
                is_redirect=redirect_element is not None,
            )
            element.clear()


def strip_wikitext(wikitext: str) -> str:
    """Remove templates, media/category links, selected tags, and wiki markup."""

    source = COMMENT_PATTERN.sub(" ", WIKITABLE_PATTERN.sub(" ", wikitext))
    code = mwparserfromhell.parse(source)

    for tag in reversed(code.filter_tags(recursive=True)):
        if str(tag.tag).strip().casefold() in REMOVABLE_TAGS:
            try:
                code.remove(tag, recursive=True)
            except ValueError:
                pass
    for template in reversed(code.filter_templates(recursive=True)):
        try:
            code.remove(template, recursive=True)
        except ValueError:
            pass
    for link in reversed(code.filter_wikilinks(recursive=True)):
        if str(link.title).strip().casefold().startswith(REMOVABLE_LINK_PREFIXES):
            try:
                code.remove(link, recursive=True)
            except ValueError:
                pass

    plain = unescape(code.strip_code(normalize=True, collapse=True))
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in plain.splitlines()]
    return "\n".join(line for line in lines if line)


def split_plaintext_sentences(text: str) -> list[str]:
    """Split paragraphs with a transparent punctuation-based rule."""

    sentences: list[str] = []
    for paragraph in text.splitlines():
        sentences.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?…])\s+", paragraph)
            if part.strip()
        )
    return sentences


def _letter_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return sum(character.isalpha() for character in visible) / len(visible)


def convert_wikipedia_dump(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_url: str,
    dump_date: str,
    config: WikipediaConversionConfig | None = None,
) -> dict[str, object]:
    """Stream main-namespace articles to clean sentence JSONL and return provenance."""

    config = config or WikipediaConversionConfig()
    counters = {
        "pages_scanned": 0,
        "main_namespace_pages": 0,
        "redirects_skipped": 0,
        "empty_pages_after_cleaning": 0,
        "sentences_written": 0,
        "sentences_filtered": 0,
    }

    def records() -> Iterator[dict[str, object]]:
        for page in iter_wikipedia_pages(input_path):
            counters["pages_scanned"] += 1
            if page.namespace != 0:
                continue
            if page.is_redirect:
                counters["redirects_skipped"] += 1
                continue
            if (
                config.maximum_pages is not None
                and counters["main_namespace_pages"] >= config.maximum_pages
            ):
                break
            counters["main_namespace_pages"] += 1

            plain_text = strip_wikitext(page.text)
            if not plain_text:
                counters["empty_pages_after_cleaning"] += 1
                continue
            sentence_number = 0
            for sentence in split_plaintext_sentences(plain_text):
                normalized = normalize_text(sentence)
                token_count = len(tokenize(normalized))
                if not (
                    config.minimum_tokens <= token_count <= config.maximum_tokens
                    and _letter_ratio(normalized) >= config.minimum_letter_ratio
                ):
                    counters["sentences_filtered"] += 1
                    continue
                sentence_number += 1
                counters["sentences_written"] += 1
                yield CleanSentence(
                    document_id=f"viwiki-{page.page_id}",
                    sentence_id=f"viwiki-{page.page_id}-s{sentence_number:05d}",
                    text=normalized,
                    source="viwiki",
                    document_title=page.title,
                ).to_dict()

    write_jsonl(output_path, records())
    input_path = Path(input_path)
    return {
        "source": "Vietnamese Wikipedia",
        "project_identifier": "viwiki",
        "source_url": source_url,
        "dump_date": dump_date,
        "input_file": input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
        "output_file": str(output_path),
        "parser": "mwparserfromhell",
        "parser_version": version("mwparserfromhell"),
        "config": asdict(config),
        **counters,
    }

