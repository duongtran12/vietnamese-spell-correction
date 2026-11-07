import bz2
from pathlib import Path

from src.data.io import read_clean_jsonl
from src.data.wikipedia import (
    WikipediaConversionConfig,
    convert_wikipedia_dump,
    iter_wikipedia_pages,
    split_plaintext_sentences,
    strip_wikitext,
)


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Trí tuệ nhân tạo</title><ns>0</ns><id>10</id>
    <revision><id>100</id><text xml:space="preserve">'''Trí tuệ nhân tạo''' là một lĩnh vực khoa học máy tính. {{Chú thích}} Nó nghiên cứu các hệ thống thông minh.</text></revision>
  </page>
  <page>
    <title>Trang chuyển hướng</title><ns>0</ns><id>11</id><redirect title="Đích" />
    <revision><id>101</id><text>#REDIRECT [[Đích]]</text></revision>
  </page>
  <page>
    <title>Thảo luận:Ví dụ</title><ns>1</ns><id>12</id>
    <revision><id>102</id><text>Nội dung thảo luận không được lấy.</text></revision>
  </page>
</mediawiki>
"""


def test_wikipedia_dump_conversion_streams_main_articles(tmp_path: Path) -> None:
    dump_path = tmp_path / "viwiki.xml.bz2"
    with bz2.open(dump_path, "wt", encoding="utf-8") as handle:
        handle.write(SAMPLE_XML)
    output_path = tmp_path / "sentences.jsonl"

    report = convert_wikipedia_dump(
        dump_path,
        output_path,
        source_url="https://dumps.wikimedia.org/viwiki/test.xml.bz2",
        dump_date="20260101",
        config=WikipediaConversionConfig(minimum_tokens=3),
    )
    records = read_clean_jsonl(output_path)

    assert report["pages_scanned"] == 3
    assert report["main_namespace_pages"] == 1
    assert report["redirects_skipped"] == 1
    assert report["sentences_written"] == 2
    assert len(records) == 2
    assert records[0].document_id == "viwiki-10"
    assert records[0].document_title == "Trí tuệ nhân tạo"
    assert "Chú thích" not in records[0].text
    assert len(report["input_sha256"]) == 64


def test_wikipedia_helpers_handle_markup_and_boundaries(tmp_path: Path) -> None:
    xml_path = tmp_path / "viwiki.xml"
    xml_path.write_text(SAMPLE_XML, encoding="utf-8")

    pages = list(iter_wikipedia_pages(xml_path))
    plain = strip_wikitext("'''Việt Nam''' có [[Hà Nội]]. <ref>Nguồn</ref>")

    assert len(pages) == 3
    assert pages[1].is_redirect is True
    assert plain == "Việt Nam có Hà Nội."
    assert split_plaintext_sentences("Câu thứ nhất. Câu thứ hai!") == [
        "Câu thứ nhất.",
        "Câu thứ hai!",
    ]

