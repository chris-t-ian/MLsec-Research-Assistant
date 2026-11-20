"""LaTeX parsing: main file detection, include resolution, structure, conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.ingest.latex_parse import (
    LatexParseError,
    find_main_tex,
    flatten,
    latex_to_text,
    parse_latex,
)

FIXTURES = Path(__file__).parent / "fixtures" / "latex"
CLEAN = FIXTURES / "clean"


def test_find_main_tex_picks_documentclass_file() -> None:
    assert find_main_tex(CLEAN) == CLEAN / "main.tex"


def test_find_main_tex_none_without_documentclass(tmp_path: Path) -> None:
    (tmp_path / "a.tex").write_text("\\section{X} text")
    (tmp_path / "b.tex").write_text("% \\documentclass{article}\n")  # commented out
    assert find_main_tex(tmp_path) is None
    with pytest.raises(LatexParseError):
        parse_latex(tmp_path, "0000.00000")


def test_find_main_tex_prefers_file_with_begin_document(tmp_path: Path) -> None:
    (tmp_path / "standalone.tex").write_text("\\documentclass{standalone}\n")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}x\\end{document}"
    )
    assert find_main_tex(tmp_path) == tmp_path / "paper.tex"


def test_flatten_resolves_input_and_maps_locations() -> None:
    flat = flatten(CLEAN / "main.tex", CLEAN)
    assert "Membership inference asks whether" in flat.text
    assert "\\input{sections/intro}" not in flat.text
    assert "does_not_exist" in flat.text  # commented-out \input is left alone
    offset = flat.text.index("Membership inference asks")
    assert flat.location(offset) == "sections/intro.tex:3"
    assert flat.location(flat.text.index("\\section{Method}")).startswith("main.tex:")


def test_flatten_refuses_includes_outside_source_dir(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "secret.tex").write_text("SECRET CONTENT")
    (src / "main.tex").write_text("\\documentclass{article}\n\\input{../secret}\n")
    flat = flatten(src / "main.tex", src)
    assert "SECRET CONTENT" not in flat.text


def test_flatten_survives_include_cycle(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text("\\documentclass{article}\nA \\input{b}")
    (tmp_path / "b.tex").write_text("B \\input{main}")
    flat = flatten(tmp_path / "main.tex", tmp_path)
    assert flat.text.count("B ") == 1


def test_parse_clean_structure() -> None:
    doc = parse_latex(CLEAN, "1234.56789").document
    titles = [s.title for s in doc.sections]
    kinds = [s.kind for s in doc.sections]
    assert titles == [
        "Abstract",
        "Introduction",
        "Method",
        "Method > Gradient Clipping",
        "Appendix: Proofs",
    ]
    assert kinds == ["abstract", "section", "section", "subsection", "appendix"]
    assert doc.source == "latex" and doc.main_file == "main.tex"


def test_parse_clean_content() -> None:
    doc = parse_latex(CLEAN, "1234.56789").document
    by_title = {s.title: s.text for s in doc.sections}
    # Math stays as inline LaTeX; simple macros are expanded.
    assert "$f_\\theta$" in by_title["Abstract"]
    assert "PrivGuard trains with DP-SGD where $\\epsilon = 8$" in by_title["Method"]
    # Comments, bibliography and title/author metadata are dropped.
    full = doc.full_text()
    assert "note to self" not in full
    assert "refs" not in full and "plain" not in full
    assert "Alice Example" not in full
    # Source line breaks inside a paragraph are unwrapped.
    assert "leak information about their training data. Membership" in by_title["Introduction"]


def test_latex_to_text_handles_href() -> None:
    assert latex_to_text("See \\href{https://example.org/x}{the code}.") == "See the code."


def test_flatten_resolves_relative_to_including_file(tmp_path: Path) -> None:
    (tmp_path / "fig").mkdir()
    (tmp_path / "main.tex").write_text("\\documentclass{article}\n\\input{fig/a}")
    (tmp_path / "fig" / "a.tex").write_text("A \\input{../defs}")  # escapes via root, not via fig/
    (tmp_path / "defs.tex").write_text("DEFS")
    assert "A DEFS" in flatten(tmp_path / "main.tex", tmp_path).text


def test_included_standalone_document_does_not_end_paper(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\section{One}\nFirst.\n"
        "\\input{fig}\n\\section{Two}\nSecond.\n\\end{document}\n"
    )
    (tmp_path / "fig.tex").write_text(
        "\\documentclass{standalone}\n\\begin{document}\nFIGURE\n\\end{document}\n"
    )
    doc = parse_latex(tmp_path, "x").document
    assert [s.title for s in doc.sections] == ["One", "Two"]
    assert "FIGURE" in doc.sections[0].text and "standalone" not in doc.full_text()


def test_template_heading_aliases(tmp_path: Path) -> None:
    (tmp_path / "conf.sty").write_text(
        "\\def\\confsection{\\@startsection {section}{1}{\\z@}{}{}{}}\n"
        "\\newcommand{\\mysub}[1]{\\subsection{#1}}\n"
    )
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\usepackage{conf}\n\\begin{document}\n"
        "\\confsection{Intro}\nText.\n\\mysub{Detail}\nMore.\n\\end{document}\n"
    )
    doc = parse_latex(tmp_path, "x").document
    assert [(s.title, s.kind) for s in doc.sections] == [
        ("Intro", "section"),
        ("Intro > Detail", "subsection"),
    ]


def test_comment_environment_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\section{S}\nKept.\n"
        "\\begin{comment}\nDropped draft text.\n\\end{comment}\n\\end{document}\n"
    )
    doc = parse_latex(tmp_path, "x").document
    assert doc.full_text() == "Kept."
