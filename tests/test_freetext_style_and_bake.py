"""Issue #104: FreeText styling, top-left origin, and real annotation flattening."""

import asyncio
import json
from pathlib import Path

import pymupdf
import pytest
from pypdf import PdfReader, PdfWriter

from pdf_mcp import pdf_tools
from pdf_mcp.pdf_tools import PdfToolError

RECT = [20, 150, 180, 170]


def _blank(path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)
    return path


def _annot(path: Path):
    """Return (doc, page, annot) for the only annotation on page 1."""
    doc = pymupdf.open(str(path))
    page = doc[0]
    annots = list(page.annots())
    assert len(annots) == 1
    return doc, page, annots[0]


def _annotated(tmp_path: Path, **style) -> Path:
    out = tmp_path / "annot.pdf"
    pdf_tools.add_text_annotation(
        str(_blank(tmp_path / "in.pdf")), 1, "Example Pty Ltd", str(out), rect=RECT, annotation_id="a1", **style
    )
    return out


def test_defaults_unchanged(tmp_path: Path):
    out = _annotated(tmp_path)
    obj = PdfReader(str(out)).pages[0]["/Annots"][0].get_object()
    assert obj["/DA"] == "/Helv 12 Tf 0 g"
    assert "/BS" not in obj and "/AP" not in obj
    assert [float(v) for v in obj["/Rect"]] == RECT


def test_border_width_zero_means_no_border(tmp_path: Path):
    doc, _, annot = _annot(_annotated(tmp_path, border_width=0))
    assert annot.border["width"] == 0
    assert "/AP" in doc.xref_object(annot.xref)


def test_font_size_and_color_honoured(tmp_path: Path):
    doc, _, annot = _annot(_annotated(tmp_path, font_size=9, font_color=[1, 0, 0]))
    assert "/Helv 9 Tf" in doc.xref_object(annot.xref)
    # The appearance stream is what viewers render; check it carries the size and colour.
    ap = doc.xref_stream(int(doc.xref_get_key(annot.xref, "AP/N")[1].split()[0])).decode()
    assert "9 Tf" in ap and "1 0 0 rg" in ap


@pytest.mark.parametrize(
    "style",
    [{"font_size": 0}, {"font_color": [2, 0, 0]}, {"font_color": [0, 0]}, {"border_width": -1}],
)
def test_invalid_style_rejected(tmp_path: Path, style):
    with pytest.raises(PdfToolError):
        _annotated(tmp_path, **style)


def test_update_keeps_style_and_refreshes_appearance(tmp_path: Path):
    src = _annotated(tmp_path, font_size=9, border_width=0)
    out = tmp_path / "upd.pdf"
    pdf_tools.update_text_annotation(str(src), str(out), "a1", "Changed Ltd")
    baked = tmp_path / "upd_baked.pdf"
    pdf_tools.flatten_pdf(str(out), str(baked), include_annotations=True)
    assert "Changed Ltd" in pymupdf.open(str(baked))[0].get_text()
    doc, _, annot = _annot(out)
    assert annot.border["width"] == 0
    assert "/Helv 9" in doc.xref_object(annot.xref)


def test_update_can_restyle(tmp_path: Path):
    src = _annotated(tmp_path)
    out = tmp_path / "upd.pdf"
    pdf_tools.update_text_annotation(str(src), str(out), "a1", "x", border_width=0, font_size=10)
    doc, _, annot = _annot(out)
    assert annot.border["width"] == 0
    assert "/Helv 10 Tf" in doc.xref_object(annot.xref)


def test_top_left_origin_matches_text_blocks(tmp_path: Path):
    out = tmp_path / "tl.pdf"
    pdf_tools.add_text_annotation(
        str(_blank(tmp_path / "in.pdf")), 1, "x", str(out), rect=[20, 30, 180, 50], origin="top-left"
    )
    _, _, annot = _annot(out)
    assert tuple(annot.rect) == (20, 30, 180, 50)  # PyMuPDF reports top-left coordinates
    obj = PdfReader(str(out)).pages[0]["/Annots"][0].get_object()
    assert [float(v) for v in obj["/Rect"]] == [20, 150, 180, 170]  # stored bottom-left


def test_top_left_origin_requires_rect(tmp_path: Path):
    with pytest.raises(PdfToolError):
        pdf_tools.add_text_annotation(
            str(_blank(tmp_path / "in.pdf")), 1, "x", str(tmp_path / "o.pdf"), origin="top-left"
        )
    with pytest.raises(PdfToolError):
        pdf_tools.add_text_annotation(
            str(_blank(tmp_path / "in.pdf")), 1, "x", str(tmp_path / "o.pdf"), origin="centre"
        )


def test_flatten_without_form_does_not_claim_success(tmp_path: Path):
    src = _annotated(tmp_path)
    flat = tmp_path / "flat.pdf"
    res = pdf_tools.flatten_pdf(str(src), str(flat))
    assert res["flattened"] is False
    assert "include_annotations" in res["warning"]
    assert len(list(pymupdf.open(str(flat))[0].annots())) == 1  # left intact, not silently dropped


def test_flatten_include_annotations_bakes_text(tmp_path: Path):
    src = _annotated(tmp_path, border_width=0)
    flat = tmp_path / "flat.pdf"
    res = pdf_tools.flatten_pdf(str(src), str(flat), include_annotations=True)
    assert res["flattened"] is True and res["annotations_baked"] == 1
    page = pymupdf.open(str(flat))[0]
    assert list(page.annots()) == []
    assert "Example Pty Ltd" in page.get_text()


def test_flatten_include_annotations_nothing_to_do(tmp_path: Path):
    res = pdf_tools.flatten_pdf(str(_blank(tmp_path / "in.pdf")), str(tmp_path / "f.pdf"), include_annotations=True)
    assert res["flattened"] is False and "warning" in res


def test_mcp_schemas_expose_new_arguments():
    from pdf_mcp import server

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in ("add_text_annotation", "update_text_annotation"):
        props = tools[name].input_schema["properties"]
        assert {"font_size", "font_color", "border_width"} <= set(props)
    assert "origin" in tools["add_text_annotation"].input_schema["properties"]
    assert "include_annotations" in tools["flatten_pdf"].input_schema["properties"]


def test_cli_issue_reproduction(tmp_path: Path):
    from typer.testing import CliRunner

    from pdf_mcp.cli import app

    src = _blank(tmp_path / "in.pdf")
    a, flat = tmp_path / "a.pdf", tmp_path / "flat.pdf"
    runner = CliRunner()
    args = {
        "input_path": str(src),
        "output_path": str(a),
        "page": 1,
        "text": "Example Pty Ltd",
        "rect": RECT,
        "border_width": 0,
        "font_size": 11,
    }
    r = runner.invoke(app, ["text", "add-text-annotation", "--json", json.dumps(args)])
    assert r.exit_code == 0, r.output
    args = {"input_path": str(a), "output_path": str(flat), "include_annotations": True}
    r = runner.invoke(app, ["form", "flatten-pdf", "--json", json.dumps(args)])
    assert r.exit_code == 0, r.output
    page = pymupdf.open(str(flat))[0]
    assert list(page.annots()) == []
    assert "Example Pty Ltd" in page.get_text()
