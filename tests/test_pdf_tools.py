from pathlib import Path

from pdf_mcp import pdf_tools
from pdf_mcp.pdf_tools import PdfToolError
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)


def _make_pdf(path: Path, pages: int = 1) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


def _make_form_pdf(path: Path) -> Path:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    # Provide a basic font resource for appearance generation.
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)  # indirect

    # Minimal AcroForm text field named "Name".
    field = DictionaryObject(
        {
            NameObject("/FT"): NameObject("/Tx"),
            NameObject("/T"): TextStringObject("Name"),
            NameObject("/Ff"): NumberObject(0),
            NameObject("/V"): TextStringObject(""),
        }
    )
    field_ref = writer._add_object(field)  # indirect

    widget = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): ArrayObject(
                [NumberObject(50), NumberObject(100), NumberObject(250), NumberObject(130)]
            ),
            NameObject("/F"): NumberObject(4),
            NameObject("/V"): TextStringObject(""),
            NameObject("/Parent"): field_ref,
        }
    )
    widget_ref = writer._add_object(widget)  # indirect

    field[NameObject("/Kids")] = ArrayObject([widget_ref])
    page[NameObject("/Annots")] = ArrayObject([widget_ref])

    acro_form = DictionaryObject(
        {
            NameObject("/Fields"): ArrayObject([field_ref]),
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
            NameObject("/DR"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font_ref})}
            ),
        }
    )
    writer._root_object.update({NameObject("/AcroForm"): writer._add_object(acro_form)})  # type: ignore[attr-defined]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


def test_get_pdf_form_fields_empty(tmp_path: Path):
    src = _make_pdf(tmp_path / "blank.pdf", pages=1)
    result = pdf_tools.get_pdf_form_fields(str(src))
    assert result["count"] == 0
    assert isinstance(result["fields"], dict)


def test_fill_and_flatten(tmp_path: Path):
    src = _make_pdf(tmp_path / "form.pdf", pages=1)
    out = tmp_path / "filled.pdf"
    result = pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "Test User"}, flatten=True)
    assert Path(result["output_path"]).exists()
    assert result["flattened"] or result.get("flattened_with") is not None

    flat_out = tmp_path / "flattened.pdf"
    flat_result = pdf_tools.flatten_pdf(str(out), str(flat_out))
    assert Path(flat_result["output_path"]).exists()


def test_fill_updates_real_form_field(tmp_path: Path):
    src = _make_form_pdf(tmp_path / "real_form.pdf")
    out = tmp_path / "filled_real.pdf"
    result = pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "Test User"}, flatten=False)
    assert Path(result["output_path"]).exists()

    # Re-open and verify the field value was actually written.
    from pypdf import PdfReader

    reader = PdfReader(str(out))
    fields = reader.get_fields() or {}
    assert "Name" in fields
    assert str(fields["Name"].get("/V")) == "Test User"


def test_mcp_layer_can_call_all_tools(tmp_path: Path):
    """
    Smoke test the MCP layer in-process (closest to Cursor invocation) by calling
    each tool through FastMCP.call_tool and validating the results.
    """
    import asyncio

    from pdf_mcp import server
    from pypdf import PdfReader

    form_src = _make_form_pdf(tmp_path / "mcp_form.pdf")
    blank_a = _make_pdf(tmp_path / "mcp_a.pdf", pages=2)
    blank_b = _make_pdf(tmp_path / "mcp_b.pdf", pages=1)

    async def call(name: str, args: dict):
        _content, meta = await server.mcp.call_tool(name, args)
        assert isinstance(meta, dict)
        assert "result" in meta and isinstance(meta["result"], dict)
        result = meta["result"]
        assert "error" not in result, result.get("error")
        return result

    # get_pdf_form_fields
    res = asyncio.run(call("get_pdf_form_fields", {"pdf_path": str(form_src)}))
    assert res["count"] >= 1

    # fill_pdf_form (no flatten)
    filled = tmp_path / "mcp_filled.pdf"
    res = asyncio.run(
        call(
            "fill_pdf_form",
            {
                "input_path": str(form_src),
                "output_path": str(filled),
                "data": {"Name": "MCP User"},
                "flatten": False,
            },
        )
    )
    assert Path(res["output_path"]).exists()
    fields = (PdfReader(str(filled)).get_fields() or {})
    assert str(fields["Name"].get("/V")) == "MCP User"

    # flatten_pdf
    flat = tmp_path / "mcp_flat.pdf"
    res = asyncio.run(call("flatten_pdf", {"input_path": str(filled), "output_path": str(flat)}))
    assert Path(res["output_path"]).exists()

    # merge_pdfs
    merged = tmp_path / "mcp_merged.pdf"
    res = asyncio.run(
        call(
            "merge_pdfs",
            {"pdf_list": [str(blank_a), str(blank_b)], "output_path": str(merged)},
        )
    )
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(merged)).get_num_pages() == 3

    # extract_pages
    extracted = tmp_path / "mcp_extracted.pdf"
    res = asyncio.run(
        call(
            "extract_pages",
            {"input_path": str(merged), "pages": [1, 3], "output_path": str(extracted)},
        )
    )
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(extracted)).get_num_pages() == 2

    # rotate_pages
    rotated = tmp_path / "mcp_rotated.pdf"
    res = asyncio.run(
        call(
            "rotate_pages",
            {"input_path": str(merged), "pages": [1], "degrees": 90, "output_path": str(rotated)},
        )
    )
    assert Path(res["output_path"]).exists()
    rr = PdfReader(str(rotated))
    assert rr.pages[0].get("/Rotate") in (90, 450)  # depending on normalization


def test_merge_extract_rotate(tmp_path: Path):
    src1 = _make_pdf(tmp_path / "a.pdf", pages=2)
    src2 = _make_pdf(tmp_path / "b.pdf", pages=1)
    merged = tmp_path / "merged.pdf"

    merge_result = pdf_tools.merge_pdfs([str(src1), str(src2)], str(merged))
    assert merge_result["merged"] == 2
    assert Path(merge_result["output_path"]).exists()

    extracted = tmp_path / "extracted.pdf"
    extract_result = pdf_tools.extract_pages(str(merged), [1, 3], str(extracted))
    assert extract_result["extracted"] == 2
    assert Path(extract_result["output_path"]).exists()

    rotated = tmp_path / "rotated.pdf"
    rotate_result = pdf_tools.rotate_pages(str(merged), [1], 90, str(rotated))
    assert rotate_result["rotated"] == 1
    assert Path(rotate_result["output_path"]).exists()


def test_rotate_invalid_degrees(tmp_path: Path):
    src = _make_pdf(tmp_path / "c.pdf", pages=1)
    out = tmp_path / "rot_invalid.pdf"
    try:
        pdf_tools.rotate_pages(str(src), [1], 45, str(out))
        assert False, "Expected PdfToolError"
    except PdfToolError as exc:
        assert "multiple of 90" in str(exc)


def test_extract_out_of_range(tmp_path: Path):
    src = _make_pdf(tmp_path / "d.pdf", pages=1)
    out = tmp_path / "extract.pdf"
    try:
        pdf_tools.extract_pages(str(src), [2], str(out))
        assert False, "Expected PdfToolError"
    except PdfToolError as exc:
        assert "out of range" in str(exc)

