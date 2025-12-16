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

    # annotations and managed text
    annotated = tmp_path / "mcp_annotated.pdf"
    res = asyncio.run(
        call(
            "add_text_annotation",
            {
                "input_path": str(blank_a),
                "page": 1,
                "text": "Hello",
                "output_path": str(annotated),
                "annotation_id": "test-annot-1",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    edited = tmp_path / "mcp_annotated_edited.pdf"
    res = asyncio.run(
        call(
            "update_text_annotation",
            {
                "input_path": str(annotated),
                "output_path": str(edited),
                "annotation_id": "test-annot-1",
                "text": "Hello Edited",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    removed = tmp_path / "mcp_annotated_removed.pdf"
    res = asyncio.run(
        call(
            "remove_text_annotation",
            {
                "input_path": str(edited),
                "output_path": str(removed),
                "annotation_id": "test-annot-1",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    # pages insert/remove
    inserted = tmp_path / "mcp_pages_inserted.pdf"
    res = asyncio.run(
        call(
            "insert_pages",
            {
                "input_path": str(blank_a),
                "insert_from_path": str(blank_b),
                "at_page": 2,
                "output_path": str(inserted),
            },
        )
    )
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(inserted)).get_num_pages() == 3

    removed_pages = tmp_path / "mcp_pages_removed.pdf"
    res = asyncio.run(
        call(
            "remove_pages",
            {"input_path": str(inserted), "pages": [2], "output_path": str(removed_pages)},
        )
    )
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(removed_pages)).get_num_pages() == 2

    # metadata tools
    meta_out = tmp_path / "mcp_meta.pdf"
    res = asyncio.run(
        call(
            "set_pdf_metadata",
            {
                "input_path": str(blank_b),
                "output_path": str(meta_out),
                "title": "T",
                "author": "A",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    res = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(meta_out)}))
    md = res["metadata"]
    assert md.get("Title") == "T"
    assert md.get("Author") == "A"

    # watermark
    wm_out = tmp_path / "mcp_wm.pdf"
    res = asyncio.run(
        call(
            "add_text_watermark",
            {
                "input_path": str(blank_a),
                "output_path": str(wm_out),
                "text": "WM",
                "pages": [1, 2],
                "annotation_id": "wm-mcp-1",
            },
        )
    )
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    # comments (PyMuPDF Text annotations)
    c1 = tmp_path / "c1.pdf"
    c2 = tmp_path / "c2.pdf"
    c3 = tmp_path / "c3.pdf"
    res = asyncio.run(
        call(
            "add_comment",
            {
                "input_path": str(blank_a),
                "output_path": str(c1),
                "page": 1,
                "text": "hello",
                "pos": [72, 72],
                "comment_id": "mcp-c-1",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    res = asyncio.run(
        call(
            "update_comment",
            {
                "input_path": str(c1),
                "output_path": str(c2),
                "comment_id": "mcp-c-1",
                "text": "updated",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    res = asyncio.run(
        call(
            "remove_comment",
            {
                "input_path": str(c2),
                "output_path": str(c3),
                "comment_id": "mcp-c-1",
            },
        )
    )
    assert Path(res["output_path"]).exists()

    # signatures (image insert / replace / resize / remove)
    sig_png = tmp_path / "sig.png"
    _write_test_png(sig_png)
    s1 = tmp_path / "s1.pdf"
    s2 = tmp_path / "s2.pdf"
    s3 = tmp_path / "s3.pdf"
    s4 = tmp_path / "s4.pdf"

    res = asyncio.run(
        call(
            "add_signature_image",
            {
                "input_path": str(blank_a),
                "output_path": str(s1),
                "page": 1,
                "image_path": str(sig_png),
                "rect": [50, 50, 150, 100],
            },
        )
    )
    assert Path(res["output_path"]).exists()
    sig_xref = int(res["signature_xref"])
    assert sig_xref > 0

    res = asyncio.run(
        call(
            "update_signature_image",
            {
                "input_path": str(s1),
                "output_path": str(s2),
                "page": 1,
                "signature_xref": sig_xref,
                "image_path": str(sig_png),
            },
        )
    )
    assert Path(res["output_path"]).exists()

    res = asyncio.run(
        call(
            "update_signature_image",
            {
                "input_path": str(s2),
                "output_path": str(s3),
                "page": 1,
                "signature_xref": sig_xref,
                "rect": [60, 60, 200, 140],
            },
        )
    )
    assert Path(res["output_path"]).exists()
    sig_xref2 = int(res["signature_xref"])
    assert sig_xref2 > 0

    res = asyncio.run(
        call(
            "remove_signature_image",
            {
                "input_path": str(s3),
                "output_path": str(s4),
                "page": 1,
                "signature_xref": sig_xref2,
            },
        )
    )
    assert Path(res["output_path"]).exists()


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


def test_annotations_and_text_tools(tmp_path: Path):
    src = _make_pdf(tmp_path / "base.pdf", pages=1)

    annotated = tmp_path / "annotated.pdf"
    res = pdf_tools.add_text_annotation(
        str(src), page=1, text="Hello", output_path=str(annotated), annotation_id="a1"
    )
    assert Path(res["output_path"]).exists()

    from pypdf import PdfReader

    r = PdfReader(str(annotated))
    annots = r.pages[0].get("/Annots")
    assert annots is not None
    annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
    assert len(list(annots_obj)) == 1
    obj = list(annots_obj)[0].get_object()
    assert str(obj.get("/Subtype")) == "/FreeText"
    assert str(obj.get("/Contents")) == "Hello"
    assert str(obj.get("/NM")) == "a1"

    edited = tmp_path / "annotated_edited.pdf"
    res = pdf_tools.update_text_annotation(str(annotated), str(edited), "a1", "Hello Edited")
    assert Path(res["output_path"]).exists()
    r2 = PdfReader(str(edited))
    annots2 = r2.pages[0].get("/Annots").get_object()
    obj2 = list(annots2)[0].get_object()
    assert str(obj2.get("/Contents")) == "Hello Edited"

    removed = tmp_path / "annotated_removed.pdf"
    res = pdf_tools.remove_text_annotation(str(edited), str(removed), "a1")
    assert Path(res["output_path"]).exists()
    r3 = PdfReader(str(removed))
    annots3 = r3.pages[0].get("/Annots")
    if annots3 is not None:
        assert len(list(annots3.get_object())) == 0

    # managed text wrappers
    inserted = tmp_path / "text_inserted.pdf"
    res = pdf_tools.insert_text(str(src), page=1, text="T", output_path=str(inserted), text_id="t1")
    assert Path(res["output_path"]).exists()
    edited2 = tmp_path / "text_edited.pdf"
    res = pdf_tools.edit_text(str(inserted), str(edited2), "t1", "T2")
    assert Path(res["output_path"]).exists()
    removed2 = tmp_path / "text_removed.pdf"
    res = pdf_tools.remove_text(str(edited2), str(removed2), "t1")
    assert Path(res["output_path"]).exists()


def test_page_insert_remove(tmp_path: Path):
    base = _make_pdf(tmp_path / "base2.pdf", pages=2)
    ins = _make_pdf(tmp_path / "ins.pdf", pages=1)

    out = tmp_path / "inserted.pdf"
    res = pdf_tools.insert_pages(str(base), str(ins), at_page=2, output_path=str(out))
    assert Path(res["output_path"]).exists()

    from pypdf import PdfReader

    assert PdfReader(str(out)).get_num_pages() == 3

    out2 = tmp_path / "removed.pdf"
    res = pdf_tools.remove_pages(str(out), [2], str(out2))
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(out2)).get_num_pages() == 2


def test_remove_pages_refuse_all(tmp_path: Path):
    base = _make_pdf(tmp_path / "one.pdf", pages=1)
    out = tmp_path / "x.pdf"
    try:
        pdf_tools.remove_pages(str(base), [1], str(out))
        assert False, "Expected PdfToolError"
    except PdfToolError as exc:
        assert "remove all pages" in str(exc)


def test_pdf_metadata_roundtrip(tmp_path: Path):
    src = _make_pdf(tmp_path / "meta.pdf", pages=1)
    out = tmp_path / "meta_out.pdf"

    res = pdf_tools.set_pdf_metadata(
        str(src),
        str(out),
        title="My Title",
        author="My Author",
        subject="My Subject",
        keywords="k1,k2",
    )
    assert Path(res["output_path"]).exists()

    got = pdf_tools.get_pdf_metadata(str(out))["metadata"]
    assert got.get("Title") == "My Title"
    assert got.get("Author") == "My Author"
    assert got.get("Subject") == "My Subject"
    assert got.get("Keywords") == "k1,k2"


def test_text_watermark_adds_annotations(tmp_path: Path):
    src = _make_pdf(tmp_path / "wm.pdf", pages=2)
    out = tmp_path / "wm_out.pdf"

    res = pdf_tools.add_text_watermark(
        str(src),
        str(out),
        text="WATERMARK",
        pages=[1, 2],
        annotation_id="wm-1",
    )
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    from pypdf import PdfReader

    r = PdfReader(str(out))
    for p in r.pages:
        annots = p.get("/Annots")
        assert annots is not None
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        # Find our watermark annotation by NM
        found = False
        for ref in list(annots_obj):
            obj = ref.get_object()
            if str(obj.get("/NM")) == "wm-1":
                assert str(obj.get("/Subtype")) == "/FreeText"
                assert str(obj.get("/Contents")) == "WATERMARK"
                found = True
        assert found


def _write_test_png(path: Path) -> Path:
    # Minimal valid 1x1 PNG (base64)
    import base64

    png_b64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+Gk0AAAAASUVORK5CYII="
    path.write_bytes(base64.b64decode(png_b64))
    return path


def test_comments_add_update_remove(tmp_path: Path):
    import pymupdf

    src = _make_pdf(tmp_path / "c.pdf", pages=1)
    out1 = tmp_path / "c1.pdf"
    out2 = tmp_path / "c2.pdf"
    out3 = tmp_path / "c3.pdf"

    res = pdf_tools.add_comment(
        input_path=str(src),
        output_path=str(out1),
        page=1,
        text="hello",
        pos=[72, 72],
        comment_id="c-1",
    )
    assert Path(res["output_path"]).exists()

    # Verify comment exists by name
    doc = pymupdf.open(str(out1))
    try:
        p = doc.load_page(0)
        found = False
        for a in p.annots() or []:
            if a.info.get("name") == "c-1":
                assert a.type[1] == "Text"
                assert (a.info.get("content") or "") == "hello"
                found = True
        assert found
    finally:
        doc.close()

    res = pdf_tools.update_comment(
        input_path=str(out1),
        output_path=str(out2),
        comment_id="c-1",
        text="updated",
    )
    assert Path(res["output_path"]).exists()

    doc = pymupdf.open(str(out2))
    try:
        p = doc.load_page(0)
        for a in p.annots() or []:
            if a.info.get("name") == "c-1":
                assert (a.info.get("content") or "") == "updated"
    finally:
        doc.close()

    res = pdf_tools.remove_comment(
        input_path=str(out2),
        output_path=str(out3),
        comment_id="c-1",
    )
    assert Path(res["output_path"]).exists()

    doc = pymupdf.open(str(out3))
    try:
        p = doc.load_page(0)
        assert all((a.info.get("name") != "c-1") for a in (p.annots() or []))
    finally:
        doc.close()


def test_signature_add_update_resize_remove(tmp_path: Path):
    import pymupdf

    src = _make_pdf(tmp_path / "s.pdf", pages=1)
    img1 = _write_test_png(tmp_path / "sig1.png")
    img2 = _write_test_png(tmp_path / "sig2.png")

    out1 = tmp_path / "s1.pdf"
    out2 = tmp_path / "s2.pdf"
    out3 = tmp_path / "s3.pdf"
    out4 = tmp_path / "s4.pdf"

    res = pdf_tools.add_signature_image(
        input_path=str(src),
        output_path=str(out1),
        page=1,
        image_path=str(img1),
        rect=[50, 50, 150, 100],
    )
    xref = int(res["signature_xref"])
    assert xref > 0

    doc = pymupdf.open(str(out1))
    try:
        p = doc.load_page(0)
        xrefs = [x[0] for x in p.get_images(full=True)]
        assert xref in xrefs
    finally:
        doc.close()

    # Update image bytes in place (keep xref)
    res = pdf_tools.update_signature_image(
        input_path=str(out1),
        output_path=str(out2),
        page=1,
        signature_xref=xref,
        image_path=str(img2),
    )
    assert int(res["signature_xref"]) == xref

    # Resize (reinsert, xref may change)
    res = pdf_tools.update_signature_image(
        input_path=str(out2),
        output_path=str(out3),
        page=1,
        signature_xref=xref,
        rect=[60, 60, 200, 140],
    )
    xref2 = int(res["signature_xref"])
    assert xref2 > 0

    doc = pymupdf.open(str(out3))
    try:
        p = doc.load_page(0)
        xrefs = [x[0] for x in p.get_images(full=True)]
        assert xref2 in xrefs
    finally:
        doc.close()

    # Remove
    res = pdf_tools.remove_signature_image(
        input_path=str(out3),
        output_path=str(out4),
        page=1,
        signature_xref=xref2,
    )
    assert Path(res["output_path"]).exists()
    doc = pymupdf.open(str(out4))
    try:
        p = doc.load_page(0)
        # The signature should no longer be placed on the page.
        try:
            assert p.get_image_rects(xref2) == []
        except ValueError:
            # If garbage collection removed the xref entirely, that's also fine.
            pass
    finally:
        doc.close()

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

