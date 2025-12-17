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


def test_fill_pdf_form_falls_back_on_pdfrw_object_stream_failure(tmp_path: Path):
    """
    Regression: some PDFs (e.g. Adobe InDesign exports with compressed object streams)
    cause fillpdf/pdfrw parsing errors. fill_pdf_form must fall back to pypdf and succeed.
    """
    src = Path(__file__).parent / "1006.pdf"
    assert src.exists(), "Missing test fixture tests/1006.pdf"

    from pypdf import PdfReader

    reader = PdfReader(str(src))
    fields = reader.get_fields() or {}
    assert fields, "Expected form fields in tests/1006.pdf"

    # Pick a text field if available, else any field.
    key = None
    for name, f in fields.items():
        try:
            if str(f.get("/FT")) == "/Tx":
                key = name
                break
        except Exception:
            continue
    if key is None:
        key = next(iter(fields.keys()))

    out = tmp_path / "1006-filled.pdf"
    result = pdf_tools.fill_pdf_form(str(src), str(out), {str(key): "Test"}, flatten=False)
    assert Path(result["output_path"]).exists()
    assert result["filled_with"] in ("fillpdf", "pypdf")

    verify = PdfReader(str(out))
    vf = verify.get_fields() or {}
    assert str(vf[str(key)].get("/V")) == "Test"


def test_fill_pdf_form_fallback_when_fillpdf_raises(tmp_path: Path, monkeypatch):
    """
    Unit-level: if fillpdf throws, we should still succeed via pypdf.
    """
    src = _make_form_pdf(tmp_path / "real_form2.pdf")
    out = tmp_path / "filled_real2.pdf"

    # Only meaningful if fillpdf is available; otherwise pypdf is already used.
    if getattr(pdf_tools, "_HAS_FILLPDF", False) is True:
        monkeypatch.setattr(pdf_tools.fillpdfs, "write_fillable_pdf", lambda *a, **k: (_ for _ in ()).throw(ValueError("pdfrw fail")))

    result = pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "X"}, flatten=False)
    assert Path(result["output_path"]).exists()

    from pypdf import PdfReader

    r = PdfReader(str(out))
    f = r.get_fields() or {}
    assert str(f["Name"].get("/V")) == "X"


def test_clear_pdf_form_fields(tmp_path: Path):
    src = _make_form_pdf(tmp_path / "real_form.pdf")
    out = tmp_path / "filled_real.pdf"
    pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "X"}, flatten=False)

    cleared = tmp_path / "cleared.pdf"
    res = pdf_tools.clear_pdf_form_fields(str(out), str(cleared), fields=["Name"])
    assert Path(res["output_path"]).exists()

    from pypdf import PdfReader

    r = PdfReader(str(cleared))
    fields = r.get_fields() or {}
    assert str(fields["Name"].get("/V")) == ""


def test_encrypt_pdf_roundtrip(tmp_path: Path):
    src = _make_pdf(tmp_path / "plain.pdf", pages=1)
    enc = tmp_path / "enc.pdf"
    res = pdf_tools.encrypt_pdf(str(src), str(enc), user_password="userpw")
    assert Path(res["output_path"]).exists()

    from pypdf import PdfReader

    r = PdfReader(str(enc))
    assert r.is_encrypted is True
    assert r.decrypt("wrong") == 0
    assert r.decrypt("userpw") in (1, 2)
    assert len(r.pages) == 1


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

    # clear_pdf_form_fields
    cleared = tmp_path / "mcp_cleared.pdf"
    res = asyncio.run(
        call(
            "clear_pdf_form_fields",
            {"input_path": str(filled), "output_path": str(cleared), "fields": ["Name"]},
        )
    )
    assert Path(res["output_path"]).exists()
    fields2 = (PdfReader(str(cleared)).get_fields() or {})
    assert str(fields2["Name"].get("/V")) == ""

    # encrypt_pdf
    encrypted = tmp_path / "mcp_encrypted.pdf"
    res = asyncio.run(
        call(
            "encrypt_pdf",
            {"input_path": str(cleared), "output_path": str(encrypted), "user_password": "pw"},
        )
    )
    assert Path(res["output_path"]).exists()
    rr = PdfReader(str(encrypted))
    assert rr.is_encrypted is True
    assert rr.decrypt("pw") in (1, 2)
    assert len(rr.pages) >= 1

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


def test_mcp_layer_real_world_1006_regression(tmp_path: Path):
    """
    Real-world regression suite using tests/1006.pdf (InDesign-style object streams).

    Covers hard requirements end-to-end via the MCP layer:
    - fill, update, clear form values
    - comments CRUD
    - managed text insert/edit/remove
    - sign (visual signature) then encrypt and validate it can be opened with password
    """
    import asyncio

    import pymupdf
    from pdf_mcp import server
    from pypdf import PdfReader

    src = Path(__file__).parent / "1006.pdf"
    assert src.exists(), "Missing fixture tests/1006.pdf"

    async def call(name: str, args: dict):
        _content, meta = await server.mcp.call_tool(name, args)
        assert isinstance(meta, dict)
        assert "result" in meta and isinstance(meta["result"], dict)
        result = meta["result"]
        assert "error" not in result, result.get("error")
        return result

    def assert_field_value(pdf_path: Path, field_name: str, expected: str):
        r = PdfReader(str(pdf_path))
        f = r.get_fields() or {}
        assert field_name in f
        assert str(f[field_name].get("/V")) == expected

    def assert_has_nm_annotation(pdf_path: Path, page_idx: int, nm: str, expected_present: bool):
        r = PdfReader(str(pdf_path))
        page = r.pages[page_idx]
        annots = page.get("/Annots")
        if annots is None:
            assert expected_present is False
            return
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        found = False
        for ref in list(annots_obj):
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            try:
                if str(obj.get("/NM")) == nm:
                    found = True
                    break
            except Exception:
                continue
        assert found is expected_present

    # Pick one or two text fields from get_pdf_form_fields output.
    res = asyncio.run(call("get_pdf_form_fields", {"pdf_path": str(src)}))
    assert res["count"] >= 1
    fields = res["fields"] or {}
    text_fields = [k for (k, v) in fields.items() if (v or {}).get("type") == "/Tx"]
    assert text_fields, "Expected at least one text field in tests/1006.pdf"
    f1 = text_fields[0]
    f2 = next((x for x in text_fields[1:] if x != f1), None)

    filled1 = tmp_path / "1006_filled1.pdf"
    data = {f1: "Alice"} if f2 is None else {f1: "Alice", f2: "Bob"}
    asyncio.run(
        call(
            "fill_pdf_form",
            {"input_path": str(src), "output_path": str(filled1), "data": data, "flatten": False},
        )
    )
    assert filled1.exists()
    assert_field_value(filled1, f1, "Alice")
    if f2 is not None:
        assert_field_value(filled1, f2, "Bob")

    # Update a value (fill again)
    filled2 = tmp_path / "1006_filled2.pdf"
    asyncio.run(
        call(
            "fill_pdf_form",
            {
                "input_path": str(filled1),
                "output_path": str(filled2),
                "data": {f1: "Alice2"},
                "flatten": False,
            },
        )
    )
    assert filled2.exists()
    assert_field_value(filled2, f1, "Alice2")

    # Clear a value
    cleared = tmp_path / "1006_cleared.pdf"
    asyncio.run(
        call(
            "clear_pdf_form_fields",
            {"input_path": str(filled2), "output_path": str(cleared), "fields": [f1]},
        )
    )
    assert cleared.exists()
    assert_field_value(cleared, f1, "")

    # Comments CRUD (PyMuPDF annotation)
    c1 = tmp_path / "1006_c1.pdf"
    c2 = tmp_path / "1006_c2.pdf"
    c3 = tmp_path / "1006_c3.pdf"
    asyncio.run(
        call(
            "add_comment",
            {
                "input_path": str(cleared),
                "output_path": str(c1),
                "page": 1,
                "text": "hello",
                "pos": [72, 72],
                "comment_id": "c-1006",
            },
        )
    )
    asyncio.run(
        call(
            "update_comment",
            {"input_path": str(c1), "output_path": str(c2), "comment_id": "c-1006", "text": "updated"},
        )
    )
    asyncio.run(
        call(
            "remove_comment",
            {"input_path": str(c2), "output_path": str(c3), "comment_id": "c-1006"},
        )
    )

    # Verify comment got removed by name
    doc = pymupdf.open(str(c3))
    try:
        p = doc.load_page(0)
        assert all((a.info.get("name") != "c-1006") for a in (p.annots() or []))
    finally:
        doc.close()

    # Managed text insert/edit/remove (FreeText annotations with stable NM)
    t1 = tmp_path / "1006_t1.pdf"
    t2 = tmp_path / "1006_t2.pdf"
    t3 = tmp_path / "1006_t3.pdf"
    asyncio.run(
        call(
            "insert_text",
            {"input_path": str(c3), "page": 1, "text": "T", "output_path": str(t1), "text_id": "t-1006"},
        )
    )
    assert_has_nm_annotation(t1, page_idx=0, nm="t-1006", expected_present=True)
    asyncio.run(call("edit_text", {"input_path": str(t1), "output_path": str(t2), "text_id": "t-1006", "text": "T2"}))
    assert_has_nm_annotation(t2, page_idx=0, nm="t-1006", expected_present=True)
    asyncio.run(call("remove_text", {"input_path": str(t2), "output_path": str(t3), "text_id": "t-1006"}))
    assert_has_nm_annotation(t3, page_idx=0, nm="t-1006", expected_present=False)

    # Signature image then encrypt (visual signature, not cryptographic)
    sig_png = _write_test_png(tmp_path / "sig1006.png")
    signed = tmp_path / "1006_signed.pdf"
    res = asyncio.run(
        call(
            "add_signature_image",
            {"input_path": str(t3), "output_path": str(signed), "page": 1, "image_path": str(sig_png), "rect": [50, 50, 150, 100]},
        )
    )
    assert signed.exists()
    sig_xref = int(res["signature_xref"])
    assert sig_xref > 0

    encrypted = tmp_path / "1006_encrypted.pdf"
    asyncio.run(call("encrypt_pdf", {"input_path": str(signed), "output_path": str(encrypted), "user_password": "pw"}))
    assert encrypted.exists()
    rr = PdfReader(str(encrypted))
    assert rr.is_encrypted is True
    assert rr.decrypt("pw") in (1, 2)
    assert len(rr.pages) >= 1


def test_mcp_layer_1006_all_tools_scenario_a(tmp_path: Path):
    """
    Scenario A: Run every MCP tool using tests/1006.pdf as the primary input (or as
    the insert/merge source) and validate basic invariants.
    """
    import asyncio

    import pymupdf
    from pdf_mcp import server
    from pypdf import PdfReader

    src = Path(__file__).parent / "1006.pdf"
    assert src.exists(), "Missing fixture tests/1006.pdf"

    async def call(name: str, args: dict):
        _content, meta = await server.mcp.call_tool(name, args)
        result = meta["result"]
        assert "error" not in result, result.get("error")
        return result

    def pick_text_fields(pdf_path: Path) -> list[str]:
        res = asyncio.run(call("get_pdf_form_fields", {"pdf_path": str(pdf_path)}))
        fields = res.get("fields") or {}
        txt = [k for (k, v) in fields.items() if (v or {}).get("type") == "/Tx"]
        assert txt, "Expected at least one /Tx field"
        return txt

    # form fill/update/clear
    f1, *rest = pick_text_fields(src)
    f2 = rest[0] if rest else None

    filled = tmp_path / "a_filled.pdf"
    data = {f1: "A1"} if f2 is None else {f1: "A1", f2: "A2"}
    asyncio.run(call("fill_pdf_form", {"input_path": str(src), "output_path": str(filled), "data": data, "flatten": False}))
    r = PdfReader(str(filled))
    ff = r.get_fields() or {}
    assert str(ff[f1].get("/V")) == "A1"
    if f2 is not None:
        assert str(ff[f2].get("/V")) == "A2"

    updated = tmp_path / "a_updated.pdf"
    asyncio.run(call("fill_pdf_form", {"input_path": str(filled), "output_path": str(updated), "data": {f1: "A1b"}, "flatten": False}))
    r2 = PdfReader(str(updated))
    ff2 = r2.get_fields() or {}
    assert str(ff2[f1].get("/V")) == "A1b"

    cleared = tmp_path / "a_cleared.pdf"
    asyncio.run(call("clear_pdf_form_fields", {"input_path": str(updated), "output_path": str(cleared), "fields": [f1]}))
    r3 = PdfReader(str(cleared))
    ff3 = r3.get_fields() or {}
    assert str(ff3[f1].get("/V")) == ""

    # metadata get/set
    meta0 = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(cleared)}))
    assert "metadata" in meta0
    meta1 = tmp_path / "a_meta.pdf"
    asyncio.run(call("set_pdf_metadata", {"input_path": str(cleared), "output_path": str(meta1), "title": "T-A", "author": "Author-A"}))
    meta_after = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(meta1)}))["metadata"]
    assert meta_after.get("Title") == "T-A"
    assert meta_after.get("Author") == "Author-A"

    # watermark
    wm = tmp_path / "a_wm.pdf"
    asyncio.run(call("add_text_watermark", {"input_path": str(meta1), "output_path": str(wm), "text": "WM-A", "pages": [1], "annotation_id": "wm-a"}))
    rwm = PdfReader(str(wm))
    annots = rwm.pages[0].get("/Annots")
    assert annots is not None

    # text annotation add/update/remove
    a1 = tmp_path / "a_annot1.pdf"
    asyncio.run(call("add_text_annotation", {"input_path": str(wm), "output_path": str(a1), "page": 1, "text": "HelloA", "annotation_id": "ann-a"}))
    a2 = tmp_path / "a_annot2.pdf"
    asyncio.run(call("update_text_annotation", {"input_path": str(a1), "output_path": str(a2), "annotation_id": "ann-a", "text": "HelloA2"}))
    a3 = tmp_path / "a_annot3.pdf"
    asyncio.run(call("remove_text_annotation", {"input_path": str(a2), "output_path": str(a3), "annotation_id": "ann-a"}))

    # managed text insert/edit/remove
    t1 = tmp_path / "a_t1.pdf"
    asyncio.run(call("insert_text", {"input_path": str(a3), "output_path": str(t1), "page": 1, "text": "T", "text_id": "t-a"}))
    t2 = tmp_path / "a_t2.pdf"
    asyncio.run(call("edit_text", {"input_path": str(t1), "output_path": str(t2), "text_id": "t-a", "text": "T2"}))
    t3 = tmp_path / "a_t3.pdf"
    asyncio.run(call("remove_text", {"input_path": str(t2), "output_path": str(t3), "text_id": "t-a"}))

    # remove_annotations (FreeText only so we don't remove /Widget form fields)
    ra = tmp_path / "a_ra.pdf"
    res = asyncio.run(call("remove_annotations", {"input_path": str(t3), "output_path": str(ra), "pages": [1], "subtype": "FreeText"}))
    assert Path(res["output_path"]).exists()

    # comments CRUD
    c1 = tmp_path / "a_c1.pdf"
    c2 = tmp_path / "a_c2.pdf"
    c3 = tmp_path / "a_c3.pdf"
    asyncio.run(call("add_comment", {"input_path": str(ra), "output_path": str(c1), "page": 1, "text": "hello", "pos": [72, 72], "comment_id": "c-a"}))
    asyncio.run(call("update_comment", {"input_path": str(c1), "output_path": str(c2), "comment_id": "c-a", "text": "updated"}))
    asyncio.run(call("remove_comment", {"input_path": str(c2), "output_path": str(c3), "comment_id": "c-a"}))
    doc = pymupdf.open(str(c3))
    try:
        p = doc.load_page(0)
        assert all((a.info.get("name") != "c-a") for a in (p.annots() or []))
    finally:
        doc.close()

    # signature add/update/remove + xref handling
    img1 = _write_test_png(tmp_path / "a_sig1.png")
    img2 = _write_test_png(tmp_path / "a_sig2.png")
    s1 = tmp_path / "a_s1.pdf"
    res = asyncio.run(call("add_signature_image", {"input_path": str(c3), "output_path": str(s1), "page": 1, "image_path": str(img1), "rect": [50, 50, 150, 100]}))
    xref = int(res["signature_xref"])
    s2 = tmp_path / "a_s2.pdf"
    asyncio.run(call("update_signature_image", {"input_path": str(s1), "output_path": str(s2), "page": 1, "signature_xref": xref, "image_path": str(img2)}))
    s3 = tmp_path / "a_s3.pdf"
    res = asyncio.run(call("update_signature_image", {"input_path": str(s2), "output_path": str(s3), "page": 1, "signature_xref": xref, "rect": [60, 60, 200, 140]}))
    xref2 = int(res["signature_xref"])
    s4 = tmp_path / "a_s4.pdf"
    asyncio.run(call("remove_signature_image", {"input_path": str(s3), "output_path": str(s4), "page": 1, "signature_xref": xref2}))

    # merge/extract/rotate/insert/remove pages using 1006 as the source
    merged = tmp_path / "a_merged.pdf"
    asyncio.run(call("merge_pdfs", {"pdf_list": [str(src), str(src)], "output_path": str(merged)}))
    assert PdfReader(str(merged)).get_num_pages() == PdfReader(str(src)).get_num_pages() * 2

    extracted = tmp_path / "a_extracted.pdf"
    asyncio.run(call("extract_pages", {"input_path": str(merged), "pages": [1, -1], "output_path": str(extracted)}))
    assert PdfReader(str(extracted)).get_num_pages() == 2

    rotated = tmp_path / "a_rotated.pdf"
    asyncio.run(call("rotate_pages", {"input_path": str(extracted), "pages": [1], "degrees": 90, "output_path": str(rotated)}))
    rr = PdfReader(str(rotated))
    assert rr.pages[0].get("/Rotate") in (90, 450)

    inserted = tmp_path / "a_inserted.pdf"
    asyncio.run(call("insert_pages", {"input_path": str(rotated), "insert_from_path": str(src), "at_page": 2, "output_path": str(inserted)}))
    assert PdfReader(str(inserted)).get_num_pages() == PdfReader(str(rotated)).get_num_pages() + PdfReader(str(src)).get_num_pages()

    removed = tmp_path / "a_removed.pdf"
    asyncio.run(call("remove_pages", {"input_path": str(inserted), "pages": [2], "output_path": str(removed)}))
    assert PdfReader(str(removed)).get_num_pages() == PdfReader(str(inserted)).get_num_pages() - 1

    # flatten then encrypt (encryption last, no decrypt tool available)
    flat = tmp_path / "a_flat.pdf"
    asyncio.run(call("flatten_pdf", {"input_path": str(removed), "output_path": str(flat)}))
    assert (PdfReader(str(flat)).get_fields() or {}) == {}

    enc = tmp_path / "a_enc.pdf"
    asyncio.run(call("encrypt_pdf", {"input_path": str(flat), "output_path": str(enc), "user_password": "pw-a"}))
    er = PdfReader(str(enc))
    assert er.is_encrypted is True
    assert er.decrypt("pw-a") in (1, 2)


def test_mcp_layer_1006_all_tools_scenario_b(tmp_path: Path):
    """
    Scenario B: Second regression pass for every tool on 1006.pdf with different
    inputs/flags so each tool has 2+ real-world cases.
    """
    import asyncio

    from pdf_mcp import server
    from pypdf import PdfReader

    src = Path(__file__).parent / "1006.pdf"
    assert src.exists(), "Missing fixture tests/1006.pdf"

    async def call(name: str, args: dict):
        _content, meta = await server.mcp.call_tool(name, args)
        result = meta["result"]
        assert "error" not in result, result.get("error")
        return result

    # get fields again (case 2)
    res = asyncio.run(call("get_pdf_form_fields", {"pdf_path": str(src)}))
    assert res["count"] >= 1
    fields = res.get("fields") or {}
    txt = [k for (k, v) in fields.items() if (v or {}).get("type") == "/Tx"]
    assert txt
    f1 = txt[0]

    # fill + clear (2nd case for both tools, keep unflattened so fields exist)
    filled = tmp_path / "b_filled.pdf"
    asyncio.run(
        call(
            "fill_pdf_form",
            {"input_path": str(src), "output_path": str(filled), "data": {f1: "B1"}, "flatten": False},
        )
    )
    r = PdfReader(str(filled))
    ff = r.get_fields() or {}
    assert str(ff[f1].get("/V")) == "B1"

    cleared = tmp_path / "b_cleared.pdf"
    asyncio.run(
        call(
            "clear_pdf_form_fields",
            {"input_path": str(filled), "output_path": str(cleared), "fields": [f1]},
        )
    )
    rc = PdfReader(str(cleared))
    ffc = rc.get_fields() or {}
    assert str(ffc[f1].get("/V")) == ""

    # fill with flatten=True (forces pypdf path on this file if fillpdf can't parse)
    filled_flat = tmp_path / "b_filled_flat.pdf"
    asyncio.run(
        call(
            "fill_pdf_form",
            {"input_path": str(src), "output_path": str(filled_flat), "data": {f1: "B1-flat"}, "flatten": True},
        )
    )
    # flattened: form fields should be gone
    assert (PdfReader(str(filled_flat)).get_fields() or {}) == {}

    # metadata second case
    meta2 = tmp_path / "b_meta.pdf"
    asyncio.run(call("set_pdf_metadata", {"input_path": str(src), "output_path": str(meta2), "title": "T-B", "keywords": "k1,k2"}))
    got = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(meta2)}))["metadata"]
    assert got.get("Title") == "T-B"
    assert got.get("Keywords") == "k1,k2"

    # watermark second case (two pages)
    wm2 = tmp_path / "b_wm.pdf"
    asyncio.run(call("add_text_watermark", {"input_path": str(meta2), "output_path": str(wm2), "text": "WM-B", "pages": [1, 2], "annotation_id": "wm-b"}))
    r = PdfReader(str(wm2))
    assert r.get_num_pages() >= 2

    # text annotation second case: add/update/remove by id, then remove_annotations filter
    ann1 = tmp_path / "b_ann1.pdf"
    asyncio.run(call("add_text_annotation", {"input_path": str(wm2), "output_path": str(ann1), "page": 1, "text": "HelloB", "annotation_id": "ann-b"}))
    ann_upd = tmp_path / "b_ann_upd.pdf"
    asyncio.run(call("update_text_annotation", {"input_path": str(ann1), "output_path": str(ann_upd), "annotation_id": "ann-b", "text": "HelloB2"}))
    ann2 = tmp_path / "b_ann2.pdf"
    asyncio.run(call("remove_text_annotation", {"input_path": str(ann_upd), "output_path": str(ann2), "annotation_id": "ann-b"}))

    ann3 = tmp_path / "b_ann3.pdf"
    asyncio.run(call("remove_annotations", {"input_path": str(ann2), "output_path": str(ann3), "pages": [1], "subtype": "FreeText"}))

    # managed text second case (different id)
    t1 = tmp_path / "b_t1.pdf"
    asyncio.run(call("insert_text", {"input_path": str(ann3), "output_path": str(t1), "page": 1, "text": "TB", "text_id": "t-b"}))
    t2 = tmp_path / "b_t2.pdf"
    asyncio.run(call("edit_text", {"input_path": str(t1), "output_path": str(t2), "text_id": "t-b", "text": "TB2"}))
    t3 = tmp_path / "b_t3.pdf"
    asyncio.run(call("remove_text", {"input_path": str(t2), "output_path": str(t3), "text_id": "t-b"}))

    # pages second case (different selections)
    ext = tmp_path / "b_ext.pdf"
    asyncio.run(call("extract_pages", {"input_path": str(src), "pages": [1, 2], "output_path": str(ext)}))
    assert PdfReader(str(ext)).get_num_pages() == 2

    rot = tmp_path / "b_rot.pdf"
    asyncio.run(call("rotate_pages", {"input_path": str(ext), "pages": [2], "degrees": -90, "output_path": str(rot)}))
    rr = PdfReader(str(rot))
    assert rr.pages[1].get("/Rotate") in (-90, 270, 630)

    merged = tmp_path / "b_merge.pdf"
    asyncio.run(call("merge_pdfs", {"pdf_list": [str(ext), str(ext)], "output_path": str(merged)}))
    assert PdfReader(str(merged)).get_num_pages() == 4

    ins = tmp_path / "b_ins.pdf"
    asyncio.run(call("insert_pages", {"input_path": str(ext), "insert_from_path": str(ext), "at_page": 1, "output_path": str(ins)}))
    assert PdfReader(str(ins)).get_num_pages() == 4

    rem = tmp_path / "b_rem.pdf"
    asyncio.run(call("remove_pages", {"input_path": str(ins), "pages": [1, 3], "output_path": str(rem)}))
    assert PdfReader(str(rem)).get_num_pages() == 2

    flat = tmp_path / "b_flat.pdf"
    asyncio.run(call("flatten_pdf", {"input_path": str(src), "output_path": str(flat)}))
    assert (PdfReader(str(flat)).get_fields() or {}) == {}

    # encrypt second case (different flags)
    enc = tmp_path / "b_enc.pdf"
    asyncio.run(
        call(
            "encrypt_pdf",
            {
                "input_path": str(flat),
                "output_path": str(enc),
                "user_password": "pw-b",
                "allow_printing": False,
                "allow_copying": False,
                "allow_modifying": False,
                "allow_annotations": False,
                "allow_form_filling": False,
            },
        )
    )
    er = PdfReader(str(enc))
    assert er.is_encrypted is True
    assert er.decrypt("pw-b") in (1, 2)

    # comments CRUD (2nd case)
    import pymupdf

    c1 = tmp_path / "b_c1.pdf"
    c2 = tmp_path / "b_c2.pdf"
    c3 = tmp_path / "b_c3.pdf"
    asyncio.run(call("add_comment", {"input_path": str(src), "output_path": str(c1), "page": 1, "text": "hello2", "pos": [80, 80], "comment_id": "c-b"}))
    asyncio.run(call("update_comment", {"input_path": str(c1), "output_path": str(c2), "comment_id": "c-b", "text": "updated2"}))
    asyncio.run(call("remove_comment", {"input_path": str(c2), "output_path": str(c3), "comment_id": "c-b"}))
    doc = pymupdf.open(str(c3))
    try:
        p = doc.load_page(0)
        assert all((a.info.get("name") != "c-b") for a in (p.annots() or []))
    finally:
        doc.close()

    # signature add/update/remove (2nd case)
    img1 = _write_test_png(tmp_path / "b_sig1.png")
    img2 = _write_test_png(tmp_path / "b_sig2.png")
    s1 = tmp_path / "b_s1.pdf"
    res = asyncio.run(call("add_signature_image", {"input_path": str(src), "output_path": str(s1), "page": 1, "image_path": str(img1), "rect": [40, 40, 140, 90]}))
    xref = int(res["signature_xref"])
    s2 = tmp_path / "b_s2.pdf"
    asyncio.run(call("update_signature_image", {"input_path": str(s1), "output_path": str(s2), "page": 1, "signature_xref": xref, "image_path": str(img2)}))
    s3 = tmp_path / "b_s3.pdf"
    asyncio.run(call("remove_signature_image", {"input_path": str(s2), "output_path": str(s3), "page": 1, "signature_xref": xref}))

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

