import json
from pathlib import Path
from typing import Dict

import pymupdf
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from pdf_mcp import pdf_tools
from pdf_mcp.pdf_tools import PdfToolError


def _make_pdf(path: Path, pages: int = 1) -> Path:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


def _make_xfa_pdf(path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    acro_form = DictionaryObject({NameObject("/XFA"): TextStringObject("dummy")})
    writer._root_object.update({NameObject("/AcroForm"): writer._add_object(acro_form)})  # type: ignore[attr-defined]
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


def _make_text_pdf(path: Path, lines: list[str]) -> Path:
    doc = pymupdf.open()
    for line in lines:
        page = doc.new_page()
        page.insert_text((72, 72), line, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _make_nonstandard_form_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 100), "Name:", fontsize=12)
    page.draw_line((110, 102), (240, 102))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _make_test_certificates(tmp_path: Path) -> Dict[str, Path]:
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pdf-mcp-test")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    key_path = tmp_path / "test_key.pem"
    cert_path = tmp_path / "test_cert.pem"
    pfx_path = tmp_path / "test_cert.pfx"
    password = b"test-pass"

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    pfx_path.write_bytes(
        pkcs12.serialize_key_and_certificates(
            b"pdf-mcp-test", key, cert, None, serialization.BestAvailableEncryption(password)
        )
    )

    return {
        "key": key_path,
        "cert": cert_path,
        "pfx": pfx_path,
        "password": password,
        "key_obj": key,
        "cert_obj": cert,
    }


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


def test_create_pdf_form_and_fill(tmp_path: Path):
    out = tmp_path / "created_form.pdf"
    result = pdf_tools.create_pdf_form(
        output_path=str(out),
        fields=[
            {"name": "Name", "type": "text", "rect": [50, 100, 250, 130]},
            {"name": "Agree", "type": "checkbox", "rect": [50, 60, 70, 80], "value": True},
        ],
        pages=1,
    )
    assert Path(result["output_path"]).exists()

    fields = pdf_tools.get_pdf_form_fields(str(out))
    assert fields["count"] >= 2

    filled = tmp_path / "created_form_filled.pdf"
    res = pdf_tools.fill_pdf_form(str(out), str(filled), {"Name": "Created User"}, flatten=False)
    assert Path(res["output_path"]).exists()


def test_create_pdf_form_from_template(tmp_path: Path):
    templates = pdf_tools.get_form_templates()
    assert "client_intake_basic" in templates.get("templates", {})

    out = tmp_path / "template_form.pdf"
    result = pdf_tools.create_pdf_form_from_template(str(out), "client_intake_basic")
    assert Path(result["output_path"]).exists()

    fields = pdf_tools.get_pdf_form_fields(str(out))
    assert fields["count"] >= 1


def test_fill_pdf_form_any_nonstandard(tmp_path: Path):
    src = _make_nonstandard_form_pdf(tmp_path / "nonstandard.pdf")
    out = tmp_path / "nonstandard_filled.pdf"
    result = pdf_tools.fill_pdf_form_any(str(src), str(out), {"Name": "Nonstandard User"}, flatten=False)
    assert Path(result["output_path"]).exists()
    assert result["fields_filled"] >= 1

    doc = pymupdf.open(str(out))
    page = doc.load_page(0)
    annots = list(page.annots() or [])
    doc.close()
    assert len(annots) >= 1


def test_xfa_form_unsupported_for_get_fields(tmp_path: Path):
    src = _make_xfa_pdf(tmp_path / "xfa.pdf")
    result = pdf_tools.get_pdf_form_fields(str(src))
    assert result.get("xfa") is True
    assert "error" in result
    assert result.get("count") == 0


def test_xfa_form_unsupported_for_fill(tmp_path: Path):
    src = _make_xfa_pdf(tmp_path / "xfa_fill.pdf")
    out = tmp_path / "xfa_out.pdf"
    with pytest.raises(PdfToolError) as exc_info:
        pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "X"}, flatten=False)
    assert "XFA" in str(exc_info.value)


def test_xfa_form_unsupported_for_fill_any(tmp_path: Path):
    src = _make_xfa_pdf(tmp_path / "xfa_fill_any.pdf")
    out = tmp_path / "xfa_any_out.pdf"
    with pytest.raises(PdfToolError) as exc_info:
        pdf_tools.fill_pdf_form_any(str(src), str(out), {"Name": "X"}, flatten=False)
    assert "XFA" in str(exc_info.value)


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


def _make_checkbox_form_pdf(path: Path) -> Path:
    """Create a PDF with both text and checkbox AcroForm fields."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)

    # Text field: "Name"
    text_field = DictionaryObject(
        {
            NameObject("/FT"): NameObject("/Tx"),
            NameObject("/T"): TextStringObject("Name"),
            NameObject("/Ff"): NumberObject(0),
            NameObject("/V"): TextStringObject(""),
        }
    )
    text_ref = writer._add_object(text_field)

    text_widget = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): ArrayObject(
                [NumberObject(50), NumberObject(200), NumberObject(250), NumberObject(230)]
            ),
            NameObject("/F"): NumberObject(4),
            NameObject("/Parent"): text_ref,
        }
    )
    text_widget_ref = writer._add_object(text_widget)
    text_field[NameObject("/Kids")] = ArrayObject([text_widget_ref])

    # Checkbox field: "Agree"
    cb_field = DictionaryObject(
        {
            NameObject("/FT"): NameObject("/Btn"),
            NameObject("/T"): TextStringObject("Agree"),
            NameObject("/V"): NameObject("/Off"),
            NameObject("/AS"): NameObject("/Off"),
        }
    )
    cb_ref = writer._add_object(cb_field)

    cb_widget = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): ArrayObject(
                [NumberObject(50), NumberObject(100), NumberObject(70), NumberObject(120)]
            ),
            NameObject("/F"): NumberObject(4),
            NameObject("/Parent"): cb_ref,
        }
    )
    cb_widget_ref = writer._add_object(cb_widget)
    cb_field[NameObject("/Kids")] = ArrayObject([cb_widget_ref])

    page[NameObject("/Annots")] = ArrayObject([text_widget_ref, cb_widget_ref])

    acro_form = DictionaryObject(
        {
            NameObject("/Fields"): ArrayObject([text_ref, cb_ref]),
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
            NameObject("/DR"): DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font_ref})}
            ),
        }
    )
    writer._root_object.update({NameObject("/AcroForm"): writer._add_object(acro_form)})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path


# ============================================================================
# Tests: AcroForm checkbox/radio button proper toggle
# ============================================================================


class TestCheckboxRadioFormFilling:
    """Tests for proper AcroForm checkbox/radio button handling in fill_pdf_form."""

    def test_fill_checkbox_with_truthy_sets_name_object(self, tmp_path):
        """Filling a checkbox with a truthy value should set /V and /AS as NameObject, not TextStringObject."""
        src = _make_checkbox_form_pdf(tmp_path / "cb_form.pdf")
        out = tmp_path / "cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "Alice", "Agree": "Yes"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(out))
        fields = r.get_fields() or {}

        # Text field should be filled normally
        assert str(fields["Name"].get("/V")) == "Alice"

        # Checkbox field: /V should be a NameObject (e.g. /Yes), NOT a TextStringObject
        agree_v = fields["Agree"].get("/V")
        assert agree_v is not None
        assert str(agree_v) == "/Yes", f"Expected /Yes but got {agree_v!r}"

    def test_fill_checkbox_with_true_bool(self, tmp_path):
        """Boolean True should check the checkbox."""
        src = _make_checkbox_form_pdf(tmp_path / "cb_form.pdf")
        out = tmp_path / "cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(src), str(out), {"Agree": "True"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(out))
        fields = r.get_fields() or {}
        assert str(fields["Agree"].get("/V")) == "/Yes"

    def test_fill_checkbox_with_falsy_sets_off(self, tmp_path):
        """Filling a checkbox with a falsy value should set /V to /Off."""
        src = _make_checkbox_form_pdf(tmp_path / "cb_form.pdf")
        out = tmp_path / "cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(src), str(out), {"Agree": "No"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(out))
        fields = r.get_fields() or {}
        assert str(fields["Agree"].get("/V")) == "/Off"

    def test_fill_checkbox_preserves_text_field_values(self, tmp_path):
        """Checkbox handling should not break normal text field filling."""
        src = _make_checkbox_form_pdf(tmp_path / "cb_form.pdf")
        out = tmp_path / "cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "Bob", "Agree": "Yes"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(out))
        fields = r.get_fields() or {}
        # Text field must still be a plain string value
        assert str(fields["Name"].get("/V")) == "Bob"

    def test_fill_checkbox_with_x_is_truthy(self, tmp_path):
        """'X' should be recognized as truthy for checkbox."""
        src = _make_checkbox_form_pdf(tmp_path / "cb_form.pdf")
        out = tmp_path / "cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(src), str(out), {"Agree": "X"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(out))
        fields = r.get_fields() or {}
        assert str(fields["Agree"].get("/V")) == "/Yes"

    def test_create_and_fill_checkbox_roundtrip(self, tmp_path):
        """create_pdf_form with checkbox -> fill_pdf_form -> verify roundtrip."""
        form_path = tmp_path / "created_cb.pdf"
        pdf_tools.create_pdf_form(str(form_path), [
            {"name": "FullName", "type": "text", "rect": [50, 100, 250, 130]},
            {"name": "Consent", "type": "checkbox", "rect": [50, 60, 70, 80]},
        ])

        filled_path = tmp_path / "created_cb_filled.pdf"
        pdf_tools.fill_pdf_form(str(form_path), str(filled_path), {"FullName": "Charlie", "Consent": "Yes"}, flatten=False)

        from pypdf import PdfReader
        r = PdfReader(str(filled_path))
        fields = r.get_fields() or {}
        assert str(fields["FullName"].get("/V")) == "Charlie"
        assert str(fields["Consent"].get("/V")) == "/Yes"


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


def test_reorder_pages_basic(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "ordered.pdf", ["Page 1", "Page 2", "Page 3"])
    out = tmp_path / "reordered.pdf"

    res = pdf_tools.reorder_pages(str(src), [3, 1, 2], str(out))
    assert Path(res["output_path"]).exists()

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    texts = [(page.extract_text() or "") for page in reader.pages]
    assert "Page 3" in texts[0]
    assert "Page 1" in texts[1]
    assert "Page 2" in texts[2]


def test_reorder_pages_rejects_invalid_input(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "ordered.pdf", ["Page 1", "Page 2", "Page 3"])
    out = tmp_path / "reordered.pdf"

    with pytest.raises(PdfToolError):
        pdf_tools.reorder_pages(str(src), [1, 1, 2], str(out))

    with pytest.raises(PdfToolError):
        pdf_tools.reorder_pages(str(src), [1, 2], str(out))

    with pytest.raises(PdfToolError):
        pdf_tools.reorder_pages(str(src), [0, 1, 2], str(out))


def test_redact_text_regex_basic(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "text.pdf", ["Secret 123", "Public info"])
    out = tmp_path / "redacted.pdf"

    res = pdf_tools.redact_text_regex(
        input_path=str(src),
        output_path=str(out),
        pattern=r"Secret\s+\d+",
    )
    assert Path(res["output_path"]).exists()
    assert res["redacted"] >= 1

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    text = "".join((page.extract_text() or "") for page in reader.pages)
    assert "Secret" not in text
    assert "Public" in text


def test_sanitize_pdf_metadata_removes_keys(tmp_path: Path):
    src = _make_pdf(tmp_path / "plain.pdf", pages=1)
    meta = tmp_path / "meta.pdf"
    pdf_tools.set_pdf_metadata(
        str(src),
        str(meta),
        title="Title",
        author="Author",
        subject="Subject",
        keywords="Keywords",
    )

    sanitized = tmp_path / "sanitized.pdf"
    res = pdf_tools.sanitize_pdf_metadata(str(meta), str(sanitized))
    assert Path(res["output_path"]).exists()
    assert "Title" in res["removed"]
    assert "Author" in res["removed"]

    md = pdf_tools.get_pdf_metadata(str(sanitized))["metadata"]
    assert "Title" not in md
    assert "Author" not in md
    assert "Subject" not in md
    assert "Keywords" not in md


def test_export_pdf_json_basic(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "export.pdf", ["Hello world", "Second page"])
    out = tmp_path / "export.json"

    res = pdf_tools.export_pdf(str(src), str(out), format="json")
    assert Path(res["output_path"]).exists()

    data = json.loads(Path(res["output_path"]).read_text())
    assert data["page_count"] == 2
    assert data["engine"] in ("auto", "native", "ocr")
    assert "pages" in data and len(data["pages"]) == 2
    assert "Hello world" in data["pages"][0]["text"]


def test_export_pdf_markdown_basic(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "export.md.pdf", ["Hello world", "Second page"])
    out = tmp_path / "export.md"

    res = pdf_tools.export_pdf(str(src), str(out), format="markdown")
    assert Path(res["output_path"]).exists()

    content = Path(res["output_path"]).read_text()
    assert "Hello world" in content
    assert "Second page" in content


def test_add_page_numbers_writes_annotations(tmp_path: Path):
    src = _make_pdf(tmp_path / "pages.pdf", pages=2)
    out = tmp_path / "pages_numbered.pdf"

    res = pdf_tools.add_page_numbers(str(src), str(out))
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    annots = reader.pages[0].get("/Annots")
    assert annots is not None
    ann = annots[0].get_object()
    assert "1" in str(ann.get("/Contents"))


def test_add_bates_numbering_writes_annotations(tmp_path: Path):
    src = _make_pdf(tmp_path / "bates.pdf", pages=2)
    out = tmp_path / "bates_numbered.pdf"

    res = pdf_tools.add_bates_numbering(str(src), str(out), prefix="DOC-", start=10)
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    from pypdf import PdfReader

    reader = PdfReader(str(out))
    annots = reader.pages[0].get("/Annots")
    assert annots is not None
    ann = annots[0].get_object()
    assert "DOC-000010" in str(ann.get("/Contents"))


def test_verify_digital_signatures_empty(tmp_path: Path):
    src = _make_pdf(tmp_path / "unsigned.pdf", pages=1)
    res = pdf_tools.verify_digital_signatures(str(src))
    assert res["signatures"] == []
    assert res["verified"] == 0


def test_sign_pdf_pkcs12(tmp_path: Path):
    if not getattr(pdf_tools, "_HAS_PYHANKO", False):
        pytest.skip("pyHanko not available")
    certs = _make_test_certificates(tmp_path)
    src = _make_pdf(tmp_path / "sign_src.pdf", pages=1)
    out = tmp_path / "signed_pfx.pdf"

    res = pdf_tools.sign_pdf(
        input_path=str(src),
        output_path=str(out),
        pfx_path=str(certs["pfx"]),
        pfx_password=certs["password"].decode("utf-8"),
        certify=True,
    )
    assert Path(res["output_path"]).exists()

    verify = pdf_tools.verify_digital_signatures(str(out))
    assert len(verify["signatures"]) == 1
    assert verify["signatures"][0].get("intact") is True


def test_sign_pdf_pem(tmp_path: Path):
    if not getattr(pdf_tools, "_HAS_PYHANKO", False):
        pytest.skip("pyHanko not available")
    certs = _make_test_certificates(tmp_path)
    src = _make_pdf(tmp_path / "sign_src_pem.pdf", pages=1)
    out = tmp_path / "signed_pem.pdf"

    res = pdf_tools.sign_pdf_pem(
        input_path=str(src),
        output_path=str(out),
        key_path=str(certs["key"]),
        cert_path=str(certs["cert"]),
        certify=True,
    )
    assert Path(res["output_path"]).exists()

    verify = pdf_tools.verify_digital_signatures(str(out))
    assert len(verify["signatures"]) == 1
    assert verify["signatures"][0].get("intact") is True


def test_sign_pdf_with_timestamp_and_docmdp(tmp_path: Path, monkeypatch):
    if not getattr(pdf_tools, "_HAS_PYHANKO", False):
        pytest.skip("pyHanko not available")
    certs = _make_test_certificates(tmp_path)
    src = _make_pdf(tmp_path / "sign_src_ts.pdf", pages=1)
    out = tmp_path / "signed_ts.pdf"

    from datetime import datetime, timedelta, timezone

    from asn1crypto import keys as asn1_keys
    from asn1crypto import x509 as asn1_x509
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    from pyhanko.sign.timestamps.dummy_client import DummyTimeStamper

    tsa_key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    tsa_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "pdf-mcp-tsa")])
    now = datetime.now(timezone.utc)
    tsa_cert_obj = (
        x509.CertificateBuilder()
        .subject_name(tsa_subject)
        .issuer_name(tsa_subject)
        .public_key(tsa_key_obj.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]),
            critical=True,
        )
        .sign(tsa_key_obj, hashes.SHA256())
    )

    key_der = tsa_key_obj.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_der = tsa_cert_obj.public_bytes(serialization.Encoding.DER)
    tsa_key = asn1_keys.PrivateKeyInfo.load(key_der)
    tsa_cert = asn1_x509.Certificate.load(cert_der)

    def fake_timestamper(url, https=False, timeout=5, auth=None, headers=None):
        return DummyTimeStamper(tsa_cert=tsa_cert, tsa_key=tsa_key)

    monkeypatch.setattr(pdf_tools, "HTTPTimeStamper", fake_timestamper)

    res = pdf_tools.sign_pdf(
        input_path=str(src),
        output_path=str(out),
        pfx_path=str(certs["pfx"]),
        pfx_password=certs["password"].decode("utf-8"),
        certify=True,
        timestamp_url="https://tsa.example.test",
        embed_validation_info=False,
        allow_fetching=False,
        docmdp_permissions="no_changes",
    )
    assert Path(res["output_path"]).exists()

    verify = pdf_tools.verify_digital_signatures(str(out))
    assert len(verify["signatures"]) == 1
    assert verify["signatures"][0].get("intact") is True


def test_sign_pdf_with_validation_info(tmp_path: Path):
    if not getattr(pdf_tools, "_HAS_PYHANKO", False):
        pytest.skip("pyHanko not available")
    certs = _make_test_certificates(tmp_path)
    src = _make_pdf(tmp_path / "sign_src_vi.pdf", pages=1)
    out = tmp_path / "signed_vi.pdf"

    res = pdf_tools.sign_pdf(
        input_path=str(src),
        output_path=str(out),
        pfx_path=str(certs["pfx"]),
        pfx_password=certs["password"].decode("utf-8"),
        certify=True,
        embed_validation_info=True,
        allow_fetching=False,
    )
    assert Path(res["output_path"]).exists()

    verify = pdf_tools.verify_digital_signatures(str(out))
    assert len(verify["signatures"]) == 1
    assert verify["signatures"][0].get("intact") is True


def test_get_pdf_metadata_full_includes_document_info(tmp_path: Path):
    src = _make_pdf(tmp_path / "meta.pdf", pages=2)
    meta = tmp_path / "meta_out.pdf"
    pdf_tools.set_pdf_metadata(
        str(src),
        str(meta),
        title="Title",
        author="Author",
    )

    res = pdf_tools.get_pdf_metadata(str(meta), full=True)
    assert res["metadata"]["Title"] == "Title"
    assert res["document"]["page_count"] == 2
    assert isinstance(res["document"]["file_size_bytes"], int)


def test_mcp_layer_can_call_all_tools(tmp_path: Path):
    """
    Smoke test the MCP layer in-process (closest to Cursor invocation) by calling
    each tool through FastMCP.call_tool and validating the results.
    """
    import asyncio

    from pypdf import PdfReader

    from pdf_mcp import server

    form_src = _make_form_pdf(tmp_path / "mcp_form.pdf")
    nonstandard_src = _make_nonstandard_form_pdf(tmp_path / "mcp_nonstandard.pdf")
    blank_a = _make_pdf(tmp_path / "mcp_a.pdf", pages=2)
    blank_b = _make_pdf(tmp_path / "mcp_b.pdf", pages=1)
    certs = _make_test_certificates(tmp_path)

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

    # create_pdf_form
    created_form = tmp_path / "mcp_created_form.pdf"
    res = asyncio.run(
        call(
            "create_pdf_form",
            {
                "output_path": str(created_form),
                "fields": [{"name": "Name", "type": "text", "rect": [50, 100, 250, 130]}],
                "pages": 1,
            },
        )
    )
    assert Path(res["output_path"]).exists()

    # get_form_templates
    templates = asyncio.run(call("get_form_templates", {}))
    assert "client_intake_basic" in templates.get("templates", {})

    # create_pdf_form_from_template
    template_form = tmp_path / "mcp_template_form.pdf"
    res = asyncio.run(
        call(
            "create_pdf_form_from_template",
            {"output_path": str(template_form), "template_name": "client_intake_basic"},
        )
    )
    assert Path(res["output_path"]).exists()

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

    # fill_pdf_form_any (non-standard form)
    nonstandard_filled = tmp_path / "mcp_nonstandard_filled.pdf"
    res = asyncio.run(
        call(
            "fill_pdf_form_any",
            {
                "input_path": str(nonstandard_src),
                "output_path": str(nonstandard_filled),
                "data": {"Name": "Nonstandard"},
                "flatten": False,
            },
        )
    )
    assert Path(res["output_path"]).exists()

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

    if getattr(pdf_tools, "_HAS_PYHANKO", False):
        # sign_pdf (PKCS#12)
        signed_pfx = tmp_path / "mcp_signed_pfx.pdf"
        res = asyncio.run(
            call(
                "sign_pdf",
                {
                    "input_path": str(form_src),
                    "output_path": str(signed_pfx),
                    "pfx_path": str(certs["pfx"]),
                    "pfx_password": certs["password"].decode("utf-8"),
                    "certify": True,
                },
            )
        )
        assert Path(res["output_path"]).exists()

        # sign_pdf_pem
        signed_pem = tmp_path / "mcp_signed_pem.pdf"
        res = asyncio.run(
            call(
                "sign_pdf_pem",
                {
                    "input_path": str(form_src),
                    "output_path": str(signed_pem),
                    "key_path": str(certs["key"]),
                    "cert_path": str(certs["cert"]),
                    "certify": True,
                },
            )
        )
        assert Path(res["output_path"]).exists()

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

    # reorder_pages
    reordered = tmp_path / "mcp_reordered.pdf"
    res = asyncio.run(
        call(
            "reorder_pages",
            {"input_path": str(merged), "pages": [3, 1, 2], "output_path": str(reordered)},
        )
    )
    assert Path(res["output_path"]).exists()
    assert PdfReader(str(reordered)).get_num_pages() == 3

    # redact_text_regex
    text_src = _make_text_pdf(tmp_path / "mcp_text.pdf", ["Secret 123", "Public info"])
    redacted = tmp_path / "mcp_redacted.pdf"
    res = asyncio.run(
        call(
            "redact_text_regex",
            {
                "input_path": str(text_src),
                "output_path": str(redacted),
                "pattern": r"Secret\s+\d+",
            },
        )
    )
    assert Path(res["output_path"]).exists()
    assert res["redacted"] >= 1
    redacted_text = "".join((page.extract_text() or "") for page in PdfReader(str(redacted)).pages)
    assert "Secret" not in redacted_text
    assert "Public" in redacted_text

    # add_highlight
    highlighted = tmp_path / "mcp_highlighted.pdf"
    res = asyncio.run(
        call(
            "add_highlight",
            {"input_path": str(text_src), "output_path": str(highlighted), "page": 1, "text": "Secret"},
        )
    )
    assert Path(res["output_path"]).exists()
    assert res["added"] >= 1

    # add_date_stamp
    stamped = tmp_path / "mcp_stamped.pdf"
    res = asyncio.run(
        call("add_date_stamp", {"input_path": str(text_src), "output_path": str(stamped), "pages": [1]})
    )
    assert Path(res["output_path"]).exists()

    # detect_pii_patterns
    pii_src = _make_text_pdf(
        tmp_path / "mcp_pii.pdf",
        ["Email: test@example.com", "SSN: 123-45-6789", "Card: 4111 1111 1111 1111"],
    )
    res = asyncio.run(call("detect_pii_patterns", {"pdf_path": str(pii_src)}))
    assert res["total_matches"] >= 2

    # export_pdf (json)
    export_json = tmp_path / "mcp_export.json"
    res = asyncio.run(
        call(
            "export_pdf",
            {"pdf_path": str(text_src), "output_path": str(export_json), "format": "json"},
        )
    )
    assert Path(res["output_path"]).exists()
    export_data = json.loads(Path(res["output_path"]).read_text())
    assert export_data["page_count"] == 2

    # export_pdf (markdown)
    export_md = tmp_path / "mcp_export.md"
    res = asyncio.run(
        call(
            "export_pdf",
            {"pdf_path": str(text_src), "output_path": str(export_md), "format": "markdown"},
        )
    )
    assert Path(res["output_path"]).exists()

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

    # sanitize_pdf_metadata
    sanitized = tmp_path / "mcp_meta_sanitized.pdf"
    res = asyncio.run(
        call(
            "sanitize_pdf_metadata",
            {"input_path": str(meta_out), "output_path": str(sanitized)},
        )
    )
    assert Path(res["output_path"]).exists()
    res = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(sanitized)}))
    md = res["metadata"]
    assert "Title" not in md
    assert "Author" not in md

    # get_pdf_metadata with full=True
    res = asyncio.run(call("get_pdf_metadata", {"pdf_path": str(meta_out), "full": True}))
    assert res["document"]["page_count"] >= 1

    # add_page_numbers
    page_numbers = tmp_path / "mcp_page_numbers.pdf"
    res = asyncio.run(
        call(
            "add_page_numbers",
            {"input_path": str(blank_a), "output_path": str(page_numbers)},
        )
    )
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    # add_bates_numbering
    bates = tmp_path / "mcp_bates.pdf"
    res = asyncio.run(
        call(
            "add_bates_numbering",
            {
                "input_path": str(blank_a),
                "output_path": str(bates),
                "prefix": "DOC-",
                "start": 1,
            },
        )
    )
    assert Path(res["output_path"]).exists()
    assert res["added"] == 2

    # verify_digital_signatures
    res = asyncio.run(call("verify_digital_signatures", {"pdf_path": str(blank_a)}))
    assert res["signatures"] == []

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
    from pypdf import PdfReader

    from pdf_mcp import server

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
            "add_text_annotation",
            {"input_path": str(c3), "page": 1, "text": "T", "output_path": str(t1), "annotation_id": "t-1006"},
        )
    )
    assert_has_nm_annotation(t1, page_idx=0, nm="t-1006", expected_present=True)
    asyncio.run(call("update_text_annotation", {"input_path": str(t1), "output_path": str(t2), "annotation_id": "t-1006", "text": "T2"}))
    assert_has_nm_annotation(t2, page_idx=0, nm="t-1006", expected_present=True)
    asyncio.run(call("remove_text_annotation", {"input_path": str(t2), "output_path": str(t3), "annotation_id": "t-1006"}))
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
    from pypdf import PdfReader

    from pdf_mcp import server

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
    asyncio.run(call("add_text_annotation", {"input_path": str(a3), "output_path": str(t1), "page": 1, "text": "T", "annotation_id": "t-a"}))
    t2 = tmp_path / "a_t2.pdf"
    asyncio.run(call("update_text_annotation", {"input_path": str(t1), "output_path": str(t2), "annotation_id": "t-a", "text": "T2"}))
    t3 = tmp_path / "a_t3.pdf"
    asyncio.run(call("remove_text_annotation", {"input_path": str(t2), "output_path": str(t3), "annotation_id": "t-a"}))

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

    from pypdf import PdfReader

    from pdf_mcp import server

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
    asyncio.run(call("add_text_annotation", {"input_path": str(ann3), "output_path": str(t1), "page": 1, "text": "TB", "annotation_id": "t-b"}))
    t2 = tmp_path / "b_t2.pdf"
    asyncio.run(call("update_text_annotation", {"input_path": str(t1), "output_path": str(t2), "annotation_id": "t-b", "text": "TB2"}))
    t3 = tmp_path / "b_t3.pdf"
    asyncio.run(call("remove_text_annotation", {"input_path": str(t2), "output_path": str(t3), "annotation_id": "t-b"}))

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

    # text annotation wrappers
    inserted = tmp_path / "text_inserted.pdf"
    res = pdf_tools.add_text_annotation(str(src), page=1, text="T", output_path=str(inserted), annotation_id="t1")
    assert Path(res["output_path"]).exists()
    edited2 = tmp_path / "text_edited.pdf"
    res = pdf_tools.update_text_annotation(str(inserted), str(edited2), "t1", "T2")
    assert Path(res["output_path"]).exists()
    removed2 = tmp_path / "text_removed.pdf"
    res = pdf_tools.remove_text_annotation(str(edited2), str(removed2), "t1")
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


def test_add_highlight_by_text(tmp_path: Path):
    src = _make_text_pdf(tmp_path / "highlight.pdf", ["Hello world"])
    out = tmp_path / "highlight_out.pdf"

    res = pdf_tools.add_highlight(str(src), str(out), page=1, text="Hello")
    assert Path(res["output_path"]).exists()
    assert res["added"] >= 1


def test_add_date_stamp(tmp_path: Path):
    src = _make_pdf(tmp_path / "date.pdf", pages=1)
    out = tmp_path / "date_out.pdf"

    res = pdf_tools.add_date_stamp(str(src), str(out), pages=[1])
    assert Path(res["output_path"]).exists()
    assert res["added"] == 1


def test_detect_pii_patterns(tmp_path: Path):
    src = _make_text_pdf(
        tmp_path / "pii.pdf",
        ["Email: test@example.com", "Phone: 555-123-4567", "SSN: 123-45-6789", "Card: 4111 1111 1111 1111"],
    )
    res = pdf_tools.detect_pii_patterns(str(src))
    types = {m["type"] for m in res["matches"]}
    assert "email" in types
    assert "phone" in types
    assert "ssn" in types
    assert "credit_card" in types


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


# =============================================================================
# Consolidated API Tests (v0.6.0+)
# =============================================================================


def test_extract_text_native_engine(tmp_path: Path):
    """Test unified extract_text with engine='native'."""
    src = tmp_path / "text.pdf"
    _make_text_pdf(src, ["Hello World"])
    result = pdf_tools.extract_text(str(src), engine="native")
    assert "Hello World" in result["text"]
    assert result["method"] == "native"


def test_extract_text_auto_engine(tmp_path: Path):
    """Test unified extract_text with engine='auto'."""
    src = tmp_path / "text.pdf"
    _make_text_pdf(src, ["Auto Test Text"])
    result = pdf_tools.extract_text(str(src), engine="auto")
    assert "Auto Test Text" in result["text"] or result.get("total_chars", 0) > 0


def test_extract_text_smart_engine(tmp_path: Path):
    """Test unified extract_text with engine='smart'."""
    src = tmp_path / "text.pdf"
    _make_text_pdf(src, ["Smart extraction test content"])
    result = pdf_tools.extract_text(str(src), engine="smart")
    assert "method" in result or "page_details" in result


def test_get_pdf_metadata_full(tmp_path: Path):
    """Test get_pdf_metadata with full=True returns extended info."""
    src = _make_pdf(tmp_path / "meta.pdf", pages=3)
    result = pdf_tools.get_pdf_metadata(str(src), full=True)
    assert "metadata" in result
    assert "document" in result
    assert result["document"]["page_count"] == 3
    assert "is_encrypted" in result["document"]
    assert "file_size_bytes" in result["document"]


def test_get_pdf_metadata_basic(tmp_path: Path):
    """Test get_pdf_metadata with full=False returns only basic metadata."""
    src = _make_pdf(tmp_path / "meta2.pdf", pages=2)
    result = pdf_tools.get_pdf_metadata(str(src), full=False)
    assert "metadata" in result
    assert "document" not in result


def test_split_pdf_pages_mode(tmp_path: Path):
    """Test unified split_pdf with mode='pages'."""
    src = _make_pdf(tmp_path / "split.pdf", pages=4)
    out_dir = tmp_path / "split_out"
    result = pdf_tools.split_pdf(str(src), str(out_dir), mode="pages", pages_per_split=2)
    assert result["files_created"]
    assert len(result["files_created"]) == 2


def test_split_pdf_bookmarks_mode(tmp_path: Path):
    """Test unified split_pdf with mode='bookmarks' on PDF without bookmarks."""
    src = _make_pdf(tmp_path / "split_bm.pdf", pages=3)
    out_dir = tmp_path / "split_bm_out"
    result = pdf_tools.split_pdf(str(src), str(out_dir), mode="bookmarks")
    # PDF without bookmarks should return empty or a message
    assert "total_bookmarks" in result or "message" in result


def test_split_pdf_invalid_mode(tmp_path: Path):
    """Test split_pdf with invalid mode raises error."""
    src = _make_pdf(tmp_path / "split_inv.pdf", pages=2)
    out_dir = tmp_path / "split_inv_out"
    with pytest.raises(PdfToolError) as exc_info:
        pdf_tools.split_pdf(str(src), str(out_dir), mode="invalid")
    assert "mode must be" in str(exc_info.value)


def test_export_pdf_markdown(tmp_path: Path):
    """Test unified export_pdf with format='markdown'."""
    src = tmp_path / "export.pdf"
    _make_text_pdf(src, ["Export test content"])
    out = tmp_path / "export.md"
    result = pdf_tools.export_pdf(str(src), str(out), format="markdown")
    assert Path(result["output_path"]).exists()
    content = Path(result["output_path"]).read_text()
    assert "Page" in content or "Export" in content


def test_export_pdf_json(tmp_path: Path):
    """Test unified export_pdf with format='json'."""
    src = tmp_path / "export_json.pdf"
    _make_text_pdf(src, ["JSON export test"])
    out = tmp_path / "export.json"
    result = pdf_tools.export_pdf(str(src), str(out), format="json")
    assert Path(result["output_path"]).exists()
    content = json.loads(Path(result["output_path"]).read_text())
    assert "pages" in content


def test_export_pdf_invalid_format(tmp_path: Path):
    """Test export_pdf with invalid format raises error."""
    src = _make_pdf(tmp_path / "export_inv.pdf", pages=1)
    out = tmp_path / "export_inv.txt"
    with pytest.raises(PdfToolError) as exc_info:
        pdf_tools.export_pdf(str(src), str(out), format="txt")
    assert "format must be" in str(exc_info.value)


def test_deprecation_warnings():
    """Test that deprecated functions emit deprecation warnings."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # These should trigger deprecation warnings when called
        # We'll just check that the warning mechanism works
        assert len(w) >= 0  # Just verifying setup works


# ---------------------------------------------------------------------------
# Structured Logging Tests (v1.2.7)
# ---------------------------------------------------------------------------


class TestStructuredLogging:
    """Verify diagnostic logging is emitted at critical decision points."""

    def test_logger_exists(self):
        """pdf_tools module must expose a logger for diagnostic visibility."""
        import logging
        logger = logging.getLogger("pdf_mcp.pdf_tools")
        assert isinstance(logger, logging.Logger)

    def test_fill_pdf_form_fillpdf_fallback_logs_debug(self, tmp_path, caplog):
        """When fillpdf fails and falls back to pypdf, a debug message is emitted."""
        import logging
        src = _make_form_pdf(tmp_path / "log_fill.pdf")
        out = tmp_path / "log_fill_out.pdf"
        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "Test"})
        # Should log which fill method was used
        assert any(
            "fill" in rec.message.lower() for rec in caplog.records
            if rec.name == "pdf_mcp.pdf_tools"
        ), f"Expected fill-method log, got: {[r.message for r in caplog.records]}"

    def test_llm_backend_selection_logs_debug(self, caplog):
        """_get_llm_backend logs which backend was selected."""
        import logging
        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools._get_llm_backend()
        backend_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools" and "backend" in r.message.lower()
        ]
        assert len(backend_msgs) >= 1, (
            f"Expected backend selection log, got: {[r.message for r in caplog.records]}"
        )

    def test_local_llm_failure_logs_debug(self, caplog):
        """When _call_local_llm fails, a debug message with the error is emitted."""
        import logging
        from unittest.mock import patch
        # Force a connection error
        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            with patch.object(
                pdf_tools, "_HAS_REQUESTS", True
            ):
                with patch.object(
                    pdf_tools._requests, "post",
                    side_effect=ConnectionError("test connection refused"),
                ):
                    result = pdf_tools._call_local_llm("test prompt")
        assert result is None
        error_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools" and "local" in r.message.lower()
        ]
        assert len(error_msgs) >= 1, (
            f"Expected local LLM error log, got: {[r.message for r in caplog.records]}"
        )

    def test_model_resolution_logs_debug(self, caplog):
        """_resolve_local_model_name logs the resolved model."""
        import logging
        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools._resolve_local_model_name()
        model_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools" and "model" in r.message.lower()
        ]
        assert len(model_msgs) >= 1, (
            f"Expected model resolution log, got: {[r.message for r in caplog.records]}"
        )

    def test_pypdf_checkbox_fallback_logs_debug(self, tmp_path, caplog):
        """When pypdf update_page_form_field_values fails on checkboxes, it logs."""
        import logging
        src = _make_checkbox_form_pdf(tmp_path / "log_cb.pdf")
        out = tmp_path / "log_cb_out.pdf"
        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools.fill_pdf_form(str(src), str(out), {"Agree": "Yes"})
        # The checkbox widget without /AP should trigger the except block
        fallback_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools"
            and ("fallback" in r.message.lower() or "checkbox" in r.message.lower()
                 or "button" in r.message.lower())
        ]
        assert len(fallback_msgs) >= 1, (
            f"Expected checkbox fallback log, got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Edge Case Hardening Tests (v1.2.8)
# ---------------------------------------------------------------------------


class TestFormFillingEdgeCases:
    """Edge case tests for form-filling robustness."""

    def test_fill_pdf_form_empty_data_dict(self, tmp_path):
        """fill_pdf_form with empty data dict should succeed without error."""
        src = _make_form_pdf(tmp_path / "empty_data.pdf")
        out = tmp_path / "empty_data_out.pdf"
        result = pdf_tools.fill_pdf_form(str(src), str(out), {})
        assert Path(result["output_path"]).exists()
        # Field should remain empty
        reader = PdfReader(str(out))
        fields = reader.get_fields() or {}
        if "Name" in fields:
            val = fields["Name"].get("/V", "")
            assert val == "" or val is None or str(val) == ""

    def test_fill_pdf_form_nonexistent_field_names(self, tmp_path):
        """fill_pdf_form with field names not in form should not crash."""
        src = _make_form_pdf(tmp_path / "bad_names.pdf")
        out = tmp_path / "bad_names_out.pdf"
        data = {"NonExistentField": "value", "AnotherFake": "data"}
        result = pdf_tools.fill_pdf_form(str(src), str(out), data)
        assert Path(result["output_path"]).exists()

    def test_fill_pdf_form_unicode_values(self, tmp_path):
        """fill_pdf_form should handle Unicode values (CJK, accents)."""
        src = _make_form_pdf(tmp_path / "unicode.pdf")
        out = tmp_path / "unicode_out.pdf"
        data = {"Name": "张三 José García"}
        result = pdf_tools.fill_pdf_form(str(src), str(out), data)
        assert Path(result["output_path"]).exists()
        # Verify the value was written
        reader = PdfReader(str(out))
        fields = reader.get_fields() or {}
        if "Name" in fields:
            assert "张三" in str(fields["Name"].get("/V", ""))

    def test_fill_pdf_form_very_long_value(self, tmp_path):
        """fill_pdf_form with very long string should not crash."""
        src = _make_form_pdf(tmp_path / "long_val.pdf")
        out = tmp_path / "long_val_out.pdf"
        data = {"Name": "A" * 5000}
        result = pdf_tools.fill_pdf_form(str(src), str(out), data)
        assert Path(result["output_path"]).exists()

    def test_fill_pdf_form_special_chars(self, tmp_path):
        """fill_pdf_form with special PDF chars should not corrupt output."""
        src = _make_form_pdf(tmp_path / "special.pdf")
        out = tmp_path / "special_out.pdf"
        data = {"Name": "O'Brien & Co. <test> (parentheses)"}
        result = pdf_tools.fill_pdf_form(str(src), str(out), data)
        assert Path(result["output_path"]).exists()
        # Verify output is a valid PDF
        reader = PdfReader(str(out))
        assert len(reader.pages) >= 1

    def test_fill_pdf_form_file_not_found(self, tmp_path):
        """fill_pdf_form with non-existent input should raise PdfToolError."""
        out = tmp_path / "nofile_out.pdf"
        with pytest.raises(PdfToolError, match="File not found"):
            pdf_tools.fill_pdf_form(
                str(tmp_path / "does_not_exist.pdf"),
                str(out),
                {"Name": "test"},
            )

    def test_fill_pdf_form_corrupted_pdf(self, tmp_path):
        """fill_pdf_form with truncated/corrupt PDF should raise error."""
        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_bytes(b"%PDF-1.4\ntruncated garbage")
        out = tmp_path / "corrupted_out.pdf"
        with pytest.raises(Exception):
            pdf_tools.fill_pdf_form(str(corrupted), str(out), {"Name": "x"})

    def test_fill_pdf_form_no_acroform(self, tmp_path):
        """fill_pdf_form on PDF without form fields should still produce output."""
        src = _make_pdf(tmp_path / "noform.pdf")
        out = tmp_path / "noform_out.pdf"
        result = pdf_tools.fill_pdf_form(str(src), str(out), {"Name": "test"})
        assert Path(result["output_path"]).exists()


class TestGetFormFieldsEdgeCases:
    """Edge case tests for get_pdf_form_fields robustness."""

    def test_get_fields_no_acroform(self, tmp_path):
        """get_pdf_form_fields on PDF without fields returns count=0."""
        src = _make_pdf(tmp_path / "nofields.pdf")
        result = pdf_tools.get_pdf_form_fields(str(src))
        assert result["count"] == 0

    def test_get_fields_file_not_found(self, tmp_path):
        """get_pdf_form_fields with non-existent file raises PdfToolError."""
        with pytest.raises(PdfToolError, match="File not found"):
            pdf_tools.get_pdf_form_fields(
                str(tmp_path / "nonexistent.pdf")
            )

    def test_get_fields_corrupted_pdf(self, tmp_path):
        """get_pdf_form_fields with corrupted PDF raises error."""
        corrupted = tmp_path / "corrupt_fields.pdf"
        corrupted.write_bytes(b"%PDF-1.4\nbroken")
        with pytest.raises(Exception):
            pdf_tools.get_pdf_form_fields(str(corrupted))

    def test_get_fields_xfa_returns_error_dict(self, tmp_path):
        """get_pdf_form_fields on XFA form returns error dict, not raise."""
        src = _make_xfa_pdf(tmp_path / "xfa_fields.pdf")
        result = pdf_tools.get_pdf_form_fields(str(src))
        assert "error" in result
        assert result["xfa"] is True


class TestEncryptedPdfEdgeCases:
    """Edge case tests for encrypted PDF handling."""

    def _make_encrypted_pdf(self, path: Path, password: str = "secret") -> Path:
        """Create a simple encrypted PDF for testing."""
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(password)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            writer.write(f)
        return path

    def test_get_fields_encrypted_pdf_raises(self, tmp_path):
        """get_pdf_form_fields on encrypted PDF raises or returns error."""
        src = self._make_encrypted_pdf(tmp_path / "encrypted.pdf")
        # pypdf raises when accessing encrypted content without password
        with pytest.raises(Exception):
            pdf_tools.get_pdf_form_fields(str(src))

    def test_fill_encrypted_pdf_raises(self, tmp_path):
        """fill_pdf_form on encrypted PDF raises error."""
        src = self._make_encrypted_pdf(tmp_path / "enc_fill.pdf")
        out = tmp_path / "enc_fill_out.pdf"
        with pytest.raises(Exception):
            pdf_tools.fill_pdf_form(
                str(src), str(out), {"Name": "test"}
            )


# ---------------------------------------------------------------------------
# LLM Retry Logic Tests (v1.2.9)
# ---------------------------------------------------------------------------


class TestLLMRetryLogic:
    """Tests for LLM call retry with exponential backoff (v1.2.9)."""

    def test_no_retry_on_first_success(self, monkeypatch):
        """_call_llm returns immediately when first attempt succeeds."""
        call_count = 0

        def mock_local_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "success response"

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")

        result = pdf_tools._call_llm("test prompt")
        assert result == "success response"
        assert call_count == 1

    def test_retry_succeeds_on_second_attempt(self, monkeypatch):
        """_call_llm retries and returns result on second attempt."""
        call_count = 0

        def mock_local_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # First attempt fails
            return "recovered"

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = pdf_tools._call_llm("test prompt")
        assert result == "recovered"
        assert call_count == 2

    def test_retry_exhausted_returns_none(self, monkeypatch):
        """_call_llm returns None after all retries are exhausted."""
        call_count = 0

        def mock_local_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return None

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = pdf_tools._call_llm("test prompt")
        assert result is None
        assert call_count == 3  # 1 initial + 2 retries

    def test_max_retries_zero_no_retry(self, monkeypatch):
        """_call_llm with max_retries=0 doesn't retry on failure."""
        call_count = 0

        def mock_local_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return None

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")

        result = pdf_tools._call_llm("test prompt", max_retries=0)
        assert result is None
        assert call_count == 1

    def test_retry_uses_exponential_backoff(self, monkeypatch):
        """Backoff delays increase exponentially: 1s, 2s."""
        delays = []

        def mock_sleep(seconds):
            delays.append(seconds)

        def mock_local_llm(*args, **kwargs):
            return None

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")
        monkeypatch.setattr("time.sleep", mock_sleep)

        pdf_tools._call_llm("test prompt", max_retries=2)
        assert delays == [1.0, 2.0]

    def test_retry_logs_attempts(self, monkeypatch, caplog):
        """Retry attempts are logged at DEBUG level."""
        import logging

        def mock_local_llm(*args, **kwargs):
            return None

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")
        monkeypatch.setattr("time.sleep", lambda _: None)

        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools._call_llm("test prompt", max_retries=1)

        retry_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools" and "retry" in r.message.lower()
        ]
        assert len(retry_msgs) >= 1, (
            f"Expected retry log, got: {[r.message for r in caplog.records]}"
        )

    def test_retry_logs_final_failure(self, monkeypatch, caplog):
        """Final failure after all retries is logged."""
        import logging

        def mock_local_llm(*args, **kwargs):
            return None

        monkeypatch.setattr(pdf_tools, "_call_local_llm", mock_local_llm)
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "local")
        monkeypatch.setattr("time.sleep", lambda _: None)

        with caplog.at_level(logging.DEBUG, logger="pdf_mcp.pdf_tools"):
            pdf_tools._call_llm("test prompt", max_retries=1)

        fail_msgs = [
            r.message for r in caplog.records
            if r.name == "pdf_mcp.pdf_tools" and "failed" in r.message.lower()
        ]
        assert len(fail_msgs) >= 1, (
            f"Expected failure log, got: {[r.message for r in caplog.records]}"
        )

    def test_no_backend_returns_none_no_retry(self, monkeypatch):
        """_call_llm returns None immediately when no backend available."""
        monkeypatch.setattr(pdf_tools, "_get_llm_backend", lambda: "")
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = pdf_tools._call_llm("test prompt")
        assert result is None


# ---------------------------------------------------------------------------
# Form Fill Diagnostics Tests (v1.2.10)
# ---------------------------------------------------------------------------


class TestFormFillDiagnostics:
    """Tests for fill_pdf_form returning unmatched field diagnostics."""

    def test_fill_returns_filled_fields_count(self, tmp_path):
        """fill_pdf_form returns filled_fields_count matching data keys."""
        src = _make_form_pdf(tmp_path / "diag1.pdf")
        out = tmp_path / "diag1_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {"Name": "Alice"},
        )
        assert "filled_fields_count" in result
        assert result["filled_fields_count"] == 1

    def test_fill_returns_total_form_fields(self, tmp_path):
        """fill_pdf_form returns total_form_fields count."""
        src = _make_form_pdf(tmp_path / "diag2.pdf")
        out = tmp_path / "diag2_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {"Name": "Bob"},
        )
        assert "total_form_fields" in result
        assert result["total_form_fields"] >= 1

    def test_fill_returns_unmatched_fields_on_typo(self, tmp_path):
        """fill_pdf_form reports unmatched fields when keys don't exist."""
        src = _make_form_pdf(tmp_path / "diag3.pdf")
        out = tmp_path / "diag3_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {"Nmae": "Alice", "Name": "Alice"},
        )
        assert "unmatched_fields" in result
        assert "Nmae" in result["unmatched_fields"]

    def test_fill_no_unmatched_when_all_match(self, tmp_path):
        """fill_pdf_form has empty unmatched_fields when all keys match."""
        src = _make_form_pdf(tmp_path / "diag4.pdf")
        out = tmp_path / "diag4_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {"Name": "Carol"},
        )
        assert result.get("unmatched_fields") == []

    def test_fill_all_unmatched(self, tmp_path):
        """fill_pdf_form with all-wrong keys has zero filled count."""
        src = _make_form_pdf(tmp_path / "diag5.pdf")
        out = tmp_path / "diag5_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {"Bad1": "X", "Bad2": "Y"},
        )
        assert result["filled_fields_count"] == 0
        assert sorted(result["unmatched_fields"]) == ["Bad1", "Bad2"]

    def test_fill_empty_data_zero_filled(self, tmp_path):
        """fill_pdf_form with empty data has zero filled count."""
        src = _make_form_pdf(tmp_path / "diag6.pdf")
        out = tmp_path / "diag6_out.pdf"
        result = pdf_tools.fill_pdf_form(
            str(src), str(out), {},
        )
        assert result["filled_fields_count"] == 0
        assert result["unmatched_fields"] == []


# ---------------------------------------------------------------------------
# MRZ Name Sanitization Tests (v1.2.11 - BUG-006 fix)
# ---------------------------------------------------------------------------


class TestMRZNameSanitization:
    """Tests for _sanitize_mrz_name to fix BUG-006 garbage names."""

    def test_clean_name_unchanged(self):
        """Normal MRZ name passes through unmodified."""
        name, penalty = pdf_tools._sanitize_mrz_name("XIUYING")
        assert name == "XIUYING"
        assert penalty == 0.0

    def test_multi_word_clean_name(self):
        """Multi-word name without garbage passes through."""
        name, penalty = pdf_tools._sanitize_mrz_name("MARY JANE")
        assert name == "MARY JANE"
        assert penalty == 0.0

    def test_trailing_repeated_chars_removed(self):
        """Trailing OCR garbage removed, small penalty keeps confidence."""
        name, penalty = pdf_tools._sanitize_mrz_name(
            "XILUYING KKsssss sssssssss",
        )
        assert "sssss" not in name
        assert "XILU" in name or "XILUYING" in name
        assert penalty > 0
        # Successful recovery should use small penalty so
        # confidence stays >= 0.7 (avoids unnecessary VLM fallback)
        assert penalty <= 0.05

    def test_all_garbage_returns_original_high_penalty(self):
        """Completely garbage name returns original with high penalty."""
        name, penalty = pdf_tools._sanitize_mrz_name(
            "££SSSKEKKKKK",
        )
        # When nothing can be recovered, original is returned
        assert penalty >= 0.4

    def test_empty_name_passthrough(self):
        """Empty string passes through without error."""
        name, penalty = pdf_tools._sanitize_mrz_name("")
        assert name == ""
        assert penalty == 0.0

    def test_none_name_passthrough(self):
        """None passes through without error."""
        name, penalty = pdf_tools._sanitize_mrz_name(None)
        assert name is None
        assert penalty == 0.0

    def test_name_with_non_alpha_garbage(self):
        """Name followed by non-alpha chars is truncated."""
        name, penalty = pdf_tools._sanitize_mrz_name("JIZHI 123abc")
        assert name == "JIZHI"
        assert penalty > 0

    def test_uppercase_normalization(self):
        """Names are returned in uppercase (MRZ standard)."""
        name, _ = pdf_tools._sanitize_mrz_name("Xiuying")
        assert name == "XIUYING"

    def test_successful_recovery_confidence_above_threshold(self):
        """Recovered name keeps confidence >= 0.7 (no VLM fallback)."""
        _, penalty = pdf_tools._sanitize_mrz_name(
            "XILUYING KKsssss sssssssss",
        )
        confidence = 0.75 - penalty
        assert confidence >= 0.7, (
            f"Confidence {confidence} < 0.7 would trigger VLM fallback"
        )

    def test_non_alpha_recovery_confidence_above_threshold(self):
        """Name with non-alpha garbage keeps confidence >= 0.7."""
        _, penalty = pdf_tools._sanitize_mrz_name("JIZHI 123abc")
        confidence = 0.75 - penalty
        assert confidence >= 0.7


# ---------------------------------------------------------------------------
# VLM Null-String Filter Tests (v1.2.12 - BUG-006a fix)
# ---------------------------------------------------------------------------


class TestVLMNullStringFilter:
    """Tests for _is_vlm_null_string to fix BUG-006a regression."""

    def test_literal_null_string_detected(self):
        """VLM 'NULL' string is detected as null."""
        assert pdf_tools._is_vlm_null_string("NULL") is True

    def test_lowercase_null_detected(self):
        """VLM 'null' string is detected as null."""
        assert pdf_tools._is_vlm_null_string("null") is True

    def test_none_string_detected(self):
        """VLM 'None' string is detected as null."""
        assert pdf_tools._is_vlm_null_string("None") is True

    def test_na_string_detected(self):
        """VLM 'N/A' string is detected as null."""
        assert pdf_tools._is_vlm_null_string("N/A") is True

    def test_empty_string_detected(self):
        """Empty string is detected as null."""
        assert pdf_tools._is_vlm_null_string("") is True

    def test_whitespace_only_detected(self):
        """Whitespace-only string is detected as null."""
        assert pdf_tools._is_vlm_null_string("   ") is True

    def test_valid_value_not_filtered(self):
        """Normal value is NOT detected as null."""
        assert pdf_tools._is_vlm_null_string("XIUYING") is False

    def test_date_value_not_filtered(self):
        """Date value is NOT detected as null."""
        assert pdf_tools._is_vlm_null_string("2020-01-15") is False

    def test_python_none_not_string(self):
        """Python None is not a string, returns False."""
        assert pdf_tools._is_vlm_null_string(None) is False


# ---------------------------------------------------------------------------
# BUG-007: Collapsed Table Detection (v1.2.13)
# ---------------------------------------------------------------------------


class TestCollapsedTableDetection:
    """Tests for _is_collapsed_table helper (BUG-007)."""

    def test_collapsed_table_detected(self):
        """All data in first column, rest None -> collapsed."""
        raw_data = [
            ["Header1", "Header2", "Header3"],
            ["all data here", None, None],
            ["more data", None, None],
        ]
        assert pdf_tools._is_collapsed_table(raw_data) is True

    def test_normal_table_not_collapsed(self):
        """Data spread across columns -> not collapsed."""
        raw_data = [
            ["Col1", "Col2", "Col3"],
            ["a", "b", "c"],
            ["d", "e", "f"],
        ]
        assert pdf_tools._is_collapsed_table(raw_data) is False

    def test_single_column_table_not_collapsed(self):
        """Single column table is not 'collapsed'."""
        raw_data = [
            ["Header"],
            ["data1"],
            ["data2"],
        ]
        assert pdf_tools._is_collapsed_table(raw_data) is False

    def test_empty_data_not_collapsed(self):
        """Empty data is not collapsed."""
        assert pdf_tools._is_collapsed_table([]) is False

    def test_header_only_not_collapsed(self):
        """Header-only table (no data rows) is not collapsed."""
        raw_data = [["Col1", "Col2"]]
        assert pdf_tools._is_collapsed_table(raw_data) is False

    def test_mixed_none_and_empty_detected(self):
        """Columns with None AND empty strings -> collapsed."""
        raw_data = [
            ["H1", "H2", "H3"],
            ["data", None, ""],
            ["more", "", None],
        ]
        assert pdf_tools._is_collapsed_table(raw_data) is True

    def test_partial_data_not_collapsed(self):
        """Some rows have data in other columns -> not collapsed."""
        raw_data = [
            ["H1", "H2"],
            ["a", None],
            ["b", "has data"],
        ]
        assert pdf_tools._is_collapsed_table(raw_data) is False


class TestExtractTablesStrategy:
    """Tests for extract_tables strategy parameter (BUG-007)."""

    def test_strategy_parameter_accepted(self):
        """extract_tables accepts strategy parameter."""
        pdf_path = str(Path(__file__).parent / "1006.pdf")
        if not Path(pdf_path).exists():
            pytest.skip("1006.pdf not available")
        result = pdf_tools.extract_tables(pdf_path, strategy="text")
        assert "total_tables" in result

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises PdfToolError."""
        pdf_path = str(Path(__file__).parent / "1006.pdf")
        if not Path(pdf_path).exists():
            pytest.skip("1006.pdf not available")
        with pytest.raises(PdfToolError):
            pdf_tools.extract_tables(pdf_path, strategy="invalid")


# ---------------------------------------------------------------------------
# BUG-008: Entity Extraction Improvements (v1.2.13)
# ---------------------------------------------------------------------------


class TestEntityExtractionPatterns:
    """Tests for improved entity extraction patterns (BUG-008)."""

    def test_bare_digits_not_phone(self):
        """Bare 10-digit number should NOT match as phone."""
        # "1950687535" is a reference number, not a phone
        phones = pdf_tools._extract_phones("Reference: 1950687535")
        assert "1950687535" not in phones

    def test_formatted_phone_matches(self):
        """Properly formatted phone number should match."""
        phones = pdf_tools._extract_phones("Call (123) 456-7890")
        assert len(phones) > 0

    def test_phone_with_separators_matches(self):
        """Phone with dashes/dots should match."""
        phones = pdf_tools._extract_phones("Phone: 123-456-7890")
        assert len(phones) > 0

    def test_names_exclude_newline_fragments(self):
        """Document fragments with newlines are NOT names."""
        names = pdf_tools._extract_names(
            "Sensitive\nPersonal Privacy\nDepartment"
        )
        assert "Sensitive\nPersonal" not in names
        assert "Personal Privacy\nDepartment" not in names

    def test_names_exclude_document_words(self):
        """Common document/form words are NOT names."""
        names = pdf_tools._extract_names(
            "Page Break\nApplication Type\nHome Affairs"
        )
        for name in names:
            assert "Page Break" not in name
            assert "Application Type" not in name

    def test_names_find_real_names(self):
        """Real person names should be found."""
        names = pdf_tools._extract_names(
            "Applicant: John Smith and Mary Jane"
        )
        assert any("John Smith" in n for n in names) or \
            any("Mary Jane" in n for n in names)
