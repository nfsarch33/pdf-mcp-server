import argparse
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from pdf_mcp import pdf_tools

try:
    # PyMuPDF can emit noisy stderr warnings for some synthetic PDFs even when operations succeed.
    # Silence these for a cleaner smoke-test signal.
    import fitz  # type: ignore

    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
except Exception:
    pass


def _make_blank_pdf(path: Path, pages: int) -> Path:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=300, height=200)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        w.write(f)
    return path


def _write_test_png(path: Path) -> None:
    # Write a tiny valid PNG via Pillow if available; otherwise use a known-good prebuilt PNG.
    try:
        from PIL import Image  # type: ignore

        img = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
        img.save(path, format="PNG")
        return
    except Exception:
        pass

    # Known-good small PNG
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D494844520000000A0000000A08060000008D32CFBD"
            "0000000C49444154789C6360000002000157FE2A9B0000000049454E44AE426082"
        )
    )


def _make_form_pdf(path: Path) -> Path:
    # Minimal AcroForm with one text field "Name".
    w = PdfWriter()
    page = w.add_blank_page(width=300, height=200)

    from pypdf.generic import ArrayObject, BooleanObject, DictionaryObject, NameObject, NumberObject, TextStringObject

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = w._add_object(font)  # type: ignore[attr-defined]

    field = DictionaryObject(
        {
            NameObject("/FT"): NameObject("/Tx"),
            NameObject("/T"): TextStringObject("Name"),
            NameObject("/Ff"): NumberObject(0),
            NameObject("/V"): TextStringObject(""),
        }
    )
    field_ref = w._add_object(field)  # type: ignore[attr-defined]

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
    widget_ref = w._add_object(widget)  # type: ignore[attr-defined]

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
    w._root_object.update({NameObject("/AcroForm"): w._add_object(acro_form)})  # type: ignore[attr-defined]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        w.write(f)
    return path


def run_smoke(inputs_dir: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if inputs_dir is None:
        blank = _make_blank_pdf(out_dir / "blank.pdf", pages=2)
        form = _make_form_pdf(out_dir / "form.pdf")
        sig_png = out_dir / "sig.png"
        _write_test_png(sig_png)
    else:
        blank = inputs_dir / "blank.pdf"
        form = inputs_dir / "form.pdf"
        sig_png = inputs_dir / "sig.png"

    # Form: list -> fill -> clear
    fields = pdf_tools.get_pdf_form_fields(str(form))
    assert fields["count"] >= 1, fields
    assert "Name" in fields["fields"], fields

    filled = out_dir / "filled.pdf"
    pdf_tools.fill_pdf_form(str(form), str(filled), {"Name": "Test"}, flatten=False)
    r = PdfReader(str(filled))
    f = (r.get_fields() or {})
    assert str(f["Name"].get("/V")) == "Test"

    cleared = out_dir / "cleared.pdf"
    pdf_tools.clear_pdf_form_fields(str(filled), str(cleared), fields=["Name"])
    r2 = PdfReader(str(cleared))
    f2 = (r2.get_fields() or {})
    assert str(f2["Name"].get("/V")) == ""

    # Sign then encrypt hard requirement
    signed = out_dir / "signed.pdf"
    sig_res = pdf_tools.add_signature_image(
        str(cleared),
        str(signed),
        page=1,
        image_path=str(sig_png),
        rect=[50, 50, 150, 100],
    )
    assert int(sig_res["signature_xref"]) > 0

    encrypted = out_dir / "signed-encrypted.pdf"
    pdf_tools.encrypt_pdf(str(signed), str(encrypted), user_password="pw")
    er = PdfReader(str(encrypted))
    assert er.is_encrypted is True
    assert er.decrypt("pw") in (1, 2)
    _ = len(er.pages)

    # Touch a couple of other common tools quickly (sanity only).
    merged = out_dir / "merged.pdf"
    pdf_tools.merge_pdfs([str(blank), str(blank)], str(merged))
    assert PdfReader(str(merged)).get_num_pages() >= 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Cursor smoke test for pdf-handler tools")
    ap.add_argument("--inputs-dir", type=Path, default=None, help="Optional dir containing blank.pdf, form.pdf, sig.png")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (defaults to a temp dir)")
    args = ap.parse_args()

    if args.out_dir is None:
        with tempfile.TemporaryDirectory(prefix="pdf-handler-smoke-") as td:
            out = Path(td)
            run_smoke(args.inputs_dir, out)
            print(f"OK: smoke test passed. outputs at {out}")
    else:
        run_smoke(args.inputs_dir, args.out_dir)
        print(f"OK: smoke test passed. outputs at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

