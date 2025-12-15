from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from pypdf import PdfReader, PdfWriter

try:
    from fillpdf import fillpdfs

    _HAS_FILLPDF = True
except ImportError:
    _HAS_FILLPDF = False


@dataclass
class PdfToolError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _ensure_file(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    if not resolved.exists():
        raise PdfToolError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise PdfToolError(f"Not a file: {resolved}")
    return resolved


def _prepare_output(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _simplify_fields(raw_fields: Dict) -> Dict:
    simplified: Dict[str, Dict] = {}
    for name, field in (raw_fields or {}).items():
        simplified[name] = {
            "value": _safe_value(field.get("/V")),
            "type": _safe_value(field.get("/FT")),
            "alternate_name": _safe_value(field.get("/T")),
            "flags": _safe_value(field.get("/Ff")),
        }
    return simplified


def _safe_value(value):
    try:
        if hasattr(value, "get_object"):
            value = value.get_object()
    except Exception:
        pass
    if value is None:
        return None
    return str(value)


def _flatten_writer(writer: PdfWriter) -> None:
    # Remove annotations and form field structures so the document is no longer editable.
    for page in writer.pages:
        if "/Annots" in page:
            page["/Annots"] = []
    acro_form = writer._root_object.get("/AcroForm")  # type: ignore[attr-defined]
    if acro_form:
        acro_form.pop("/Fields", None)
        acro_form.update({"/NeedAppearances": False})


def get_pdf_form_fields(pdf_path: str) -> Dict:
    path = _ensure_file(pdf_path)
    reader = PdfReader(str(path))
    fields = reader.get_fields()
    return {"fields": _simplify_fields(fields), "count": len(fields or {})}


def fill_pdf_form(
    input_path: str,
    output_path: str,
    data: Dict[str, str],
    flatten: bool = False,
) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    has_fields = bool(reader.get_fields())

    if _HAS_FILLPDF and has_fields:
        # Prefer fillpdf when possible for robust form filling on real AcroForm PDFs.
        fillpdfs.write_fillable_pdf(str(src), str(dst), data)
        if flatten:
            fillpdfs.flatten_pdf(str(dst), str(dst))
        return {"output_path": str(dst), "flattened": flatten, "filled_with": "fillpdf"}

    writer = PdfWriter()
    # Important: When updating form fields with pypdf, the PdfWriter must have
    # the document's /AcroForm dictionary. Cloning the document preserves it.
    writer.clone_document_from_reader(reader)
    if has_fields:
        for page in writer.pages:
            writer.update_page_form_field_values(page, data)

    if flatten:
        _flatten_writer(writer)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "flattened": flatten, "filled_with": "pypdf"}


def flatten_pdf(input_path: str, output_path: str) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    if _HAS_FILLPDF:
        fillpdfs.flatten_pdf(str(src), str(dst))
        return {"output_path": str(dst), "flattened_with": "fillpdf"}

    reader = PdfReader(str(src))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    _flatten_writer(writer)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "flattened_with": "pypdf"}


def merge_pdfs(pdf_list: Iterable[str], output_path: str) -> Dict:
    paths: List[Path] = [_ensure_file(p) for p in pdf_list]
    if not paths:
        raise PdfToolError("No input PDFs provided for merge")

    dst = _prepare_output(output_path)
    writer = PdfWriter()

    for pdf in paths:
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "merged": len(paths)}


def extract_pages(input_path: str, pages: List[int], output_path: str) -> Dict:
    src = _ensure_file(input_path)
    if not pages:
        raise PdfToolError("No pages specified for extraction")

    reader = PdfReader(str(src))
    total = len(reader.pages)
    zero_based = _to_zero_based_pages(pages, total)

    dst = _prepare_output(output_path)
    writer = PdfWriter()
    for idx in zero_based:
        writer.add_page(reader.pages[idx])

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "extracted": len(zero_based)}


def rotate_pages(
    input_path: str,
    pages: List[int],
    degrees: int,
    output_path: str,
) -> Dict:
    if degrees % 90 != 0:
        raise PdfToolError("Rotation degrees must be a multiple of 90")

    src = _ensure_file(input_path)
    reader = PdfReader(str(src))
    total = len(reader.pages)
    zero_based = _to_zero_based_pages(pages, total)

    dst = _prepare_output(output_path)
    writer = PdfWriter()

    for idx, page in enumerate(reader.pages):
        page_copy = page
        if idx in zero_based:
            page_copy = page_copy.rotate(degrees)
        writer.add_page(page_copy)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "rotated": len(zero_based), "degrees": degrees}


def _to_zero_based_pages(pages: List[int], total: int) -> List[int]:
    converted: List[int] = []
    for page in pages:
        if page == 0:
            raise PdfToolError("Page numbers must be 1-based")
        idx = page - 1 if page > 0 else total + page
        if idx < 0 or idx >= total:
            raise PdfToolError(f"Page {page} is out of range (1-{total})")
        converted.append(idx)
    return sorted(set(converted))

