from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject, TextStringObject

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


def _ensure_rect(rect: Optional[Sequence[float]]) -> ArrayObject:
    if rect is None:
        rect = (50, 50, 250, 100)
    if len(rect) != 4:
        raise PdfToolError("rect must contain exactly 4 numbers: [x1, y1, x2, y2]")
    return ArrayObject([NumberObject(float(x)) for x in rect])


def add_text_annotation(
    input_path: str,
    page: int,
    text: str,
    output_path: str,
    rect: Optional[Sequence[float]] = None,
    annotation_id: Optional[str] = None,
) -> Dict:
    """
    Add a FreeText annotation (managed text insert) to a page.

    This is used to provide a deterministic, testable way to insert text without
    editing PDF content streams.
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    if page == 0:
        raise PdfToolError("Page numbers must be 1-based")

    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_idx = _to_zero_based_pages([page], total)[0]

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    if not annotation_id:
        annotation_id = f"pdf-mcp-{secrets.token_hex(6)}"

    annot = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/FreeText"),
            NameObject("/Rect"): _ensure_rect(rect),
            NameObject("/Contents"): TextStringObject(text),
            # Name/identifier used to find and update this annotation later.
            NameObject("/NM"): TextStringObject(annotation_id),
            # Default appearance: Helvetica, size 12, black.
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
            NameObject("/F"): NumberObject(4),
        }
    )
    annot_ref = writer._add_object(annot)  # type: ignore[attr-defined]

    target_page = writer.pages[page_idx]
    existing = target_page.get("/Annots")
    if existing is None:
        annots = ArrayObject()
    else:
        existing_obj = existing.get_object() if hasattr(existing, "get_object") else existing
        annots = ArrayObject(list(existing_obj))

    annots.append(annot_ref)
    target_page[NameObject("/Annots")] = annots

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "annotation_id": annotation_id, "page": page}


def update_text_annotation(
    input_path: str,
    output_path: str,
    annotation_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = (
        _to_zero_based_pages(pages, total) if pages else list(range(total))
    )

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    updated = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        annots = page_obj.get("/Annots")
        if not annots:
            continue
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        for ref in list(annots_obj):
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            if str(obj.get("/NM")) == annotation_id:
                obj[NameObject("/Contents")] = TextStringObject(text)
                updated += 1

    if updated == 0:
        raise PdfToolError(f"Annotation not found: {annotation_id}")

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "updated": updated, "annotation_id": annotation_id}


def remove_text_annotation(
    input_path: str,
    output_path: str,
    annotation_id: str,
    pages: Optional[List[int]] = None,
) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = (
        _to_zero_based_pages(pages, total) if pages else list(range(total))
    )

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    removed = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        annots = page_obj.get("/Annots")
        if not annots:
            continue
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        new_refs = []
        for ref in list(annots_obj):
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            if str(obj.get("/NM")) == annotation_id:
                removed += 1
                continue
            new_refs.append(ref)
        page_obj[NameObject("/Annots")] = ArrayObject(new_refs)

    if removed == 0:
        raise PdfToolError(f"Annotation not found: {annotation_id}")

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "removed": removed, "annotation_id": annotation_id}


def remove_annotations(
    input_path: str,
    output_path: str,
    pages: List[int],
    subtype: Optional[str] = None,
) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if not pages:
        raise PdfToolError("No pages specified for annotation removal")

    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = _to_zero_based_pages(pages, total)

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    removed = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        annots = page_obj.get("/Annots")
        if not annots:
            continue
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        if subtype is None:
            removed += len(list(annots_obj))
            page_obj[NameObject("/Annots")] = ArrayObject()
            continue

        target_subtype = f"/{subtype.lstrip('/')}"
        new_refs = []
        for ref in list(annots_obj):
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            if str(obj.get("/Subtype")) == target_subtype:
                removed += 1
                continue
            new_refs.append(ref)
        page_obj[NameObject("/Annots")] = ArrayObject(new_refs)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "removed": removed}


def insert_pages(
    input_path: str,
    insert_from_path: str,
    at_page: int,
    output_path: str,
) -> Dict:
    src = _ensure_file(input_path)
    ins = _ensure_file(insert_from_path)
    dst = _prepare_output(output_path)

    if at_page <= 0:
        raise PdfToolError("at_page must be 1-based")

    reader = PdfReader(str(src))
    insert_reader = PdfReader(str(ins))

    total = len(reader.pages)
    insert_total = len(insert_reader.pages)
    if insert_total == 0:
        raise PdfToolError("insert_from_path has no pages")

    # Allow inserting at end: at_page == total + 1
    if at_page > total + 1:
        raise PdfToolError(f"at_page is out of range (1-{total + 1})")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    idx = at_page - 1
    for page_obj in insert_reader.pages:
        writer.insert_page(page_obj, idx)
        idx += 1

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {
        "output_path": str(dst),
        "inserted": insert_total,
        "at_page": at_page,
        "total_pages": len(writer.pages),
    }


def remove_pages(input_path: str, pages: List[int], output_path: str) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if not pages:
        raise PdfToolError("No pages specified for removal")

    reader = PdfReader(str(src))
    total = len(reader.pages)
    zero_based = _to_zero_based_pages(pages, total)
    if len(zero_based) == total:
        raise PdfToolError("Refusing to remove all pages")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    for idx in sorted(zero_based, reverse=True):
        writer.remove_page(idx)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "removed": len(zero_based), "total_pages": len(writer.pages)}


# Text insert/edit/remove: implemented via managed FreeText annotations.
def insert_text(
    input_path: str,
    page: int,
    text: str,
    output_path: str,
    rect: Optional[Sequence[float]] = None,
    text_id: Optional[str] = None,
) -> Dict:
    return add_text_annotation(input_path, page, text, output_path, rect=rect, annotation_id=text_id)


def edit_text(
    input_path: str,
    output_path: str,
    text_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict:
    return update_text_annotation(input_path, output_path, text_id, text, pages=pages)


def remove_text(
    input_path: str,
    output_path: str,
    text_id: str,
    pages: Optional[List[int]] = None,
) -> Dict:
    return remove_text_annotation(input_path, output_path, text_id, pages=pages)

