from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Any

from pypdf import PdfReader, PdfWriter
from pypdf.constants import UserAccessPermissions
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
import pymupdf

try:
    from fillpdf import fillpdfs

    _HAS_FILLPDF = True
except ImportError:
    _HAS_FILLPDF = False

try:
    import pytesseract
    from PIL import Image
    import io

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


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
    annots_key = NameObject("/Annots")
    for page in writer.pages:
        if annots_key in page:
            page[annots_key] = ArrayObject()
    acro_form = writer._root_object.get(NameObject("/AcroForm"))  # type: ignore[attr-defined]
    if hasattr(acro_form, "get_object"):
        try:
            acro_form = acro_form.get_object()
        except Exception:
            pass
    if acro_form:
        try:
            acro_form.pop(NameObject("/Fields"), None)
        except Exception:
            # Defensive: some PDFs may store keys as plain strings.
            acro_form.pop("/Fields", None)
        acro_form[NameObject("/NeedAppearances")] = BooleanObject(False)


def _apply_form_field_values(writer: PdfWriter, data: Dict[str, str]) -> int:
    """
    Best-effort form filling that handles both typical AcroForm structures and
    less standard PDFs where widgets are missing /Subtype or are merged into fields.
    """

    def _apply_to_obj(obj) -> int:
        updated_local = 0
        try:
            field_name = obj.get("/T")
        except Exception:
            field_name = None

        if field_name is not None:
            key = str(field_name)
            if key in data:
                obj[NameObject("/V")] = TextStringObject(str(data[key]))
                updated_local += 1

        kids = obj.get("/Kids")
        if kids:
            kids_obj = kids.get_object() if hasattr(kids, "get_object") else kids
            for kid in list(kids_obj):
                kobj = kid.get_object() if hasattr(kid, "get_object") else kid
                updated_local += _apply_to_obj(kobj)
        return updated_local

    updated = 0

    # Update AcroForm fields array if present.
    acro_form = writer._root_object.get("/AcroForm")  # type: ignore[attr-defined]
    if acro_form:
        try:
            acro_form[NameObject("/NeedAppearances")] = BooleanObject(True)
        except Exception:
            pass
        fields_arr = acro_form.get("/Fields")
        if fields_arr:
            fields_obj = fields_arr.get_object() if hasattr(fields_arr, "get_object") else fields_arr
            for ref in list(fields_obj):
                obj = ref.get_object() if hasattr(ref, "get_object") else ref
                updated += _apply_to_obj(obj)

    # Update page annotations (widgets), even if they're not well-formed.
    for page in writer.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        annots_obj = annots.get_object() if hasattr(annots, "get_object") else annots
        for ref in list(annots_obj):
            obj = ref.get_object() if hasattr(ref, "get_object") else ref
            if not hasattr(obj, "get"):
                continue
            t = obj.get("/T")
            if t is None:
                continue
            key = str(t)
            if key in data:
                obj[NameObject("/V")] = TextStringObject(str(data[key]))
                updated += 1

    return updated


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
        try:
            fillpdfs.write_fillable_pdf(str(src), str(dst), data)
            if flatten:
                fillpdfs.flatten_pdf(str(dst), str(dst))
        except Exception:
            # fillpdf uses pdfrw which can fail on PDFs with compressed object streams
            # (common in some Adobe InDesign exports). Fall back to pypdf path below.
            pass
        # Some real-world PDFs get their appearances updated but don't persist /V values.
        # Verify and fall back to pypdf if needed so that filled contents are durable.
        if not flatten:
            try:
                verify_reader = PdfReader(str(dst))
                verify_fields = verify_reader.get_fields() or {}
                mismatched = []
                for k, v in data.items():
                    if k not in verify_fields:
                        continue
                    actual = _safe_value(verify_fields[k].get("/V"))
                    if actual != str(v):
                        mismatched.append(k)
                if mismatched:
                    raise PdfToolError(
                        "fillpdf did not persist field values for: " + ", ".join(mismatched)
                    )
            except PdfToolError:
                # Fall back to pypdf path below.
                pass
            except Exception:
                # Defensive: don't fail the operation just due to verification issues.
                pass
            else:
                return {"output_path": str(dst), "flattened": flatten, "filled_with": "fillpdf"}

    writer = PdfWriter()
    # Important: When updating form fields with pypdf, the PdfWriter must have
    # the document's /AcroForm dictionary. Cloning the document preserves it.
    writer.clone_document_from_reader(reader)
    if has_fields:
        for page in writer.pages:
            writer.update_page_form_field_values(page, data)
        _apply_form_field_values(writer, data)

    if flatten:
        _flatten_writer(writer)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "flattened": flatten, "filled_with": "pypdf"}


def clear_pdf_form_fields(
    input_path: str,
    output_path: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Clear (delete) values for form fields by setting them to an empty string.

    This keeps the AcroForm structure intact (fields remain fillable). To remove
    fields entirely, use flattening (which removes editability).
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    available = list((reader.get_fields() or {}).keys())
    if not available:
        raise PdfToolError("No form fields found in PDF")

    target = available if fields is None else fields
    missing = [f for f in target if f not in available]
    if missing:
        raise PdfToolError(f"Unknown form fields: {', '.join(missing)}")

    # Delegate to the existing fill logic for maximum reuse.
    data = {name: "" for name in target}
    result = fill_pdf_form(str(src), str(dst), data, flatten=False)
    result.update({"cleared": len(target), "fields": target})
    return result


def encrypt_pdf(
    input_path: str,
    output_path: str,
    user_password: str,
    owner_password: Optional[str] = None,
    allow_printing: bool = True,
    allow_modifying: bool = False,
    allow_copying: bool = False,
    allow_annotations: bool = False,
    allow_form_filling: bool = True,
    use_128bit: bool = True,
) -> Dict:
    """
    Encrypt (password-protect) a PDF using pypdf.

    Note: This is PDF encryption (access control). It is not a cryptographic
    digital signature. Use add_signature_image for a visual signature, then
    encrypt_pdf to protect the signed PDF.
    """
    if not user_password:
        raise PdfToolError("user_password must be non-empty")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    # Some PDFs carry a trailer /ID as TextStringObject(s). pypdf encryption expects bytes-like IDs.
    # Normalize by generating a fresh byte-string ID pair.
    try:
        writer._ID = [  # type: ignore[attr-defined]
            ByteStringObject(secrets.token_bytes(16)),
            ByteStringObject(secrets.token_bytes(16)),
        ]
    except Exception:
        pass

    perms = UserAccessPermissions(0)
    if allow_printing:
        perms |= UserAccessPermissions.PRINT
        perms |= UserAccessPermissions.PRINT_TO_REPRESENTATION
    if allow_modifying:
        perms |= UserAccessPermissions.MODIFY
        perms |= UserAccessPermissions.ASSEMBLE_DOC
    if allow_copying:
        perms |= UserAccessPermissions.EXTRACT
        perms |= UserAccessPermissions.EXTRACT_TEXT_AND_GRAPHICS
    if allow_annotations:
        perms |= UserAccessPermissions.ADD_OR_MODIFY
    if allow_form_filling:
        perms |= UserAccessPermissions.FILL_FORM_FIELDS

    writer.encrypt(
        user_password=user_password,
        owner_password=owner_password,
        use_128bit=use_128bit,
        permissions_flag=perms,
    )

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {
        "output_path": str(dst),
        "encrypted": True,
        "use_128bit": use_128bit,
        "permissions": int(perms),
        "owner_password_provided": owner_password is not None,
    }


def flatten_pdf(input_path: str, output_path: str) -> Dict:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    if _HAS_FILLPDF:
        try:
            fillpdfs.flatten_pdf(str(src), str(dst))
            return {"output_path": str(dst), "flattened_with": "fillpdf"}
        except Exception:
            # fillpdf uses pdfrw which can fail on PDFs with compressed object streams
            # (common in some Adobe InDesign exports). Fall back to pypdf below.
            pass

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


def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Return basic document metadata (title, author, etc.) if present."""
    path = _ensure_file(pdf_path)
    reader = PdfReader(str(path))
    md = reader.metadata or {}
    # pypdf metadata keys can be like "/Title", "/Author". Normalize to plain keys.
    normalized: Dict[str, Any] = {}
    for k, v in dict(md).items():
        key = str(k)
        if key.startswith("/"):
            key = key[1:]
        normalized[key] = None if v is None else str(v)
    return {"metadata": normalized}


def set_pdf_metadata(
    input_path: str,
    output_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    subject: Optional[str] = None,
    keywords: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set basic PDF document metadata.

    Only provided fields are updated; unspecified fields are preserved when possible.
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    existing = reader.metadata or {}
    merged: Dict[str, str] = {}
    for k, v in dict(existing).items():
        if v is None:
            continue
        key = str(k)
        if not key.startswith("/"):
            key = f"/{key}"
        merged[key] = str(v)

    if title is not None:
        merged["/Title"] = title
    if author is not None:
        merged["/Author"] = author
    if subject is not None:
        merged["/Subject"] = subject
    if keywords is not None:
        merged["/Keywords"] = keywords

    if merged:
        writer.add_metadata(merged)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "updated": {k: v for k, v in {"title": title, "author": author, "subject": subject, "keywords": keywords}.items() if v is not None}}


def add_text_watermark(
    input_path: str,
    output_path: str,
    text: str,
    pages: Optional[List[int]] = None,
    rect: Optional[Sequence[float]] = None,
    annotation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a simple text watermark or stamp using FreeText annotations.

    This is intentionally implemented as annotations (KISS, deterministic, testable),
    not by rewriting content streams.
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    reader = PdfReader(str(src))
    total = len(reader.pages)

    page_indices = _to_zero_based_pages(pages, total) if pages else list(range(total))
    if not page_indices:
        raise PdfToolError("No pages selected for watermark")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    if not annotation_id:
        annotation_id = f"pdf-mcp-watermark-{secrets.token_hex(6)}"

    rect_obj = _ensure_rect(rect)

    annot = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/FreeText"),
            NameObject("/Rect"): rect_obj,
            NameObject("/Contents"): TextStringObject(text),
            NameObject("/NM"): TextStringObject(annotation_id),
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
            NameObject("/F"): NumberObject(4),
        }
    )
    annot_ref = writer._add_object(annot)  # type: ignore[attr-defined]

    added = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        existing = page_obj.get("/Annots")
        if existing is None:
            annots = ArrayObject()
        else:
            existing_obj = existing.get_object() if hasattr(existing, "get_object") else existing
            annots = ArrayObject(list(existing_obj))
        annots.append(annot_ref)
        page_obj[NameObject("/Annots")] = annots
        added += 1

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "annotation_id": annotation_id, "added": added}


def add_comment(
    input_path: str,
    output_path: str,
    page: int,
    text: str,
    pos: Sequence[float],
    comment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a PDF comment (Subtype /Text) using PyMuPDF."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if page < 1:
        raise PdfToolError("page must be >= 1")
    if len(pos) != 2:
        raise PdfToolError("pos must be [x, y]")

    if not comment_id:
        comment_id = f"pdf-mcp-comment-{secrets.token_hex(6)}"

    doc = pymupdf.open(str(src))
    try:
        if page > doc.page_count:
            raise PdfToolError(f"page out of range: {page}")
        p = doc.load_page(page - 1)
        annot = p.add_text_annot(pymupdf.Point(pos[0], pos[1]), text)
        annot.set_name(comment_id)
        annot.update()
        doc.save(str(dst), garbage=4, deflate=True)
    finally:
        doc.close()

    return {"output_path": str(dst), "comment_id": comment_id, "page": page}


def update_comment(
    input_path: str,
    output_path: str,
    comment_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Update a PDF comment by id using PyMuPDF."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if not comment_id:
        raise PdfToolError("comment_id is required")

    doc = pymupdf.open(str(src))
    try:
        page_indices = _to_zero_based_pages(pages, doc.page_count) if pages else list(range(doc.page_count))
        updated = 0
        for idx in page_indices:
            p = doc.load_page(idx)
            for annot in p.annots() or []:
                if annot.info.get("name") == comment_id:
                    annot.set_info(content=text)
                    annot.update()
                    updated += 1
        doc.save(str(dst), garbage=4, deflate=True)
    finally:
        doc.close()

    if updated == 0:
        raise PdfToolError(f"comment not found: {comment_id}")
    return {"output_path": str(dst), "updated": updated}


def remove_comment(
    input_path: str,
    output_path: str,
    comment_id: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Remove a PDF comment by id using PyMuPDF."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if not comment_id:
        raise PdfToolError("comment_id is required")

    doc = pymupdf.open(str(src))
    try:
        page_indices = _to_zero_based_pages(pages, doc.page_count) if pages else list(range(doc.page_count))
        removed = 0
        for idx in page_indices:
            p = doc.load_page(idx)
            for annot in list(p.annots() or []):
                if annot.info.get("name") == comment_id:
                    p.delete_annot(annot)
                    removed += 1
        doc.save(str(dst), garbage=4, deflate=True)
    finally:
        doc.close()

    if removed == 0:
        raise PdfToolError(f"comment not found: {comment_id}")
    return {"output_path": str(dst), "removed": removed}


def add_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    image_path: str,
    rect: Sequence[float],
) -> Dict[str, Any]:
    """Add a signature image by inserting an image onto a page (returns xref)."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    img = _ensure_file(image_path)
    if page < 1:
        raise PdfToolError("page must be >= 1")
    if len(rect) != 4:
        raise PdfToolError("rect must be [x0, y0, x1, y1]")

    doc = pymupdf.open(str(src))
    try:
        if page > doc.page_count:
            raise PdfToolError(f"page out of range: {page}")
        p = doc.load_page(page - 1)
        xref = p.insert_image(pymupdf.Rect(rect[0], rect[1], rect[2], rect[3]), filename=str(img))
        # Keep xref stable for downstream update/remove by saving without garbage collection.
        doc.save(str(dst), deflate=True)
    finally:
        doc.close()

    return {"output_path": str(dst), "signature_xref": int(xref), "page": page}


def update_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    signature_xref: int,
    image_path: Optional[str] = None,
    rect: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Update or resize a signature image. If rect is provided, the image is reinserted and may get a new xref."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if page < 1:
        raise PdfToolError("page must be >= 1")
    if signature_xref <= 0:
        raise PdfToolError("signature_xref must be > 0")
    img_path = _ensure_file(image_path) if image_path else None
    if rect is not None and len(rect) != 4:
        raise PdfToolError("rect must be [x0, y0, x1, y1]")

    doc = pymupdf.open(str(src))
    try:
        if page > doc.page_count:
            raise PdfToolError(f"page out of range: {page}")
        p = doc.load_page(page - 1)

        new_xref = int(signature_xref)
        if rect is None:
            if img_path is None:
                raise PdfToolError("Either image_path or rect must be provided")
            p.replace_image(signature_xref, filename=str(img_path))
        else:
            # We need to reinsert at a new rectangle. If no new image is provided, reuse existing image bytes.
            if img_path is None:
                extracted = doc.extract_image(signature_xref)
                stream = extracted.get("image")
                if not stream:
                    raise PdfToolError(f"Could not extract existing image for xref: {signature_xref}")
                p.delete_image(signature_xref)
                new_xref = p.insert_image(pymupdf.Rect(rect[0], rect[1], rect[2], rect[3]), stream=stream)
            else:
                p.delete_image(signature_xref)
                new_xref = p.insert_image(pymupdf.Rect(rect[0], rect[1], rect[2], rect[3]), filename=str(img_path))

        # Keep xref stable for downstream update/remove by saving without garbage collection.
        doc.save(str(dst), deflate=True)
    finally:
        doc.close()

    return {"output_path": str(dst), "signature_xref": int(new_xref), "page": page}


def remove_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    signature_xref: int,
) -> Dict[str, Any]:
    """Remove a signature image by xref."""
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    if page < 1:
        raise PdfToolError("page must be >= 1")
    if signature_xref <= 0:
        raise PdfToolError("signature_xref must be > 0")

    doc = pymupdf.open(str(src))
    try:
        if page > doc.page_count:
            raise PdfToolError(f"page out of range: {page}")
        p = doc.load_page(page - 1)
        p.delete_image(signature_xref)
        # For removals, run garbage collection to drop now-unused objects when possible.
        doc.save(str(dst), garbage=4, deflate=True)
    finally:
        doc.close()

    return {"output_path": str(dst), "removed": 1, "page": page}


# =============================================================================
# OCR and Text Extraction Tools
# =============================================================================


def detect_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze a PDF to classify its content type.

    Returns classification:
    - "searchable": PDF has native text layer (text can be selected/copied)
    - "image_based": PDF consists primarily of images with no/minimal text layer
    - "hybrid": PDF has both native text and significant image content

    Also returns detailed metrics for each page.
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        page_analyses: List[Dict[str, Any]] = []
        total_native_chars = 0
        total_images = 0
        pages_with_text = 0
        pages_with_images = 0

        for page_num in range(total_pages):
            page = doc.load_page(page_num)

            # Extract native text
            native_text = page.get_text("text")
            native_char_count = len(native_text.strip())

            # Count images on page
            images = page.get_images(full=True)
            image_count = len(images)

            # Calculate image coverage (approximate)
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            image_area = 0.0
            for img in images:
                try:
                    xref = img[0]
                    img_rects = page.get_image_rects(xref)
                    for rect in img_rects:
                        image_area += rect.width * rect.height
                except Exception:
                    pass
            image_coverage = min(image_area / page_area, 1.0) if page_area > 0 else 0.0

            page_analysis = {
                "page": page_num + 1,
                "native_char_count": native_char_count,
                "image_count": image_count,
                "image_coverage": round(image_coverage, 3),
                "has_native_text": native_char_count > 50,  # threshold for meaningful text
                "is_primarily_image": image_coverage > 0.5 and native_char_count < 100,
            }
            page_analyses.append(page_analysis)

            total_native_chars += native_char_count
            total_images += image_count
            if native_char_count > 50:
                pages_with_text += 1
            if image_count > 0:
                pages_with_images += 1

        # Determine overall classification
        text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
        image_ratio = pages_with_images / total_pages if total_pages > 0 else 0

        if text_ratio >= 0.8:
            classification = "searchable"
        elif text_ratio <= 0.2 and image_ratio >= 0.5:
            classification = "image_based"
        else:
            classification = "hybrid"

        # Determine if OCR is recommended
        needs_ocr = classification in ("image_based", "hybrid") and text_ratio < 0.9

        return {
            "pdf_path": str(path),
            "classification": classification,
            "total_pages": total_pages,
            "pages_with_native_text": pages_with_text,
            "pages_with_images": pages_with_images,
            "total_native_chars": total_native_chars,
            "total_images": total_images,
            "text_coverage_ratio": round(text_ratio, 3),
            "image_coverage_ratio": round(image_ratio, 3),
            "needs_ocr": needs_ocr,
            "tesseract_available": _HAS_TESSERACT,
            "page_details": page_analyses,
        }
    finally:
        doc.close()


def extract_text_native(pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Extract text from PDF using native text layer only (no OCR).

    Uses PyMuPDF for robust text extraction with layout preservation.
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        extracted_pages: List[Dict[str, Any]] = []
        total_chars = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            text = page.get_text("text")
            char_count = len(text.strip())

            extracted_pages.append({
                "page": idx + 1,
                "text": text,
                "char_count": char_count,
            })
            total_chars += char_count

        # Combine all text
        full_text = "\n\n--- Page Break ---\n\n".join(
            p["text"] for p in extracted_pages
        )

        return {
            "pdf_path": str(path),
            "method": "native",
            "pages_extracted": len(extracted_pages),
            "total_chars": total_chars,
            "text": full_text,
            "page_details": extracted_pages,
        }
    finally:
        doc.close()


def extract_text_ocr(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """
    Extract text from PDF with OCR support.

    Engine options:
    - "auto": Try native extraction first; fall back to OCR if insufficient text
    - "native": Use only native text extraction (no OCR)
    - "tesseract": Force OCR using Tesseract
    - "force_ocr": Always use OCR even if native text exists

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)
        engine: OCR engine selection ("auto", "native", "tesseract", "force_ocr")
        dpi: Resolution for rendering pages to images (default: 300)
        language: Tesseract language code (default: "eng")

    Returns:
        Dict with extracted text and metadata
    """
    path = _ensure_file(pdf_path)

    # Validate engine choice
    valid_engines = ("auto", "native", "tesseract", "force_ocr")
    if engine not in valid_engines:
        raise PdfToolError(f"Invalid engine: {engine}. Must be one of {valid_engines}")

    if engine in ("tesseract", "force_ocr") and not _HAS_TESSERACT:
        raise PdfToolError(
            "Tesseract OCR not available. Install pytesseract and tesseract-ocr: "
            "pip install pytesseract pillow && brew install tesseract (macOS) "
            "or apt-get install tesseract-ocr (Linux)"
        )

    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        extracted_pages: List[Dict[str, Any]] = []
        total_chars = 0
        ocr_used = False
        native_used = False

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {"page": idx + 1}

            # Try native extraction first (unless force_ocr)
            native_text = ""
            if engine != "force_ocr":
                native_text = page.get_text("text").strip()
                page_result["native_chars"] = len(native_text)

            # Determine if we should use OCR for this page
            use_ocr_for_page = False
            if engine == "tesseract" or engine == "force_ocr":
                use_ocr_for_page = True
            elif engine == "auto":
                # Use OCR if native text is insufficient (less than 50 chars)
                # and the page has images
                has_images = len(page.get_images()) > 0
                insufficient_text = len(native_text) < 50
                use_ocr_for_page = has_images and insufficient_text

            # Perform OCR if needed
            ocr_text = ""
            if use_ocr_for_page and _HAS_TESSERACT:
                try:
                    # Render page to image
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")

                    # OCR with Tesseract
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img, lang=language)
                    ocr_text = ocr_text.strip()
                    page_result["ocr_chars"] = len(ocr_text)
                    ocr_used = True
                except Exception as e:
                    page_result["ocr_error"] = str(e)

            # Choose best text for this page
            if use_ocr_for_page and ocr_text:
                final_text = ocr_text
                page_result["method"] = "ocr"
            else:
                final_text = native_text
                page_result["method"] = "native"
                if native_text:
                    native_used = True

            page_result["text"] = final_text
            page_result["char_count"] = len(final_text)
            extracted_pages.append(page_result)
            total_chars += len(final_text)

        # Combine all text
        full_text = "\n\n--- Page Break ---\n\n".join(
            p["text"] for p in extracted_pages if p["text"]
        )

        # Determine overall method used
        if ocr_used and native_used:
            method = "hybrid"
        elif ocr_used:
            method = "ocr"
        else:
            method = "native"

        return {
            "pdf_path": str(path),
            "engine_requested": engine,
            "method_used": method,
            "pages_extracted": len(extracted_pages),
            "total_chars": total_chars,
            "ocr_available": _HAS_TESSERACT,
            "dpi": dpi if ocr_used else None,
            "language": language if ocr_used else None,
            "text": full_text,
            "page_details": extracted_pages,
        }
    finally:
        doc.close()


def get_pdf_text_blocks(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract text blocks with position information from PDF.

    Returns structured text blocks with bounding boxes, useful for
    understanding document layout and identifying form field locations.
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        page_blocks: List[Dict[str, Any]] = []

        for idx in page_indices:
            page = doc.load_page(idx)
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

            page_data = {
                "page": idx + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "blocks": [],
            }

            for block in blocks.get("blocks", []):
                block_type = block.get("type", 0)

                if block_type == 0:  # Text block
                    block_info = {
                        "type": "text",
                        "bbox": block.get("bbox"),
                        "lines": [],
                    }
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            block_info["lines"].append({
                                "text": line_text,
                                "bbox": line.get("bbox"),
                            })
                    if block_info["lines"]:
                        page_data["blocks"].append(block_info)

                elif block_type == 1:  # Image block
                    page_data["blocks"].append({
                        "type": "image",
                        "bbox": block.get("bbox"),
                        "width": block.get("width"),
                        "height": block.get("height"),
                    })

            page_blocks.append(page_data)

        return {
            "pdf_path": str(path),
            "total_pages": total_pages,
            "pages_analyzed": len(page_blocks),
            "page_blocks": page_blocks,
        }
    finally:
        doc.close()

