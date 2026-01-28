"""
PDF Tools - Core functionality for the PDF MCP Server.

This module provides PDF manipulation, OCR, and extraction capabilities:
- Form handling: fill, clear, flatten, create PDF forms
- Page operations: merge, extract, rotate, reorder, insert, remove
- Annotations: text, comments, watermarks, signatures, redaction, numbering, highlights, date stamps
- OCR: text extraction with Tesseract support, confidence scores
- Extraction: tables, images, text blocks with positions
- Form detection: auto-detect fields in non-AcroForm PDFs
- Export: markdown and JSON export
- PII detection: scan for common personal data patterns

Version: 0.5.0
License: AGPL-3.0
"""
from __future__ import annotations

import json
import os
import re
import secrets
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pymupdf
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

try:
    from fillpdf import fillpdfs

    _HAS_FILLPDF = True
except ImportError:
    _HAS_FILLPDF = False

try:
    import io

    import pytesseract
    from PIL import Image

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

try:
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign import validation

    _HAS_PYHANKO = True
except ImportError:
    _HAS_PYHANKO = False

# Common Tesseract language codes
TESSERACT_LANGUAGES = {
    "eng": "English",
    "chi_sim": "Chinese (Simplified)",
    "chi_tra": "Chinese (Traditional)",
    "jpn": "Japanese",
    "kor": "Korean",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "ara": "Arabic",
    "hin": "Hindi",
    "vie": "Vietnamese",
    "tha": "Thai",
}


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


def _normalize_field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "checked", "x"}


def _find_nearest_underline(label_bbox: Sequence[float], underlines: List[Dict[str, Any]]) -> Optional[Sequence[float]]:
    if not label_bbox:
        return None
    x1, y1, x2, y2 = label_bbox
    best = None
    best_score = None
    for underline in underlines:
        rect = underline.get("bbox") or []
        if len(rect) != 4:
            continue
        ux1, uy1, ux2, uy2 = rect
        if uy1 < y1 - 8 or uy1 > y2 + 12:
            continue
        if ux1 < x2 - 10:
            continue
        score = abs(uy1 - y2) + abs(ux1 - x2)
        if best_score is None or score < best_score:
            best_score = score
            best = rect
    return best


def _rect_for_label(label_bbox: Sequence[float], width: float = 200, height: float = 18) -> Sequence[float]:
    x1, y1, x2, y2 = label_bbox
    target_x1 = x2 + 6
    target_y1 = max(0, y1 - 2)
    return [target_x1, target_y1, target_x1 + width, target_y1 + height]


def create_pdf_form(
    output_path: str,
    fields: List[Dict[str, Any]],
    page_size: Optional[Sequence[float]] = None,
    pages: int = 1,
) -> Dict[str, Any]:
    """
    Create a new PDF with AcroForm fields.

    Fields format:
    - name (str, required)
    - type (str, "text" or "checkbox", default "text")
    - rect (list[float], required) in PDF coordinates [x1, y1, x2, y2]
    - page (int, 1-based, default 1)
    - value (str/bool, optional)
    - multiline (bool, optional, text only)
    """
    if not fields:
        raise PdfToolError("fields must include at least one field definition")
    if pages < 1:
        raise PdfToolError("pages must be >= 1")

    width, height = (595.0, 842.0) if page_size is None else (float(page_size[0]), float(page_size[1]))
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=width, height=height)

    field_refs = ArrayObject()
    for field_def in fields:
        name = field_def.get("name")
        if not name:
            raise PdfToolError("Each field must include a name")
        field_type = (field_def.get("type") or "text").lower()
        rect = _ensure_rect(field_def.get("rect"))
        page_index = int(field_def.get("page", 1)) - 1
        if page_index < 0 or page_index >= len(writer.pages):
            raise PdfToolError(f"Field page out of range: {field_def.get('page')}")

        if field_type not in {"text", "checkbox"}:
            raise PdfToolError("field type must be 'text' or 'checkbox'")

        if field_type == "text":
            field = DictionaryObject(
                {
                    NameObject("/FT"): NameObject("/Tx"),
                    NameObject("/T"): TextStringObject(str(name)),
                    NameObject("/Ff"): NumberObject(4096 if field_def.get("multiline") else 0),
                    NameObject("/V"): TextStringObject(str(field_def.get("value", ""))),
                }
            )
        else:
            checked = _is_truthy(field_def.get("value"))
            state = NameObject("/Yes") if checked else NameObject("/Off")
            field = DictionaryObject(
                {
                    NameObject("/FT"): NameObject("/Btn"),
                    NameObject("/T"): TextStringObject(str(name)),
                    NameObject("/V"): state,
                    NameObject("/AS"): state,
                }
            )

        field_ref = writer._add_object(field)  # type: ignore[attr-defined]
        field_refs.append(field_ref)

        widget = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/Rect"): rect,
                NameObject("/F"): NumberObject(4),
                NameObject("/Parent"): field_ref,
            }
        )
        widget_ref = writer._add_object(widget)  # type: ignore[attr-defined]
        field[NameObject("/Kids")] = ArrayObject([widget_ref])

        page_obj = writer.pages[page_index]
        existing = page_obj.get("/Annots")
        if existing is None:
            annots = ArrayObject()
        else:
            existing_obj = existing.get_object() if hasattr(existing, "get_object") else existing
            annots = ArrayObject(list(existing_obj))
        annots.append(widget_ref)
        page_obj[NameObject("/Annots")] = annots

    acro_form = DictionaryObject(
        {
            NameObject("/Fields"): field_refs,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
        }
    )
    writer._root_object.update({NameObject("/AcroForm"): writer._add_object(acro_form)})  # type: ignore[attr-defined]

    dst = _prepare_output(output_path)
    with dst.open("wb") as f:
        writer.write(f)

    return {
        "output_path": str(dst),
        "pages": pages,
        "fields_created": len(field_refs),
        "field_names": [f.get("name") for f in fields],
    }


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


def fill_pdf_form_any(
    input_path: str,
    output_path: str,
    data: Dict[str, Any],
    flatten: bool = False,
) -> Dict[str, Any]:
    """
    Fill standard (AcroForm) PDFs and attempt best-effort filling for non-standard forms.

    If the PDF has AcroForm fields, this defers to fill_pdf_form.
    Otherwise, it detects field-like labels and writes FreeText annotations near them.
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    reader = PdfReader(str(src))
    has_fields = bool(reader.get_fields())

    if has_fields:
        result = fill_pdf_form(str(src), str(dst), {str(k): str(v) for k, v in data.items()}, flatten=flatten)
        result["method"] = "acroform"
        return result

    detection = detect_form_fields(str(src))
    detected = detection.get("detected_fields") or []
    if not detected:
        raise PdfToolError("No form fields detected for non-standard form filling")

    normalized_labels = []
    for entry in detected:
        label = entry.get("text", "")
        if label:
            normalized_labels.append((_normalize_field_key(label), entry))

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    filled = 0
    missing = []
    page_analysis = {p["page"]: p for p in detection.get("page_analysis", [])}
    for key, value in data.items():
        normalized_key = _normalize_field_key(str(key))
        match = next((entry for n, entry in normalized_labels if n == normalized_key or normalized_key in n), None)
        if match is None:
            missing.append(str(key))
            continue

        page_num = match.get("page", 1)
        page_index = int(page_num) - 1
        if page_index < 0 or page_index >= len(writer.pages):
            continue

        label_bbox = match.get("bbox") or []
        underline_rect = None
        analysis = page_analysis.get(page_num, {})
        underlines = analysis.get("detected_underlines", [])
        if label_bbox:
            underline_rect = _find_nearest_underline(label_bbox, underlines)

        if underline_rect:
            rect = _ensure_rect(underline_rect)
        elif label_bbox:
            rect = _ensure_rect(_rect_for_label(label_bbox))
        else:
            missing.append(str(key))
            continue

        text_value = "X" if match.get("type") == "checkbox" and _is_truthy(value) else str(value)
        annotation_id = secrets.token_hex(8)
        _add_freetext_annotation(writer, writer.pages[page_index], text_value, rect, annotation_id)
        filled += 1

    if flatten:
        _flatten_writer(writer)

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {
        "output_path": str(dst),
        "method": "detected_labels",
        "fields_filled": filled,
        "missing_fields": missing,
        "flattened": flatten,
        "detected_fields": len(detected),
    }


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


def reorder_pages(input_path: str, pages: List[int], output_path: str) -> Dict:
    src = _ensure_file(input_path)
    if not pages:
        raise PdfToolError("No pages specified for reorder")

    reader = PdfReader(str(src))
    total = len(reader.pages)
    zero_based = _validate_reorder_pages(pages, total)

    dst = _prepare_output(output_path)
    writer = PdfWriter()
    for idx in zero_based:
        writer.add_page(reader.pages[idx])

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "reordered": len(zero_based)}


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


def _validate_reorder_pages(pages: List[int], total: int) -> List[int]:
    converted: List[int] = []
    seen: set[int] = set()
    for page in pages:
        if page == 0:
            raise PdfToolError("Page numbers must be 1-based")
        idx = page - 1 if page > 0 else total + page
        if idx < 0 or idx >= total:
            raise PdfToolError(f"Page {page} is out of range (1-{total})")
        if idx in seen:
            raise PdfToolError(f"Duplicate page specified for reorder: {page}")
        seen.add(idx)
        converted.append(idx)

    if len(converted) != total:
        raise PdfToolError(
            "Reorder requires a complete page list matching the document length"
        )

    return converted


def _ensure_rect(rect: Optional[Sequence[float]]) -> ArrayObject:
    if rect is None:
        rect = (50, 50, 250, 100)
    if len(rect) != 4:
        raise PdfToolError("rect must contain exactly 4 numbers: [x1, y1, x2, y2]")
    return ArrayObject([NumberObject(float(x)) for x in rect])


def _freetext_rect_for_position(
    page: Any,
    position: str,
    margin: float,
    width: float,
    height: float,
) -> ArrayObject:
    mediabox = page.mediabox
    page_width = float(mediabox.width)
    page_height = float(mediabox.height)

    if position == "bottom-left":
        x1, y1 = margin, margin
    elif position == "bottom-center":
        x1, y1 = (page_width - width) / 2, margin
    else:  # bottom-right
        x1, y1 = page_width - width - margin, margin

    x2 = x1 + width
    y2 = y1 + height
    return _ensure_rect((x1, y1, x2, y2))


def _add_freetext_annotation(
    writer: PdfWriter,
    page_obj: Any,
    text: str,
    rect: ArrayObject,
    annotation_id: str,
) -> None:
    annot = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/FreeText"),
            NameObject("/Rect"): rect,
            NameObject("/Contents"): TextStringObject(text),
            NameObject("/NM"): TextStringObject(annotation_id),
            NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g"),
            NameObject("/F"): NumberObject(4),
        }
    )
    annot_ref = writer._add_object(annot)  # type: ignore[attr-defined]

    existing = page_obj.get("/Annots")
    if existing is None:
        annots = ArrayObject()
    else:
        existing_obj = existing.get_object() if hasattr(existing, "get_object") else existing
        annots = ArrayObject(list(existing_obj))
    annots.append(annot_ref)
    page_obj[NameObject("/Annots")] = annots

def redact_text_regex(
    input_path: str,
    output_path: str,
    pattern: str,
    pages: Optional[List[int]] = None,
    case_insensitive: bool = False,
    whole_words: bool = False,
    fill: Optional[Sequence[float]] = None,
) -> Dict:
    if not pattern:
        raise PdfToolError("Redaction pattern must be provided")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    regex_flags = re.IGNORECASE if case_insensitive else 0
    if whole_words:
        pattern = rf"\\b{pattern}\\b"
    regex = re.compile(pattern, regex_flags)

    doc = pymupdf.open(str(src))
    total = doc.page_count
    page_indices = _to_zero_based_pages(pages, total) if pages else list(range(total))
    if not page_indices:
        raise PdfToolError("No pages selected for redaction")

    redacted = 0
    for idx in page_indices:
        page = doc.load_page(idx)
        words = page.get_text("words") or []
        if not words:
            continue
        words_sorted = sorted(words, key=lambda w: (w[5], w[6], w[7]))
        combined_parts: List[str] = []
        spans: List[tuple[int, int, tuple[float, float, float, float]]] = []
        offset = 0
        for w in words_sorted:
            if combined_parts:
                combined_parts.append(" ")
                offset += 1
            word_text = str(w[4])
            start = offset
            combined_parts.append(word_text)
            offset += len(word_text)
            spans.append((start, offset, (w[0], w[1], w[2], w[3])))

        combined_text = "".join(combined_parts)
        page_redactions = 0
        for match in regex.finditer(combined_text):
            match_start, match_end = match.span()
            for span_start, span_end, rect in spans:
                if span_start < match_end and span_end > match_start:
                    page.add_redact_annot(rect, fill=fill or (0, 0, 0))
                    redacted += 1
                    page_redactions += 1
        if page_redactions:
            page.apply_redactions()

    doc.save(str(dst))
    doc.close()

    return {"output_path": str(dst), "redacted": redacted, "pages": len(page_indices)}


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


def sanitize_pdf_metadata(
    input_path: str,
    output_path: str,
    remove_custom: bool = True,
    remove_xmp: bool = True,
) -> Dict[str, Any]:
    """
    Remove metadata keys from a PDF.

    By default, this removes standard metadata keys and any custom keys.
    """
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    reader = PdfReader(str(src))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    standard_keys = {
        "/Title",
        "/Author",
        "/Subject",
        "/Keywords",
        "/Creator",
        "/Producer",
        "/CreationDate",
        "/ModDate",
        "/Trapped",
    }

    removed: List[str] = []
    info = getattr(writer, "_info", None)
    if info is not None:
        info_obj = info.get_object() if hasattr(info, "get_object") else info
        for key in list(info_obj.keys()):
            key_str = str(key)
            normalized = key_str[1:] if key_str.startswith("/") else key_str
            if key_str in standard_keys or remove_custom:
                removed.append(normalized)
                del info_obj[key]

    if remove_xmp:
        root_obj = writer._root_object  # type: ignore[attr-defined]
        if NameObject("/Metadata") in root_obj:
            del root_obj[NameObject("/Metadata")]
            removed.append("Metadata")

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "removed": sorted(set(removed))}


def _extract_text_for_export(
    pdf_path: str,
    pages: Optional[List[int]],
    engine: str,
    dpi: int,
    language: str,
) -> Dict[str, Any]:
    if engine == "auto":
        return extract_text_smart(pdf_path, pages=pages)
    if engine == "native":
        return extract_text_native(pdf_path, pages=pages)
    if engine in ("ocr", "tesseract", "force_ocr"):
        ocr_engine = "tesseract" if engine == "ocr" else engine
        return extract_text_ocr(pdf_path, pages=pages, engine=ocr_engine, dpi=dpi, language=language)
    raise PdfToolError("engine must be one of: auto, native, ocr, tesseract, force_ocr")


def export_to_json(
    pdf_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    src = _ensure_file(pdf_path)
    dst = _prepare_output(output_path)

    text_result = _extract_text_for_export(str(src), pages, engine, dpi, language)
    reader = PdfReader(str(src))

    payload = {
        "pdf_path": str(src),
        "engine": engine,
        "page_count": len(reader.pages),
        "metadata": get_pdf_metadata(str(src))["metadata"],
        "pages": [
            {"page": p["page"], "text": p["text"], "char_count": p["char_count"]}
            for p in text_result.get("page_details", [])
        ],
    }

    dst.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    return {"output_path": str(dst), "page_count": payload["page_count"], "engine": engine}


def export_to_markdown(
    pdf_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    src = _ensure_file(pdf_path)
    dst = _prepare_output(output_path)

    text_result = _extract_text_for_export(str(src), pages, engine, dpi, language)
    parts: List[str] = []
    for page in text_result.get("page_details", []):
        parts.append(f"# Page {page['page']}")
        parts.append(page["text"].rstrip())
        parts.append("")

    dst.write_text("\n".join(parts), encoding="utf-8")
    return {"output_path": str(dst), "engine": engine, "pages": len(text_result.get("page_details", []))}


def add_page_numbers(
    input_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    start: int = 1,
    position: str = "bottom-right",
    width: float = 120,
    height: float = 20,
    margin: float = 20,
) -> Dict[str, Any]:
    if position not in ("bottom-left", "bottom-center", "bottom-right"):
        raise PdfToolError("position must be bottom-left, bottom-center, or bottom-right")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = _to_zero_based_pages(pages, total) if pages else list(range(total))
    if not page_indices:
        raise PdfToolError("No pages selected for numbering")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    added = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        label = str(start + idx)
        rect = _freetext_rect_for_position(page_obj, position, margin, width, height)
        annotation_id = f"pdf-mcp-page-number-{idx + 1}"
        _add_freetext_annotation(writer, page_obj, label, rect, annotation_id)
        added += 1

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "added": added}


def add_bates_numbering(
    input_path: str,
    output_path: str,
    prefix: str = "",
    start: int = 1,
    width: int = 6,
    pages: Optional[List[int]] = None,
    position: str = "bottom-right",
    margin: float = 20,
    box_width: float = 160,
    box_height: float = 20,
) -> Dict[str, Any]:
    if position not in ("bottom-left", "bottom-center", "bottom-right"):
        raise PdfToolError("position must be bottom-left, bottom-center, or bottom-right")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = _to_zero_based_pages(pages, total) if pages else list(range(total))
    if not page_indices:
        raise PdfToolError("No pages selected for Bates numbering")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    added = 0
    for i, idx in enumerate(page_indices):
        page_obj = writer.pages[idx]
        number = start + i
        label = f"{prefix}{number:0{width}d}"
        rect = _freetext_rect_for_position(page_obj, position, margin, box_width, box_height)
        annotation_id = f"pdf-mcp-bates-{idx + 1}"
        _add_freetext_annotation(writer, page_obj, label, rect, annotation_id)
        added += 1

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "added": added}


def verify_digital_signatures(pdf_path: str) -> Dict[str, Any]:
    src = _ensure_file(pdf_path)
    if not _HAS_PYHANKO:
        raise PdfToolError("pyHanko not available. Install pyhanko to verify signatures.")

    with src.open("rb") as fh:
        reader = PdfFileReader(fh)
        signatures = reader.embedded_signatures

        if not signatures:
            return {"pdf_path": str(src), "signatures": [], "verified": 0}

        results = []
        verified = 0
        vc = validation.ValidationContext(allow_fetching=False)
        for sig in signatures:
            try:
                status = validation.validate_pdf_signature(sig, vc)
                result = {
                    "field_name": sig.field_name,
                    "intact": status.intact,
                    "valid": status.valid,
                    "trusted": status.trusted,
                    "modification_level": str(status.modification_level),
                }
                if status.valid:
                    verified += 1
            except Exception as exc:
                result = {
                    "field_name": sig.field_name,
                    "error": str(exc),
                }
            results.append(result)

    return {"pdf_path": str(src), "signatures": results, "verified": verified}


def get_full_metadata(pdf_path: str) -> Dict[str, Any]:
    path = _ensure_file(pdf_path)
    reader = PdfReader(str(path))
    return {
        "metadata": get_pdf_metadata(str(path))["metadata"],
        "document": {
            "page_count": len(reader.pages),
            "is_encrypted": bool(reader.is_encrypted),
            "file_size_bytes": os.path.getsize(path),
        },
    }


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


# =============================================================================
# OCR Phase 2: Enhanced OCR with multi-language and confidence scores
# =============================================================================


def get_ocr_languages() -> Dict[str, Any]:
    """
    Get available OCR languages and Tesseract installation status.

    Returns list of common language codes and whether Tesseract is available.
    """
    installed_languages: List[str] = []
    if _HAS_TESSERACT:
        try:
            # Get installed languages from tesseract
            langs = pytesseract.get_languages()
            installed_languages = [l for l in langs if l != "osd"]
        except Exception:
            pass

    return {
        "tesseract_available": _HAS_TESSERACT,
        "installed_languages": installed_languages,
        "common_language_codes": TESSERACT_LANGUAGES,
    }


def extract_text_with_confidence(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    language: str = "eng",
    dpi: int = 300,
    min_confidence: int = 0,
) -> Dict[str, Any]:
    """
    Extract text from PDF with OCR confidence scores.

    This function performs OCR and returns word-level confidence scores,
    allowing filtering of low-confidence text.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)
        language: Tesseract language code (default: "eng"). Use "+" for multiple: "eng+fra"
        dpi: Resolution for rendering pages (default: 300)
        min_confidence: Minimum confidence threshold 0-100 (default: 0 = all text)

    Returns:
        Dict with text, confidence scores, and word-level details
    """
    if not _HAS_TESSERACT:
        raise PdfToolError(
            "Tesseract OCR not available. Install pytesseract and tesseract-ocr."
        )

    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        extracted_pages: List[Dict[str, Any]] = []
        total_words = 0
        total_confidence_sum = 0.0
        all_text_parts: List[str] = []

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {"page": idx + 1, "words": []}

            # Render page to image
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Get word-level OCR data with confidence
            try:
                ocr_data = pytesseract.image_to_data(
                    img, lang=language, output_type=pytesseract.Output.DICT
                )

                page_text_parts: List[str] = []
                page_words: List[Dict[str, Any]] = []
                page_confidence_sum = 0.0
                page_word_count = 0

                for i, word in enumerate(ocr_data["text"]):
                    conf = int(ocr_data["conf"][i])
                    if word.strip() and conf >= min_confidence:
                        word_info = {
                            "text": word,
                            "confidence": conf,
                            "bbox": [
                                ocr_data["left"][i],
                                ocr_data["top"][i],
                                ocr_data["left"][i] + ocr_data["width"][i],
                                ocr_data["top"][i] + ocr_data["height"][i],
                            ],
                            "line_num": ocr_data["line_num"][i],
                            "block_num": ocr_data["block_num"][i],
                        }
                        page_words.append(word_info)
                        page_text_parts.append(word)
                        if conf >= 0:  # Tesseract returns -1 for non-text
                            page_confidence_sum += conf
                            page_word_count += 1

                page_text = " ".join(page_text_parts)
                page_avg_confidence = (
                    page_confidence_sum / page_word_count if page_word_count > 0 else 0
                )

                page_result["text"] = page_text
                page_result["words"] = page_words
                page_result["word_count"] = page_word_count
                page_result["average_confidence"] = round(page_avg_confidence, 1)

                all_text_parts.append(page_text)
                total_words += page_word_count
                total_confidence_sum += page_confidence_sum

            except Exception as e:
                page_result["error"] = str(e)
                page_result["text"] = ""
                page_result["words"] = []
                page_result["word_count"] = 0
                page_result["average_confidence"] = 0

            extracted_pages.append(page_result)

        overall_avg_confidence = (
            total_confidence_sum / total_words if total_words > 0 else 0
        )

        return {
            "pdf_path": str(path),
            "language": language,
            "dpi": dpi,
            "min_confidence": min_confidence,
            "pages_extracted": len(extracted_pages),
            "total_words": total_words,
            "overall_average_confidence": round(overall_avg_confidence, 1),
            "text": "\n\n--- Page Break ---\n\n".join(all_text_parts),
            "page_details": extracted_pages,
        }
    finally:
        doc.close()


# =============================================================================
# Table Extraction
# =============================================================================


def extract_tables(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_format: str = "list",
) -> Dict[str, Any]:
    """
    Extract tables from PDF pages.

    Uses PyMuPDF's table detection to find and extract tabular data.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)
        output_format: "list" for list of lists, "dict" for list of dicts with headers

    Returns:
        Dict with extracted tables per page
    """
    if output_format not in ("list", "dict"):
        raise PdfToolError("output_format must be 'list' or 'dict'")

    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        page_tables: List[Dict[str, Any]] = []
        total_tables = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {
                "page": idx + 1,
                "tables": [],
            }

            # Use PyMuPDF's table finder
            try:
                tabs = page.find_tables()

                for table_idx, table in enumerate(tabs):
                    # Extract table data
                    raw_data = table.extract()

                    if not raw_data:
                        continue

                    table_info: Dict[str, Any] = {
                        "table_index": table_idx,
                        "bbox": list(table.bbox),
                        "rows": len(raw_data),
                        "cols": len(raw_data[0]) if raw_data else 0,
                    }

                    if output_format == "dict" and len(raw_data) > 1:
                        # Use first row as headers
                        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(raw_data[0])]
                        table_info["headers"] = headers
                        table_info["data"] = [
                            {headers[i]: cell for i, cell in enumerate(row)}
                            for row in raw_data[1:]
                        ]
                    else:
                        table_info["data"] = raw_data

                    page_result["tables"].append(table_info)
                    total_tables += 1

            except Exception as e:
                page_result["error"] = str(e)

            page_tables.append(page_result)

        return {
            "pdf_path": str(path),
            "total_pages": total_pages,
            "pages_analyzed": len(page_tables),
            "total_tables": total_tables,
            "output_format": output_format,
            "page_tables": page_tables,
        }
    finally:
        doc.close()


# =============================================================================
# Image Extraction
# =============================================================================


def extract_images(
    pdf_path: str,
    output_dir: str,
    pages: Optional[List[int]] = None,
    min_width: int = 50,
    min_height: int = 50,
    image_format: str = "png",
) -> Dict[str, Any]:
    """
    Extract embedded images from PDF pages.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
        pages: Optional list of 1-based page numbers (default: all pages)
        min_width: Minimum image width to extract (default: 50)
        min_height: Minimum image height to extract (default: 50)
        image_format: Output format: "png", "jpeg", "ppm" (default: "png")

    Returns:
        Dict with list of extracted image paths and metadata
    """
    if image_format not in ("png", "jpeg", "ppm"):
        raise PdfToolError("image_format must be 'png', 'jpeg', or 'ppm'")

    path = _ensure_file(pdf_path)
    out_dir = Path(output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        extracted_images: List[Dict[str, Any]] = []
        skipped_count = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            images = page.get_images(full=True)

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]

                try:
                    # Extract image data
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    img_bytes = base_image.get("image")
                    img_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip small images
                    if width < min_width or height < min_height:
                        skipped_count += 1
                        continue

                    # Determine output filename
                    output_ext = "jpg" if image_format == "jpeg" else image_format
                    filename = f"page{idx + 1}_img{img_idx + 1}.{output_ext}"
                    output_path = out_dir / filename

                    # Convert format if needed
                    if image_format != img_ext:
                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            # PIL uses "JPEG" not "jpeg"
                            pil_format = "JPEG" if image_format == "jpeg" else image_format.upper()
                            with output_path.open("wb") as f:
                                pil_img.save(f, format=pil_format)
                        except Exception:
                            # Fall back to original format
                            fallback_path = out_dir / f"page{idx + 1}_img{img_idx + 1}.{img_ext}"
                            with fallback_path.open("wb") as f:
                                f.write(img_bytes)
                            output_path = fallback_path
                    else:
                        with output_path.open("wb") as f:
                            f.write(img_bytes)

                    extracted_images.append({
                        "page": idx + 1,
                        "image_index": img_idx,
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "original_format": img_ext,
                        "output_path": str(output_path),
                    })

                except Exception as e:
                    extracted_images.append({
                        "page": idx + 1,
                        "image_index": img_idx,
                        "xref": xref,
                        "error": str(e),
                    })

        return {
            "pdf_path": str(path),
            "output_dir": str(out_dir),
            "total_pages": total_pages,
            "pages_processed": len(page_indices),
            "images_extracted": len([i for i in extracted_images if "output_path" in i]),
            "images_skipped": skipped_count,
            "min_dimensions": {"width": min_width, "height": min_height},
            "images": extracted_images,
        }
    finally:
        doc.close()


def get_image_info(pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Get information about images in a PDF without extracting them.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)

    Returns:
        Dict with image metadata per page
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        page_images: List[Dict[str, Any]] = []
        total_images = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            images = page.get_images(full=True)

            page_info: Dict[str, Any] = {
                "page": idx + 1,
                "image_count": len(images),
                "images": [],
            }

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    colorspace = base_image.get("colorspace", 0)
                    bpc = base_image.get("bpc", 0)
                    img_ext = base_image.get("ext", "unknown")

                    # Get image position on page
                    img_rects = page.get_image_rects(xref)
                    positions = [list(r) for r in img_rects] if img_rects else []

                    page_info["images"].append({
                        "index": img_idx,
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "format": img_ext,
                        "colorspace": colorspace,
                        "bits_per_component": bpc,
                        "positions": positions,
                    })
                except Exception as e:
                    page_info["images"].append({
                        "index": img_idx,
                        "xref": xref,
                        "error": str(e),
                    })

            total_images += len(images)
            page_images.append(page_info)

        return {
            "pdf_path": str(path),
            "total_pages": total_pages,
            "pages_analyzed": len(page_images),
            "total_images": total_images,
            "page_images": page_images,
        }
    finally:
        doc.close()


# =============================================================================
# Hybrid Document Processing (Enhanced)
# =============================================================================


def extract_text_smart(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    native_threshold: int = 100,
    ocr_dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """
    Smart text extraction with per-page method selection.

    For each page, decides whether to use native extraction or OCR based on
    the native text content. This provides optimal results for hybrid documents.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)
        native_threshold: Minimum chars to prefer native extraction (default: 100)
        ocr_dpi: DPI for OCR rendering (default: 300)
        language: Tesseract language code (default: "eng")

    Returns:
        Dict with extracted text and per-page method details
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
        all_text_parts: List[str] = []
        total_chars = 0
        native_pages = 0
        ocr_pages = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {"page": idx + 1}

            # Try native extraction first
            native_text = page.get_text("text").strip()
            native_chars = len(native_text)
            page_result["native_chars"] = native_chars

            # Check if page has images
            images = page.get_images()
            has_images = len(images) > 0
            page_result["has_images"] = has_images

            # Decide method for this page
            use_native = native_chars >= native_threshold

            if use_native:
                page_result["method"] = "native"
                page_result["text"] = native_text
                page_result["char_count"] = native_chars
                all_text_parts.append(native_text)
                total_chars += native_chars
                native_pages += 1
            else:
                # Try OCR if Tesseract is available and page has images
                if _HAS_TESSERACT and has_images:
                    try:
                        mat = pymupdf.Matrix(ocr_dpi / 72, ocr_dpi / 72)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, lang=language).strip()

                        page_result["method"] = "ocr"
                        page_result["text"] = ocr_text
                        page_result["char_count"] = len(ocr_text)
                        all_text_parts.append(ocr_text)
                        total_chars += len(ocr_text)
                        ocr_pages += 1
                    except Exception as e:
                        # Fall back to native on OCR error
                        page_result["method"] = "native"
                        page_result["text"] = native_text
                        page_result["char_count"] = native_chars
                        page_result["ocr_error"] = str(e)
                        all_text_parts.append(native_text)
                        total_chars += native_chars
                        native_pages += 1
                else:
                    # No OCR available, use native
                    page_result["method"] = "native"
                    page_result["text"] = native_text
                    page_result["char_count"] = native_chars
                    page_result["ocr_unavailable"] = not _HAS_TESSERACT
                    all_text_parts.append(native_text)
                    total_chars += native_chars
                    native_pages += 1

            extracted_pages.append(page_result)

        return {
            "pdf_path": str(path),
            "total_pages": total_pages,
            "pages_extracted": len(extracted_pages),
            "total_chars": total_chars,
            "native_threshold": native_threshold,
            "pages_using_native": native_pages,
            "pages_using_ocr": ocr_pages,
            "tesseract_available": _HAS_TESSERACT,
            "text": "\n\n--- Page Break ---\n\n".join(all_text_parts),
            "page_details": extracted_pages,
        }
    finally:
        doc.close()


# =============================================================================
# Form Auto-Detection
# =============================================================================


def detect_form_fields(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Detect potential form fields in a PDF using text analysis.

    Analyzes text blocks to find patterns that suggest fillable fields:
    - Text followed by underlines or boxes
    - Label patterns (e.g., "Name:", "Date:", "Address:")
    - Checkbox indicators
    - Empty rectangular regions near labels

    This is useful for PDFs that don't have AcroForm fields but appear
    to be forms visually.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)

    Returns:
        Dict with detected potential form fields
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    # Common form field label patterns
    import re
    label_patterns = [
        re.compile(r"^(name|full name|first name|last name)\s*:?\s*$", re.I),
        re.compile(r"^(date|dob|date of birth)\s*:?\s*$", re.I),
        re.compile(r"^(address|street|city|state|zip|postal)\s*:?\s*$", re.I),
        re.compile(r"^(phone|telephone|mobile|cell|fax)\s*:?\s*$", re.I),
        re.compile(r"^(email|e-mail)\s*:?\s*$", re.I),
        re.compile(r"^(signature)\s*:?\s*$", re.I),
        re.compile(r"^(company|organization|employer)\s*:?\s*$", re.I),
        re.compile(r"^(title|position|job title)\s*:?\s*$", re.I),
        re.compile(r"^(ssn|social security|tax id|ein)\s*:?\s*$", re.I),
        re.compile(r"^(amount|total|subtotal|price)\s*:?\s*$", re.I),
        re.compile(r"^(comments?|notes?|remarks?)\s*:?\s*$", re.I),
    ]

    # Checkbox/selection patterns
    checkbox_patterns = [
        re.compile(r"^\s*[\[\(\{\<]\s*[\]\)\}\>]\s*", re.I),  # [ ] or ( ) etc.
        re.compile(r"^\s*[\u2610\u2611\u2612\u25A1\u25A0]\s*", re.I),  # Unicode checkboxes
        re.compile(r"^(yes|no)\s*[\[\(\{]?\s*[\]\)\}]?\s*$", re.I),
    ]

    try:
        total_pages = doc.page_count
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))

        # Check if PDF already has AcroForm fields
        reader = PdfReader(str(path))
        existing_fields = reader.get_fields() or {}
        has_acroform = len(existing_fields) > 0

        detected_fields: List[Dict[str, Any]] = []
        page_analyses: List[Dict[str, Any]] = []

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {
                "page": idx + 1,
                "detected_labels": [],
                "detected_checkboxes": [],
                "detected_underlines": [],
            }

            # Get text blocks with positions
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

            for block in blocks.get("blocks", []):
                if block.get("type", 0) != 0:  # Skip non-text blocks
                    continue

                for line in block.get("lines", []):
                    line_text = ""
                    line_bbox = line.get("bbox", [])
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")

                    line_text_clean = line_text.strip()
                    if not line_text_clean:
                        continue

                    # Check for label patterns
                    for pattern in label_patterns:
                        if pattern.match(line_text_clean):
                            field_info = {
                                "type": "label",
                                "text": line_text_clean,
                                "bbox": list(line_bbox) if line_bbox else None,
                                "page": idx + 1,
                                "suggested_field_type": _guess_field_type(line_text_clean),
                            }
                            page_result["detected_labels"].append(field_info)
                            detected_fields.append(field_info)
                            break

                    # Check for checkbox patterns
                    for pattern in checkbox_patterns:
                        if pattern.match(line_text_clean):
                            field_info = {
                                "type": "checkbox",
                                "text": line_text_clean,
                                "bbox": list(line_bbox) if line_bbox else None,
                                "page": idx + 1,
                            }
                            page_result["detected_checkboxes"].append(field_info)
                            detected_fields.append(field_info)
                            break

            # Detect drawings that might be form fields (lines, rectangles)
            try:
                drawings = page.get_drawings()
                for drawing in drawings:
                    if drawing.get("type") == "l":  # Line
                        rect = drawing.get("rect", [])
                        if rect:
                            # Horizontal lines might be underlines for text fields
                            if abs(rect[3] - rect[1]) < 5:  # Nearly horizontal
                                width = abs(rect[2] - rect[0])
                                if width > 50:  # Minimum width
                                    page_result["detected_underlines"].append({
                                        "bbox": list(rect),
                                        "width": width,
                                    })
            except Exception:
                pass  # get_drawings might not be available in all PyMuPDF versions

            page_analyses.append(page_result)

        return {
            "pdf_path": str(path),
            "total_pages": total_pages,
            "pages_analyzed": len(page_analyses),
            "has_existing_acroform": has_acroform,
            "existing_field_count": len(existing_fields),
            "detected_potential_fields": len(detected_fields),
            "detected_fields": detected_fields,
            "page_analysis": page_analyses,
            "recommendation": _form_recommendation(has_acroform, len(detected_fields)),
        }
    finally:
        doc.close()


def _guess_field_type(label_text: str) -> str:
    """Guess the appropriate form field type based on label text."""
    label_lower = label_text.lower()
    if any(x in label_lower for x in ["date", "dob"]):
        return "date"
    if any(x in label_lower for x in ["email", "e-mail"]):
        return "email"
    if any(x in label_lower for x in ["phone", "telephone", "mobile", "cell", "fax"]):
        return "phone"
    if any(x in label_lower for x in ["signature"]):
        return "signature"
    if any(x in label_lower for x in ["address", "street", "city"]):
        return "address"
    if any(x in label_lower for x in ["amount", "total", "price"]):
        return "number"
    if any(x in label_lower for x in ["comments", "notes", "remarks"]):
        return "textarea"
    return "text"


def _form_recommendation(has_acroform: bool, detected_count: int) -> str:
    """Generate recommendation based on form analysis."""
    if has_acroform:
        return "PDF has existing AcroForm fields. Use get_pdf_form_fields and fill_pdf_form."
    if detected_count > 0:
        return (
            f"Detected {detected_count} potential form fields. "
            "Consider using fill_pdf_form_any to fill fields at detected positions."
        )
    return "No form fields detected. PDF may not be a form."


# =============================================================================
# Phase 3 Features
# =============================================================================


def add_highlight(
    input_path: str,
    output_path: str,
    page: int,
    text: Optional[str] = None,
    rect: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Add highlight annotations by text search or by rectangle.
    """
    if text is None and rect is None:
        raise PdfToolError("Provide either text or rect to highlight")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    doc = pymupdf.open(str(src))

    try:
        page_index = page - 1
        if page_index < 0 or page_index >= doc.page_count:
            raise PdfToolError(f"Page {page} is out of range")
        page_obj = doc.load_page(page_index)

        rects = []
        if text:
            rects = page_obj.search_for(text)
        elif rect is not None:
            if len(rect) != 4:
                raise PdfToolError("rect must contain exactly 4 numbers: [x1, y1, x2, y2]")
            rects = [pymupdf.Rect(rect)]

        added = 0
        for r in rects:
            page_obj.add_highlight_annot(r)
            added += 1

        doc.save(str(dst))
        return {"output_path": str(dst), "added": added}
    finally:
        doc.close()


def add_date_stamp(
    input_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    position: str = "bottom-right",
    margin: float = 20,
    width: float = 120,
    height: float = 20,
    date_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a date stamp as a FreeText annotation.
    """
    if position not in ("bottom-left", "bottom-center", "bottom-right"):
        raise PdfToolError("position must be bottom-left, bottom-center, or bottom-right")

    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)
    reader = PdfReader(str(src))
    total = len(reader.pages)
    page_indices = _to_zero_based_pages(pages, total) if pages else list(range(total))
    if not page_indices:
        raise PdfToolError("No pages selected for date stamp")

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    stamp_text = date_text or date.today().isoformat()
    added = 0
    for idx in page_indices:
        page_obj = writer.pages[idx]
        rect = _freetext_rect_for_position(page_obj, position, margin, width, height)
        annotation_id = f"pdf-mcp-date-stamp-{idx + 1}"
        _add_freetext_annotation(writer, page_obj, stamp_text, rect, annotation_id)
        added += 1

    with dst.open("wb") as output_file:
        writer.write(output_file)

    return {"output_path": str(dst), "added": added, "date": stamp_text}


def _luhn_check(value: str) -> bool:
    digits = [int(ch) for ch in value if ch.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def detect_pii_patterns(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Detect common PII patterns (email, phone, SSN, credit card) using regex.
    """
    src = _ensure_file(pdf_path)
    doc = pymupdf.open(str(src))

    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone_re = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
    ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    cc_re = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    try:
        total_pages = doc.page_count
        page_indices = _to_zero_based_pages(pages, total_pages) if pages else list(range(total_pages))
        matches: List[Dict[str, Any]] = []

        for idx in page_indices:
            page = doc.load_page(idx)
            text = page.get_text()
            for m in email_re.findall(text):
                matches.append({"type": "email", "value": m, "page": idx + 1})
            for m in phone_re.findall(text):
                matches.append({"type": "phone", "value": m, "page": idx + 1})
            for m in ssn_re.findall(text):
                matches.append({"type": "ssn", "value": m, "page": idx + 1})
            for m in cc_re.findall(text):
                cleaned = re.sub(r"[^0-9]", "", m)
                if _luhn_check(cleaned):
                    matches.append({"type": "credit_card", "value": cleaned, "page": idx + 1})

        return {
            "pdf_path": str(src),
            "pages_scanned": len(page_indices),
            "total_matches": len(matches),
            "matches": matches,
        }
    finally:
        doc.close()

# Optional pyzbar for barcode detection
try:
    from pyzbar import pyzbar
    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False


def extract_links(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract links (URLs, hyperlinks, internal references) from a PDF.

    Args:
        pdf_path: Path to the PDF file
        pages: Optional list of page numbers (1-indexed). None = all pages.

    Returns:
        Dict with link information:
        - pdf_path: Path to the PDF
        - total_links: Total number of links found
        - links: List of link details (page, type, uri, rect)
        - link_types: Count of links by type
        - pages_scanned: Number of pages scanned
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = len(doc)
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))
        
        all_links = []
        link_type_counts: Dict[str, int] = {}

        for page_idx in page_indices:
            page = doc[page_idx]
            links = page.get_links()

            for link in links:
                link_info = {
                    "page": page_idx + 1,
                    "type": link.get("kind", 0),
                    "rect": list(link.get("from", [])) if link.get("from") else None,
                }

                # Map link kind to type name
                kind = link.get("kind", 0)
                if kind == 1:  # LINK_URI
                    link_info["type"] = "uri"
                    link_info["uri"] = link.get("uri", "")
                elif kind == 2:  # LINK_GOTO
                    link_info["type"] = "goto"
                    link_info["destination_page"] = link.get("page", 0) + 1
                elif kind == 3:  # LINK_GOTOR
                    link_info["type"] = "external_goto"
                    link_info["file"] = link.get("file", "")
                elif kind == 4:  # LINK_LAUNCH
                    link_info["type"] = "launch"
                    link_info["file"] = link.get("file", "")
                elif kind == 5:  # LINK_NAMED
                    link_info["type"] = "named"
                    link_info["name"] = link.get("name", "")
                else:
                    link_info["type"] = "unknown"

                all_links.append(link_info)
                
                # Count by type
                link_type = link_info["type"]
                link_type_counts[link_type] = link_type_counts.get(link_type, 0) + 1

        return {
            "pdf_path": str(path),
            "total_links": len(all_links),
            "links": all_links,
            "link_types": link_type_counts,
            "pages_scanned": len(page_indices),
        }
    finally:
        doc.close()


def optimize_pdf(
    pdf_path: str,
    output_path: str,
    quality: str = "medium",
) -> Dict[str, Any]:
    """
    Optimize/compress a PDF to reduce file size.

    Args:
        pdf_path: Path to the input PDF
        output_path: Path for the optimized PDF
        quality: Compression quality - "low", "medium", or "high"
                 low = maximum compression, high = minimum compression

    Returns:
        Dict with optimization results:
        - input_path: Original file path
        - output_path: Optimized file path
        - original_size: Original file size in bytes
        - optimized_size: New file size in bytes
        - compression_ratio: Ratio of new/original size
        - size_reduction_percent: Percentage of size reduced
    """
    path = _ensure_file(pdf_path)
    original_size = path.stat().st_size

    # Quality settings map to PyMuPDF garbage collection levels
    quality_map = {
        "low": 4,      # Maximum compression
        "medium": 3,   # Balanced
        "high": 2,     # Minimal compression
    }
    garbage_level = quality_map.get(quality, 3)

    doc = pymupdf.open(str(path))
    try:
        # Save with optimization options
        output = Path(output_path)
        doc.save(
            str(output),
            garbage=garbage_level,
            deflate=True,
            clean=True,
            deflate_images=True,
            deflate_fonts=True,
        )

        optimized_size = output.stat().st_size
        compression_ratio = optimized_size / original_size if original_size > 0 else 1.0
        reduction_percent = (1 - compression_ratio) * 100

        return {
            "input_path": str(path),
            "output_path": str(output),
            "original_size": original_size,
            "optimized_size": optimized_size,
            "compression_ratio": round(compression_ratio, 4),
            "size_reduction_percent": round(reduction_percent, 2),
            "quality_setting": quality,
        }
    finally:
        doc.close()


def detect_barcodes(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Detect and decode barcodes/QR codes in a PDF.

    Requires pyzbar library for barcode decoding.

    Args:
        pdf_path: Path to the PDF file
        pages: Optional list of page numbers (1-indexed). None = all pages.
        dpi: Resolution for rendering pages (higher = better detection)

    Returns:
        Dict with barcode information:
        - pdf_path: Path to the PDF
        - total_barcodes: Total number of barcodes found
        - barcodes: List of barcode details (page, type, data, position)
        - pages_scanned: Number of pages scanned
        - pyzbar_available: Whether pyzbar is installed
    """
    path = _ensure_file(pdf_path)
    doc = pymupdf.open(str(path))

    try:
        total_pages = len(doc)
        if pages:
            page_indices = _to_zero_based_pages(pages, total_pages)
        else:
            page_indices = list(range(total_pages))
        
        all_barcodes = []

        if _HAS_PYZBAR:
            for page_idx in page_indices:
                page = doc[page_idx]
                
                # Render page to image
                mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image for pyzbar
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Detect barcodes
                decoded_objects = pyzbar.decode(img)
                
                for obj in decoded_objects:
                    barcode_info = {
                        "page": page_idx + 1,
                        "type": obj.type,
                        "data": obj.data.decode("utf-8", errors="replace"),
                        "position": {
                            "left": obj.rect.left,
                            "top": obj.rect.top,
                            "width": obj.rect.width,
                            "height": obj.rect.height,
                        },
                    }
                    all_barcodes.append(barcode_info)

        return {
            "pdf_path": str(path),
            "total_barcodes": len(all_barcodes),
            "barcodes": all_barcodes,
            "pages_scanned": len(page_indices),
            "pyzbar_available": _HAS_PYZBAR,
        }
    finally:
        doc.close()


def split_pdf_by_bookmarks(
    pdf_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Split a PDF by its bookmarks (table of contents).

    Each bookmark creates a separate PDF file containing pages
    from that bookmark to the next one.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save split PDFs

    Returns:
        Dict with splitting results:
        - input_path: Original file path
        - output_dir: Output directory
        - total_bookmarks: Number of bookmarks found
        - files_created: List of created file details
    """
    path = _ensure_file(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(path))
    try:
        toc = doc.get_toc()  # [[level, title, page], ...]
        total_pages = len(doc)
        files_created = []

        if not toc:
            # No bookmarks - return single file or empty
            return {
                "input_path": str(path),
                "output_dir": str(out_dir),
                "total_bookmarks": 0,
                "files_created": [],
                "message": "No bookmarks found in PDF",
            }

        # Process bookmarks
        for i, bookmark in enumerate(toc):
            level, title, start_page = bookmark
            
            # Determine end page
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1  # Page before next bookmark
            else:
                end_page = total_pages

            # Create safe filename
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            safe_title = safe_title[:50].strip()  # Limit length
            output_file = out_dir / f"{i + 1:03d}_{safe_title}.pdf"

            # Extract pages
            new_doc = pymupdf.open()
            try:
                new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
                new_doc.save(str(output_file))
                
                files_created.append({
                    "path": str(output_file),
                    "title": title,
                    "page_range": f"{start_page}-{end_page}",
                    "page_count": end_page - start_page + 1,
                })
            finally:
                new_doc.close()

        return {
            "input_path": str(path),
            "output_dir": str(out_dir),
            "total_bookmarks": len(toc),
            "files_created": files_created,
        }
    finally:
        doc.close()


def split_pdf_by_pages(
    pdf_path: str,
    output_dir: str,
    pages_per_split: int = 1,
) -> Dict[str, Any]:
    """
    Split a PDF into multiple files by page count.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save split PDFs
        pages_per_split: Number of pages per output file

    Returns:
        Dict with splitting results:
        - input_path: Original file path
        - output_dir: Output directory
        - files_created: List of created file details
    """
    path = _ensure_file(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(path))
    try:
        total_pages = len(doc)
        files_created = []
        
        base_name = path.stem

        for start in range(0, total_pages, pages_per_split):
            end = min(start + pages_per_split - 1, total_pages - 1)
            
            output_file = out_dir / f"{base_name}_pages_{start + 1}-{end + 1}.pdf"

            new_doc = pymupdf.open()
            try:
                new_doc.insert_pdf(doc, from_page=start, to_page=end)
                new_doc.save(str(output_file))
                
                files_created.append({
                    "path": str(output_file),
                    "title": f"Pages {start + 1}-{end + 1}",
                    "page_range": f"{start + 1}-{end + 1}",
                    "page_count": end - start + 1,
                })
            finally:
                new_doc.close()

        return {
            "input_path": str(path),
            "output_dir": str(out_dir),
            "total_pages": total_pages,
            "pages_per_split": pages_per_split,
            "files_created": files_created,
        }
    finally:
        doc.close()


def compare_pdfs(
    pdf1_path: str,
    pdf2_path: str,
    compare_text: bool = True,
    compare_images: bool = False,
) -> Dict[str, Any]:
    """
    Compare two PDFs and identify differences.

    Args:
        pdf1_path: Path to the first PDF
        pdf2_path: Path to the second PDF
        compare_text: Whether to compare text content
        compare_images: Whether to compare images (slower)

    Returns:
        Dict with comparison results:
        - pdf1_path: First PDF path
        - pdf2_path: Second PDF path
        - are_identical: Whether PDFs are identical
        - differences: List of differences found
        - summary: Human-readable summary
    """
    path1 = _ensure_file(pdf1_path)
    path2 = _ensure_file(pdf2_path)

    doc1 = pymupdf.open(str(path1))
    doc2 = pymupdf.open(str(path2))

    try:
        differences = []
        
        # Compare page count
        if len(doc1) != len(doc2):
            differences.append({
                "type": "page_count",
                "pdf1_pages": len(doc1),
                "pdf2_pages": len(doc2),
                "description": f"Page count differs: {len(doc1)} vs {len(doc2)}",
            })

        # Compare text content per page
        if compare_text:
            min_pages = min(len(doc1), len(doc2))
            for page_idx in range(min_pages):
                text1 = doc1[page_idx].get_text().strip()
                text2 = doc2[page_idx].get_text().strip()
                
                if text1 != text2:
                    differences.append({
                        "type": "text",
                        "page": page_idx + 1,
                        "description": f"Text differs on page {page_idx + 1}",
                        "pdf1_text_length": len(text1),
                        "pdf2_text_length": len(text2),
                    })

        # Compare images if requested
        if compare_images:
            min_pages = min(len(doc1), len(doc2))
            for page_idx in range(min_pages):
                images1 = doc1[page_idx].get_images()
                images2 = doc2[page_idx].get_images()
                
                if len(images1) != len(images2):
                    differences.append({
                        "type": "images",
                        "page": page_idx + 1,
                        "description": f"Image count differs on page {page_idx + 1}",
                        "pdf1_images": len(images1),
                        "pdf2_images": len(images2),
                    })

        # Generate summary
        are_identical = len(differences) == 0
        if are_identical:
            summary = "PDFs are identical"
        else:
            diff_types = set(d["type"] for d in differences)
            summary = f"Found {len(differences)} difference(s): {', '.join(diff_types)}"

        return {
            "pdf1_path": str(path1),
            "pdf2_path": str(path2),
            "are_identical": are_identical,
            "differences": differences,
            "summary": summary,
            "pdf1_page_count": len(doc1),
            "pdf2_page_count": len(doc2),
        }
    finally:
        doc1.close()
        doc2.close()


def batch_process(
    pdf_paths: List[str],
    operation: str,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Process multiple PDFs with a single operation.

    Args:
        pdf_paths: List of PDF file paths
        operation: Operation to perform. Supported:
                   - "get_info": Get basic PDF info
                   - "extract_text": Extract text from each PDF
                   - "extract_links": Extract links from each PDF
                   - "optimize": Optimize each PDF (requires output_dir)
        output_dir: Directory for output files (required for some operations)
        **kwargs: Additional arguments for the operation

    Returns:
        Dict with batch results:
        - operation: The operation performed
        - total_files: Total number of files processed
        - successful: Number of successful operations
        - failed: Number of failed operations
        - results: List of individual results
    """
    supported_ops = ["get_info", "extract_text", "extract_links", "optimize"]
    if operation not in supported_ops:
        raise PdfToolError(f"Unsupported operation: {operation}. Supported: {supported_ops}")

    results = []
    successful = 0
    failed = 0

    for pdf_path in pdf_paths:
        try:
            if operation == "get_info":
                path = Path(pdf_path)
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {pdf_path}")
                doc = pymupdf.open(str(path))
                try:
                    result = {
                        "pdf_path": str(path),
                        "page_count": len(doc),
                        "metadata": doc.metadata,
                        "file_size": path.stat().st_size,
                    }
                finally:
                    doc.close()

            elif operation == "extract_text":
                result = extract_text_native(pdf_path)

            elif operation == "extract_links":
                result = extract_links(pdf_path)

            elif operation == "optimize":
                if not output_dir:
                    raise PdfToolError("output_dir required for optimize operation")
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = out_dir / f"optimized_{Path(pdf_path).name}"
                result = optimize_pdf(pdf_path, str(output_file), **kwargs)

            results.append({
                "pdf_path": pdf_path,
                "success": True,
                "result": result,
            })
            successful += 1

        except Exception as e:
            results.append({
                "pdf_path": pdf_path,
                "success": False,
                "error": str(e),
            })
            failed += 1

    return {
        "operation": operation,
        "total_files": len(pdf_paths),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
