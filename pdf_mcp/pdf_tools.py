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
- Agentic AI: LLM-powered form filling, entity extraction, document analysis (v0.8.0+)
- Local VLM: Cost-free local model integration via Qwen3-VL (v0.9.0+)

Version: 1.0.7
License: AGPL-3.0
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
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
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign import fields, signers, validation
    from pyhanko.sign.timestamps.requests_client import HTTPTimeStamper

    _HAS_PYHANKO = True
except ImportError:
    _HAS_PYHANKO = False

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import ollama as _ollama

    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

# LLM Backend Configuration
# Priority: local > ollama > openai (local is free, no API costs)
LLM_BACKEND_LOCAL = "local"
LLM_BACKEND_OLLAMA = "ollama"
LLM_BACKEND_OPENAI = "openai"

# Import LLM configuration from llm_setup (DRY - single source of truth)
from pdf_mcp import llm_setup
from pdf_mcp.llm_setup import LOCAL_MODEL_SERVER_URL, LOCAL_VLM_MODEL

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


def _has_xfa_form(reader: PdfReader) -> bool:
    try:
        root = reader.trailer.get("/Root")
        if not root:
            return False
        acro_form = root.get("/AcroForm")
        if not acro_form:
            return False
        acro_obj = acro_form.get_object() if hasattr(acro_form, "get_object") else acro_form
        return bool(acro_obj.get("/XFA"))
    except Exception:
        return False


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


def _field_tokens(value: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    if not tokens:
        return []
    stopwords = {
        "the",
        "and",
        "of",
        "to",
        "for",
        "please",
        "select",
        "check",
        "mark",
        "if",
        "applicable",
        "yes",
        "no",
    }
    return [t for t in tokens if t not in stopwords]


def _score_label_match(key: str, label_normalized: str, label_tokens: List[str]) -> int:
    key_normalized = _normalize_field_key(key)
    if not key_normalized:
        return 0
    if key_normalized == label_normalized:
        return 3
    if key_normalized in label_normalized or label_normalized in key_normalized:
        return 2
    key_tokens = _field_tokens(key)
    if not key_tokens or not label_tokens:
        return 0
    overlap = set(key_tokens) & set(label_tokens)
    if not overlap:
        return 0
    if len(overlap) == len(set(key_tokens)):
        return 2
    return 1


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


_FORM_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "client_intake_basic": {
        "description": "Basic client intake (contact + ID + consent)",
        "fields": [
            {"name": "Full Name", "type": "text", "rect": [50, 760, 300, 780]},
            {"name": "Date of Birth", "type": "text", "rect": [320, 760, 520, 780]},
            {"name": "Email", "type": "text", "rect": [50, 725, 300, 745]},
            {"name": "Phone", "type": "text", "rect": [320, 725, 520, 745]},
            {"name": "Address", "type": "text", "rect": [50, 690, 520, 710], "multiline": True},
            {"name": "Passport Number", "type": "text", "rect": [50, 645, 250, 665]},
            {"name": "Nationality", "type": "text", "rect": [270, 645, 520, 665]},
            {"name": "Travel Dates", "type": "text", "rect": [50, 610, 300, 630]},
            {"name": "Overseas Address", "type": "text", "rect": [50, 575, 520, 595], "multiline": True},
            {"name": "Consent", "type": "checkbox", "rect": [50, 535, 65, 550]},
        ],
    },
    "payment_receipt_basic": {
        "description": "Payment receipt (payer + amount + method)",
        "fields": [
            {"name": "Receipt Number", "type": "text", "rect": [50, 760, 240, 780]},
            {"name": "Receipt Date", "type": "text", "rect": [260, 760, 520, 780]},
            {"name": "Payer Name", "type": "text", "rect": [50, 725, 300, 745]},
            {"name": "Amount", "type": "text", "rect": [320, 725, 520, 745]},
            {"name": "Payment Method", "type": "text", "rect": [50, 690, 300, 710]},
            {"name": "Reference", "type": "text", "rect": [320, 690, 520, 710]},
            {"name": "Notes", "type": "text", "rect": [50, 650, 520, 670], "multiline": True},
        ],
    },
    "travel_authorization_basic": {
        "description": "Travel authorization (traveler + itinerary + signature)",
        "fields": [
            {"name": "Traveler Name", "type": "text", "rect": [50, 760, 300, 780]},
            {"name": "Passport Number", "type": "text", "rect": [320, 760, 520, 780]},
            {"name": "Departure Date", "type": "text", "rect": [50, 725, 240, 745]},
            {"name": "Return Date", "type": "text", "rect": [260, 725, 520, 745]},
            {"name": "Destination", "type": "text", "rect": [50, 690, 300, 710]},
            {"name": "Purpose of Travel", "type": "text", "rect": [50, 655, 520, 675], "multiline": True},
            {"name": "Approver Name", "type": "text", "rect": [50, 610, 300, 630]},
            {"name": "Signature", "type": "text", "rect": [320, 610, 520, 630]},
        ],
    },
}


def get_form_templates() -> Dict[str, Any]:
    """
    Return built-in form templates for common client workflows.

    Returns:
        Dict with template names, descriptions, and field definitions.
    """
    return {
        "templates": {
            name: {
                "description": meta["description"],
                "fields": meta["fields"],
            }
            for name, meta in _FORM_TEMPLATES.items()
        }
    }


def create_pdf_form_from_template(output_path: str, template_name: str) -> Dict[str, Any]:
    """
    Create a PDF form using a built-in template.

    Args:
        output_path: Path to the output PDF
        template_name: One of the names returned by get_form_templates()
    """
    template = _FORM_TEMPLATES.get(template_name)
    if not template:
        raise PdfToolError(f"Unknown template: {template_name}")
    return create_pdf_form(output_path=output_path, fields=template["fields"], pages=1)


def get_pdf_form_fields(pdf_path: str) -> Dict:
    path = _ensure_file(pdf_path)
    reader = PdfReader(str(path))
    if _has_xfa_form(reader):
        return {
            "error": "XFA forms are not supported. Convert to AcroForm or flatten first.",
            "xfa": True,
            "fields": {},
            "count": 0,
        }
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
    if _has_xfa_form(reader):
        raise PdfToolError(
            "XFA forms are not supported. Convert to AcroForm or flatten first."
        )
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
    if _has_xfa_form(reader):
        raise PdfToolError(
            "XFA forms are not supported. Convert to AcroForm or flatten first."
        )
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
            normalized_labels.append(
                {
                    "normalized": _normalize_field_key(label),
                    "tokens": _field_tokens(label),
                    "entry": entry,
                }
            )

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    filled = 0
    missing = []
    page_analysis = {p["page"]: p for p in detection.get("page_analysis", [])}
    used_indices = set()
    for key, value in data.items():
        best_score = 0
        best_index = None
        for idx, candidate in enumerate(normalized_labels):
            if idx in used_indices:
                continue
            score = _score_label_match(str(key), candidate["normalized"], candidate["tokens"])
            if score > best_score:
                best_score = score
                best_index = idx
        if best_index is None:
            missing.append(str(key))
            continue
        used_indices.add(best_index)
        match = normalized_labels[best_index]["entry"]

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

        if match.get("type") == "checkbox":
            if not _is_truthy(value):
                continue
            text_value = "X"
        else:
            text_value = str(value)
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
    float(mediabox.height)

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


def get_pdf_metadata(pdf_path: str, full: bool = False) -> Dict[str, Any]:
    """
    Return PDF document metadata.

    Args:
        pdf_path: Path to PDF file
        full: If True, include extended document info (page count, encryption status, file size).
              If False (default), return only basic metadata.

    Returns:
        Dict with metadata. When full=True, also includes 'document' key with extended info.
    """
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

    result: Dict[str, Any] = {"metadata": normalized}

    if full:
        result["document"] = {
            "page_count": len(reader.pages),
            "is_encrypted": bool(reader.is_encrypted),
            "file_size_bytes": os.path.getsize(path),
        }

    return result


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


def _parse_docmdp_permissions(value: Optional[str]):
    if value is None:
        return None
    normalized = value.strip().lower()
    mapping = {
        "no_changes": fields.MDPPerm.NO_CHANGES,
        "fill_forms": fields.MDPPerm.FILL_FORMS,
        "annotate": fields.MDPPerm.ANNOTATE,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise PdfToolError("docmdp_permissions must be one of: no_changes, fill_forms, annotate")


def _build_validation_context(
    signer: "signers.SimpleSigner",
    allow_fetching: bool,
    embed_validation_info: bool,
):
    if not allow_fetching and not embed_validation_info:
        return None
    trust_roots = None
    try:
        if signer.signing_cert is not None:
            trust_roots = [signer.signing_cert]
    except Exception:
        trust_roots = None
    return validation.ValidationContext(allow_fetching=allow_fetching, trust_roots=trust_roots)


def _sign_pdf(
    input_path: str,
    output_path: str,
    signer: "signers.SimpleSigner",
    field_name: str,
    certify: bool,
    reason: Optional[str],
    location: Optional[str],
    timestamp_url: Optional[str],
    embed_validation_info: bool,
    allow_fetching: bool,
    docmdp_permissions: Optional[str],
) -> Dict[str, Any]:
    src = _ensure_file(input_path)
    dst = _prepare_output(output_path)

    mdp_perm = _parse_docmdp_permissions(docmdp_permissions)
    validation_context = _build_validation_context(
        signer,
        allow_fetching=allow_fetching,
        embed_validation_info=embed_validation_info,
    )
    timestamper = HTTPTimeStamper(timestamp_url) if timestamp_url else None

    with src.open("rb") as inf:
        pdf_out = IncrementalPdfFileWriter(inf)
        writer = signers.PdfSigner(
            signers.PdfSignatureMetadata(
                field_name=field_name,
                certify=certify,
                reason=reason,
                location=location,
                embed_validation_info=embed_validation_info,
                validation_context=validation_context,
                docmdp_permissions=mdp_perm,
            ),
            signer=signer,
            timestamper=timestamper,
            new_field_spec=fields.SigFieldSpec(field_name),
        )

        def _sign() -> None:
            with dst.open("wb") as outf:
                asyncio.run(writer.async_sign_pdf(pdf_out, output=outf))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _sign()
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(_sign).result()

    return {"output_path": str(dst), "field_name": field_name, "certify": certify}


def sign_pdf(
    input_path: str,
    output_path: str,
    pfx_path: str,
    pfx_password: Optional[str] = None,
    field_name: str = "Signature1",
    certify: bool = True,
    reason: Optional[str] = None,
    location: Optional[str] = None,
    timestamp_url: Optional[str] = None,
    embed_validation_info: bool = False,
    allow_fetching: bool = False,
    docmdp_permissions: Optional[str] = "fill_forms",
) -> Dict[str, Any]:
    """
    Digitally sign a PDF using a PKCS#12/PFX certificate.
    """
    if not _HAS_PYHANKO:
        raise PdfToolError("pyHanko not available. Install pyhanko to sign PDFs.")
    pfx = _ensure_file(pfx_path)
    password_bytes = None if pfx_password is None else pfx_password.encode("utf-8")
    signer = signers.SimpleSigner.load_pkcs12(str(pfx), passphrase=password_bytes)
    return _sign_pdf(
        input_path,
        output_path,
        signer,
        field_name,
        certify,
        reason,
        location,
        timestamp_url,
        embed_validation_info,
        allow_fetching,
        docmdp_permissions,
    )


def sign_pdf_pem(
    input_path: str,
    output_path: str,
    key_path: str,
    cert_path: str,
    chain_paths: Optional[List[str]] = None,
    key_password: Optional[str] = None,
    field_name: str = "Signature1",
    certify: bool = True,
    reason: Optional[str] = None,
    location: Optional[str] = None,
    timestamp_url: Optional[str] = None,
    embed_validation_info: bool = False,
    allow_fetching: bool = False,
    docmdp_permissions: Optional[str] = "fill_forms",
) -> Dict[str, Any]:
    """
    Digitally sign a PDF using PEM key + certificate chain.
    """
    if not _HAS_PYHANKO:
        raise PdfToolError("pyHanko not available. Install pyhanko to sign PDFs.")
    key = _ensure_file(key_path)
    cert = _ensure_file(cert_path)
    chain = [str(_ensure_file(p)) for p in (chain_paths or [])]
    password_bytes = None if key_password is None else key_password.encode("utf-8")
    other_certs: List[Any] = []
    if chain:
        if hasattr(signers, "load_certs_from_pemder"):
            for path in chain:
                other_certs.extend(signers.load_certs_from_pemder(Path(path).read_bytes()))
        else:
            raise PdfToolError("pyHanko does not support loading cert chains from PEM files")
    signer = signers.SimpleSigner.load(
        key_file=str(key),
        cert_file=str(cert),
        key_passphrase=password_bytes,
        other_certs=other_certs or None,
    )
    return _sign_pdf(
        input_path,
        output_path,
        signer,
        field_name,
        certify,
        reason,
        location,
        timestamp_url,
        embed_validation_info,
        allow_fetching,
        docmdp_permissions,
    )


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
        re.compile(r"^[a-z0-9][a-z0-9 \-/#(),]{1,80}:\s*$", re.I),
        re.compile(r"^[a-z0-9].*_{3,}\s*$", re.I),
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
                result = extract_text(pdf_path, engine="native")

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


# =============================================================================
# Internal Implementation Functions (v0.7.0+)
# These are the core implementations called by the consolidated API.
# =============================================================================


def _extract_text_native_impl(pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """Internal: Extract text using native text layer only (no OCR)."""
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


def _extract_text_ocr_impl(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """Internal: Extract text with OCR support."""
    path = _ensure_file(pdf_path)

    valid_engines = ("auto", "native", "tesseract", "force_ocr", "ocr")
    if engine not in valid_engines:
        raise PdfToolError(f"Invalid engine: {engine}. Must be one of {valid_engines}")

    if engine in ("tesseract", "force_ocr", "ocr") and not _HAS_TESSERACT:
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

            native_text = ""
            if engine != "force_ocr":
                native_text = page.get_text("text").strip()
                page_result["native_chars"] = len(native_text)

            use_ocr_for_page = False
            if engine in ("tesseract", "force_ocr", "ocr"):
                use_ocr_for_page = True
            elif engine == "auto":
                has_images = len(page.get_images()) > 0
                insufficient_text = len(native_text) < 50
                use_ocr_for_page = has_images and insufficient_text

            ocr_text = ""
            if use_ocr_for_page and _HAS_TESSERACT:
                try:
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img, lang=language).strip()
                    page_result["ocr_chars"] = len(ocr_text)
                    ocr_used = True
                except Exception as e:
                    page_result["ocr_error"] = str(e)

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

        full_text = "\n\n--- Page Break ---\n\n".join(
            p["text"] for p in extracted_pages if p["text"]
        )

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


def _extract_text_smart_impl(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    native_threshold: int = 100,
    ocr_dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """Internal: Smart per-page method selection based on native text availability."""
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
        ocr_pages = 0
        native_pages = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            page_result: Dict[str, Any] = {"page": idx + 1}

            native_text = page.get_text("text").strip()
            native_chars = len(native_text)
            page_result["native_chars"] = native_chars

            if native_chars >= native_threshold:
                final_text = native_text
                page_result["method"] = "native"
                native_pages += 1
            else:
                if _HAS_TESSERACT:
                    try:
                        mat = pymupdf.Matrix(ocr_dpi / 72, ocr_dpi / 72)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, lang=language).strip()
                        if len(ocr_text) > native_chars:
                            final_text = ocr_text
                            page_result["method"] = "ocr"
                            page_result["ocr_chars"] = len(ocr_text)
                            ocr_pages += 1
                        else:
                            final_text = native_text
                            page_result["method"] = "native"
                            native_pages += 1
                    except Exception as e:
                        final_text = native_text
                        page_result["method"] = "native"
                        page_result["ocr_error"] = str(e)
                        native_pages += 1
                else:
                    final_text = native_text
                    page_result["method"] = "native"
                    native_pages += 1

            page_result["text"] = final_text
            page_result["char_count"] = len(final_text)
            extracted_pages.append(page_result)
            total_chars += len(final_text)

        full_text = "\n\n--- Page Break ---\n\n".join(
            p["text"] for p in extracted_pages if p["text"]
        )

        return {
            "pdf_path": str(path),
            "method": "smart",
            "native_threshold": native_threshold,
            "pages_extracted": len(extracted_pages),
            "native_pages": native_pages,
            "ocr_pages": ocr_pages,
            "total_chars": total_chars,
            "ocr_available": _HAS_TESSERACT,
            "text": full_text,
            "page_details": extracted_pages,
        }
    finally:
        doc.close()


def _extract_text_with_confidence_impl(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    language: str = "eng",
    dpi: int = 300,
    min_confidence: int = 0,
) -> Dict[str, Any]:
    """Internal: Extract text with OCR confidence scores."""
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

        page_details: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []
        total_words = 0
        confidence_sum = 0

        for idx in page_indices:
            page = doc.load_page(idx)
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)

            page_words: List[Dict[str, Any]] = []
            page_text_parts: List[str] = []

            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if word.strip() and conf >= min_confidence:
                    word_info = {
                        "text": word,
                        "confidence": conf,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    }
                    page_words.append(word_info)
                    page_text_parts.append(word)
                    confidence_sum += conf
                    total_words += 1

            page_text = " ".join(page_text_parts)
            avg_conf = sum(w["confidence"] for w in page_words) / len(page_words) if page_words else 0

            page_details.append({
                "page": idx + 1,
                "text": page_text,
                "word_count": len(page_words),
                "average_confidence": round(avg_conf, 2),
                "words": page_words,
            })
            all_text_parts.append(page_text)

        overall_avg = confidence_sum / total_words if total_words > 0 else 0

        return {
            "pdf_path": str(path),
            "language": language,
            "dpi": dpi,
            "min_confidence": min_confidence,
            "pages_extracted": len(page_details),
            "total_words": total_words,
            "overall_average_confidence": round(overall_avg, 2),
            "text": "\n\n--- Page Break ---\n\n".join(all_text_parts),
            "page_details": page_details,
        }
    finally:
        doc.close()


def _split_pdf_by_bookmarks_impl(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """Internal: Split PDF by bookmarks/table of contents."""
    path = _ensure_file(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(path))
    try:
        toc = doc.get_toc()
        total_pages = len(doc)
        files_created = []

        if not toc:
            return {
                "input_path": str(path),
                "output_dir": str(out_dir),
                "total_bookmarks": 0,
                "files_created": [],
                "message": "No bookmarks found in PDF",
            }

        for i, bookmark in enumerate(toc):
            level, title, start_page = bookmark
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = total_pages

            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            safe_title = safe_title[:50].strip()
            output_file = out_dir / f"{i + 1:03d}_{safe_title}.pdf"

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


def _split_pdf_by_pages_impl(
    pdf_path: str,
    output_dir: str,
    pages_per_split: int = 1,
) -> Dict[str, Any]:
    """Internal: Split PDF by page count."""
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


# =============================================================================
# Consolidated API (v0.7.0+)
# Unified tools for cleaner, more maintainable API surface.
# =============================================================================


def extract_text(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    include_confidence: bool = False,
    native_threshold: int = 100,
    dpi: int = 300,
    language: str = "eng",
    min_confidence: int = 0,
) -> Dict[str, Any]:
    """
    Unified text extraction with multiple engine options and optional confidence scores.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of 1-based page numbers (default: all pages)
        engine: Extraction engine selection:
            - "native": Native text layer only (fast, no OCR)
            - "auto": Try native first, fallback to OCR if insufficient
            - "smart": Per-page method selection based on native_threshold
            - "ocr" or "tesseract": Force OCR using Tesseract
            - "force_ocr": Always use OCR even if native text exists
        include_confidence: If True, return word-level OCR confidence scores
        native_threshold: Min chars to prefer native extraction in "smart" mode (default: 100)
        dpi: Resolution for OCR rendering (default: 300)
        language: Tesseract language code (default: "eng"). Use "+" for multiple: "eng+fra"
        min_confidence: Minimum confidence threshold 0-100 when include_confidence=True

    Returns:
        Dict with extracted text and metadata
    """
    # Use internal implementations directly
    if include_confidence:
        return _extract_text_with_confidence_impl(
            pdf_path, pages=pages, language=language, dpi=dpi, min_confidence=min_confidence
        )
    elif engine == "native":
        return _extract_text_native_impl(pdf_path, pages=pages)
    elif engine == "smart":
        return _extract_text_smart_impl(
            pdf_path, pages=pages, native_threshold=native_threshold, ocr_dpi=dpi, language=language
        )
    else:
        return _extract_text_ocr_impl(pdf_path, pages=pages, engine=engine, dpi=dpi, language=language)


def split_pdf(
    pdf_path: str,
    output_dir: str,
    mode: str = "pages",
    pages_per_split: int = 1,
) -> Dict[str, Any]:
    """
    Split a PDF into multiple files.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save split PDFs
        mode: Split mode:
            - "pages": Split by page count (uses pages_per_split)
            - "bookmarks": Split by table of contents/bookmarks
        pages_per_split: Number of pages per output file (only for mode="pages")

    Returns:
        Dict with splitting results
    """
    if mode not in ("pages", "bookmarks"):
        raise PdfToolError("mode must be 'pages' or 'bookmarks'")

    if mode == "bookmarks":
        return _split_pdf_by_bookmarks_impl(pdf_path, output_dir)
    else:
        return _split_pdf_by_pages_impl(pdf_path, output_dir, pages_per_split=pages_per_split)


def export_pdf(
    pdf_path: str,
    output_path: str,
    format: str = "markdown",
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """
    Export PDF content to different formats.

    Args:
        pdf_path: Path to the input PDF
        output_path: Path for the output file
        format: Export format:
            - "markdown": Export as Markdown
            - "json": Export as JSON with metadata
        pages: Optional list of 1-based page numbers (default: all pages)
        engine: Text extraction engine (see extract_text)
        dpi: Resolution for OCR (default: 300)
        language: Tesseract language code (default: "eng")

    Returns:
        Dict with export results
    """
    if format not in ("markdown", "json"):
        raise PdfToolError("format must be 'markdown' or 'json'")

    src = _ensure_file(pdf_path)
    dst = _prepare_output(output_path)
    
    # Extract text using unified extract_text
    text_result = extract_text(str(src), pages=pages, engine=engine, dpi=dpi, language=language)

    if format == "json":
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
    else:
        parts: List[str] = []
        for page in text_result.get("page_details", []):
            parts.append(f"# Page {page['page']}")
            parts.append(page["text"].rstrip())
            parts.append("")
        dst.write_text("\n".join(parts), encoding="utf-8")
        return {"output_path": str(dst), "engine": engine, "pages": len(text_result.get("page_details", []))}


# ============================================================================
# Agentic AI Functions (v0.8.0+) with Local VLM Support (v0.9.0+)
# ============================================================================


def _check_local_model_server() -> bool:
    """Check if local model server is available at localhost:8100."""
    if not _HAS_REQUESTS:
        return False
    try:
        response = _requests.get(f"{LOCAL_MODEL_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _get_llm_backend() -> str:
    """
    Determine which LLM backend to use.
    
    Priority: local > ollama > openai (local is free, no API costs)
    Can be overridden with PDF_MCP_LLM_BACKEND environment variable.
    """
    # Check for explicit override
    override = os.environ.get("PDF_MCP_LLM_BACKEND", "").lower()
    if override in (LLM_BACKEND_LOCAL, LLM_BACKEND_OLLAMA, LLM_BACKEND_OPENAI):
        return override
    
    # Auto-detect: prefer local (free) over paid APIs
    if _check_local_model_server():
        return LLM_BACKEND_LOCAL
    
    if _HAS_OLLAMA:
        return LLM_BACKEND_OLLAMA
    
    if _HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return LLM_BACKEND_OPENAI
    
    return ""  # No backend available


def _call_local_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Call local model server at localhost:8100.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt (prepended to prompt)
        model: Model to use (default: from LOCAL_VLM_MODEL env var)
    
    Returns:
        LLM response content or None if unavailable
    """
    if not _HAS_REQUESTS:
        return None
    
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    try:
        response = _requests.post(
            f"{LOCAL_MODEL_SERVER_URL}/generate",
            json={
                "prompt": full_prompt,
                "model": model or LOCAL_VLM_MODEL,
                "max_tokens": 1024,
            },
            timeout=120,  # Local models can be slow on first load
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        return None
    except Exception:
        return None


def _call_ollama_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "qwen3-vl:8b",
) -> Optional[str]:
    """
    Call Ollama LLM with the given prompt.
    
    Uses Qwen3-VL by default for vision-language capabilities
    (OCR accuracy improvement on PDF page images).
    
    Qwen3 models use a "thinking" mode by default that generates
    internal reasoning tokens before the answer.  We set a generous
    num_predict budget (4096) and keep_alive (10m) so the model has
    room for thinking tokens plus the actual response, and does not
    get unloaded between requests in the same session.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        model: Ollama model to use (default: qwen3-vl:8b)
    
    Returns:
        LLM response content or None if unavailable
    """
    if not _HAS_OLLAMA:
        return None
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = _ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            options={"num_predict": 4096},
            keep_alive="10m",
        )
        # Ollama returns Pydantic ChatResponse; access via attribute
        return response.message.content
    except Exception:
        return None


def _call_openai_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Optional[str]:
    """
    Call OpenAI LLM with the given prompt.

    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: Sampling temperature (0.0 for deterministic)

    Returns:
        LLM response content or None if unavailable
    """
    if not _HAS_OPENAI:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "auto",
    temperature: float = 0.0,
    backend: Optional[str] = None,
) -> Optional[str]:
    """
    Call LLM using the best available backend.
    
    Priority: local > ollama > openai (local is free, no API costs)
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        model: Model to use (default: auto-select based on backend)
        temperature: Sampling temperature (0.0 for deterministic, OpenAI only)
        backend: Force specific backend (local, ollama, openai)

    Returns:
        LLM response content or None if unavailable
    """
    selected_backend = backend or _get_llm_backend()
    
    if selected_backend == LLM_BACKEND_LOCAL:
        return _call_local_llm(prompt, system_prompt, model if model != "auto" else None)
    
    if selected_backend == LLM_BACKEND_OLLAMA:
        ollama_model = model if model != "auto" else llm_setup.get_ollama_model_name()
        return _call_ollama_llm(prompt, system_prompt, ollama_model)
    
    if selected_backend == LLM_BACKEND_OPENAI:
        openai_model = model if model != "auto" else "gpt-4o-mini"
        return _call_openai_llm(prompt, system_prompt, openai_model, temperature)
    
    return None


def get_llm_backend_info() -> Dict[str, Any]:
    """
    Get information about available LLM backends.
    
    Returns:
        Dict with backend availability and current selection
    """
    return {
        "current_backend": _get_llm_backend(),
        "backends": {
            "local": {
                "available": _check_local_model_server(),
                "url": LOCAL_MODEL_SERVER_URL,
                "model": LOCAL_VLM_MODEL,
                "cost": "free",
            },
            "ollama": {
                "available": _HAS_OLLAMA,
                "cost": "free",
            },
            "openai": {
                "available": _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY")),
                "cost": "paid (per token)",
            },
        },
        "override_env": "PDF_MCP_LLM_BACKEND",
    }


def auto_fill_pdf_form(
    pdf_path: str,
    output_path: str,
    source_data: Dict[str, Any],
    model: str = "auto",
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Intelligently fill PDF form fields using LLM-powered field mapping.

    This function analyzes form field names and source data keys to create
    intelligent mappings, even when names don't exactly match. For example,
    it can map "full_name" in the source to "Name" in the form.
    
    Uses local VLM by default (free, no API costs). Falls back to Ollama or OpenAI.

    Args:
        pdf_path: Path to the input PDF form
        output_path: Path for the filled output PDF
        source_data: Dictionary of data to fill into the form
        model: Model to use (default: auto-select based on backend)
        backend: Force specific backend: "local", "ollama", or "openai" (default: auto)

    Returns:
        Dict with:
            - filled_fields: Number of fields successfully filled
            - mappings: Dict showing source->field mappings used
            - unmapped_fields: List of form fields that couldn't be mapped
            - output_path: Path to the output file
            - backend: Which LLM backend was used

    Example:
        >>> source = {"name": "John Smith", "email_address": "john@example.com"}
        >>> result = auto_fill_pdf_form("form.pdf", "filled.pdf", source)
        >>> print(result["filled_fields"])  # May fill "Full Name" and "Email"
    """
    try:
        src = _ensure_file(pdf_path)
    except PdfToolError as e:
        return {"error": str(e)}

    # Get form fields
    fields_result = get_pdf_form_fields(str(src))
    if "error" in fields_result:
        return fields_result

    form_fields = fields_result.get("fields", {})
    if not form_fields:
        return {"error": "No form fields found in PDF"}

    field_names = list(form_fields.keys())

    # Try direct mapping first (exact or normalized matches)
    direct_mappings = {}
    for source_key, source_value in source_data.items():
        normalized_source = _normalize_field_key(source_key)
        for field_name in field_names:
            normalized_field = _normalize_field_key(field_name)
            if normalized_source == normalized_field:
                direct_mappings[field_name] = str(source_value)
                break

    # If LLM available and there are unmapped fields, use LLM for intelligent mapping
    llm_mappings = {}
    unmapped_source_keys = [k for k in source_data.keys() if _normalize_field_key(k) not in 
                           [_normalize_field_key(f) for f in direct_mappings.keys()]]
    unmapped_fields = [f for f in field_names if f not in direct_mappings]

    used_backend = None
    if unmapped_source_keys and unmapped_fields:
        # Check for available LLM backend
        selected_backend = backend or _get_llm_backend()
        if not selected_backend:
            return {
                "error": "No LLM backend available. Options: start local model server, install ollama, or set OPENAI_API_KEY",
                "hint": "Start local server: cd pdf-mcp-server && ./scripts/run_local_vlm.sh",
                "partial_mappings": direct_mappings
            }

        # Build LLM prompt for intelligent mapping
        system_prompt = """You are a form field mapping assistant. Given source data keys and PDF form field names, 
        determine the best mapping between them. Return ONLY a valid JSON object mapping form field names to values.
        Only include fields where you're confident in the mapping. Be conservative - don't map if unsure."""

        prompt = f"""Map the following source data to PDF form fields.

Source data keys and values:
{json.dumps({k: source_data[k] for k in unmapped_source_keys}, indent=2)}

Available PDF form field names (unmapped):
{json.dumps(unmapped_fields, indent=2)}

Return a JSON object where keys are PDF form field names and values are the corresponding source values.
Only include mappings you're confident about."""

        llm_response = _call_llm(prompt, system_prompt, model=model, backend=selected_backend)
        used_backend = selected_backend
        if llm_response:
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = llm_response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                llm_mappings = json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass  # Fall back to direct mappings only

    # Combine mappings
    all_mappings = {**direct_mappings, **llm_mappings}

    if not all_mappings:
        return {
            "error": "Could not map any source data to form fields",
            "form_fields": field_names,
            "source_keys": list(source_data.keys())
        }

    # Fill the form
    fill_result = fill_pdf_form(str(src), output_path, all_mappings, flatten=False)

    return {
        "output_path": output_path,
        "filled_fields": fill_result.get("filled", 0),
        "mappings": all_mappings,
        "unmapped_fields": [f for f in field_names if f not in all_mappings],
        "method": "llm" if llm_mappings else "direct",
        "backend": used_backend,
    }


def _parse_mrz_date(value: str) -> Optional[str]:
    if not value or len(value) != 6 or not value.isdigit():
        return None
    year = int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])
    current_year = date.today().year % 100
    century = 2000 if year <= current_year else 1900
    full_year = century + year
    try:
        return f"{full_year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


def _normalize_issue_date(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip()
    formats = (
        "%d %b %Y",
        "%d %B %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d-%m-%y",
        "%m-%d-%y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_mrz_lines(text: str) -> Optional[tuple[str, str]]:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s", "", line.strip())
        if "<" in cleaned and len(cleaned) >= 30:
            lines.append(cleaned)
    for i in range(len(lines) - 1):
        if len(lines[i]) == 44 and len(lines[i + 1]) == 44:
            return lines[i], lines[i + 1]
    return None


def _extract_passport_fields(full_text: str) -> tuple[Dict[str, Any], Dict[str, float]]:
    extracted: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}

    mrz = _extract_mrz_lines(full_text)
    if mrz:
        line1, line2 = mrz
        if line1.startswith("P<"):
            issuing_country = line1[2:5].replace("<", "").strip()
            names = line1[5:]
            surname = ""
            given_names = ""
            if "<<" in names:
                surname_part, given_part = names.split("<<", 1)
                surname = surname_part.replace("<", " ").strip()
                given_names = given_part.replace("<", " ").strip()
            passport_number = line2[0:9].replace("<", "").strip()
            nationality = line2[10:13].replace("<", "").strip()
            birth_raw = line2[13:19]
            sex = line2[20:21].replace("<", "").strip()
            expiry_raw = line2[21:27]
            personal_number = line2[28:42].replace("<", "").strip()

            birth_date = _parse_mrz_date(birth_raw)
            expiry_date = _parse_mrz_date(expiry_raw)

            extracted.update({
                "passport_number": passport_number or None,
                "issuing_country": issuing_country or None,
                "nationality": nationality or None,
                "surname": surname or None,
                "given_names": given_names or None,
                "birth_date": birth_date or birth_raw,
                "sex": sex or None,
                "expiry_date": expiry_date or expiry_raw,
                "personal_number": personal_number or None,
            })
            confidence.update({
                "passport_number": 0.85,
                "issuing_country": 0.8,
                "nationality": 0.8,
                "surname": 0.75,
                "given_names": 0.75,
                "birth_date": 0.85,
                "sex": 0.9,
                "expiry_date": 0.85,
                "personal_number": 0.6,
            })

    def _label_value(patterns: list[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    if not extracted.get("surname"):
        surname_value = _label_value([
            r"(?:surname|last name)\s*[:\-]?\s*([^\n\r]+)",
        ])
        if surname_value:
            extracted["surname"] = surname_value
            confidence["surname"] = 0.55

    if not extracted.get("given_names"):
        given_value = _label_value([
            r"(?:given names?|first name|forename)\s*[:\-]?\s*([^\n\r]+)",
        ])
        if given_value:
            extracted["given_names"] = given_value
            confidence["given_names"] = 0.55

    if not extracted.get("nationality"):
        nationality_value = _label_value([
            r"(?:nationality)\s*[:\-]?\s*([^\n\r]+)",
        ])
        if nationality_value:
            extracted["nationality"] = nationality_value
            confidence["nationality"] = 0.55

    if not extracted.get("issuing_country"):
        issuing_country_value = _label_value([
            r"(?:issuing country|country of issue)\s*[:\-]?\s*([^\n\r]+)",
        ])
        if issuing_country_value:
            extracted["issuing_country"] = issuing_country_value
            confidence["issuing_country"] = 0.55

    if not extracted.get("passport_number"):
        passport_number_value = _label_value([
            r"(?:passport number|passport no\.?|document number)\s*[:\-]?\s*([^\n\r]+)",
        ])
        if passport_number_value:
            extracted["passport_number"] = passport_number_value.replace("<", "").strip()
            confidence["passport_number"] = 0.55

    issue_date_patterns = [
        r"(?:date of issue|issue date|date of issuance|issued on|issued)\s*[:\-]?\s*([0-9]{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:\u7b7e\u53d1\u65e5\u671f|\u53d1\u8bc1\u65e5\u671f)\s*[:\-\uFF1A]?\s*([0-9]{4}[.\-/][0-9]{1,2}[.\-/][0-9]{1,2})",
    ]
    for pattern in issue_date_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            raw_issue_date = match.group(1).strip()
            normalized_issue_date = _normalize_issue_date(raw_issue_date)
            extracted["issue_date"] = normalized_issue_date or raw_issue_date
            confidence["issue_date"] = 0.6
            break

    issuing_authority_patterns = [
        r"(?:issuing authority|issue authority|issuing office|authority|place of issue|place of issuance)\s*[:\-]?\s*([^\n\r]+)",
        r"(?:\u7b7e\u53d1\u673a\u5173|\u7b7e\u53d1\u5730)\s*[:\-\uFF1A]?\s*([^\n\r]+)",
    ]
    for pattern in issuing_authority_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted["issuing_authority"] = match.group(1).strip()
            confidence["issuing_authority"] = 0.6
            break

    return extracted, confidence


def extract_structured_data(
    pdf_path: str,
    data_type: Optional[str] = None,
    schema: Optional[Dict[str, str]] = None,
    pages: Optional[List[int]] = None,
    ocr_engine: str = "auto",
    ocr_language: str = "eng",
    model: str = "auto",
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured data from PDF using pattern matching or LLM.

    Supports common document types (invoice, receipt, contract) with
    pre-defined extraction patterns, or custom schemas for specific needs.
    
    Uses local VLM by default (free, no API costs). Falls back to Ollama or OpenAI.

    Args:
        pdf_path: Path to the PDF file
        data_type: Predefined type: "invoice", "receipt", "contract", "form", "passport", or None
        schema: Custom extraction schema as Dict[field_name, field_type]
                Types: "string", "number", "date", "currency", "list"
        pages: Optional list of 1-based page numbers (default: all)
        ocr_engine: OCR engine for image-based docs ("auto", "ocr", "tesseract", "force_ocr")
        ocr_language: Tesseract language code (default: "eng")
        model: Model to use (default: auto-select based on backend)
        backend: Force specific backend: "local", "ollama", or "openai" (default: auto)

    Returns:
        Dict with:
            - data: Extracted structured data
            - confidence: Extraction confidence scores
            - method: "pattern" or "llm" or "llm+pattern"
            - page_count: Number of pages processed
            - backend: Which LLM backend was used (if any)

    Example:
        >>> result = extract_structured_data("invoice.pdf", data_type="invoice")
        >>> print(result["data"]["total"])  # Extracted total amount
    """
    try:
        src = _ensure_file(pdf_path)
    except PdfToolError as e:
        return {"error": str(e)}

    # Extract text from PDF
    text_result = extract_text(str(src), pages=pages, engine=ocr_engine, language=ocr_language)
    if "error" in text_result:
        return text_result

    full_text = text_result.get("text", "")
    if not full_text.strip():
        return {"error": "No text content found in PDF", "page_count": text_result.get("page_count", 0)}

    if data_type == "passport":
        extracted_data, confidence = _extract_passport_fields(full_text)
        return {
            "data": extracted_data,
            "confidence": confidence,
            "method": "passport",
            "page_count": text_result.get("page_count", 0),
            "data_type": data_type,
            "backend": None,
            "backend_available": None,
        }

    # Define patterns for common data types
    patterns = {
        "invoice": {
            "invoice_number": r"(?:invoice|inv)[\s#:]*([A-Z0-9-]+)",
            "date": r"(?:date|dated?)[\s:]*(\d{1,2}[\s/-]\w+[\s/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
            "total": r"(?:total|amount due|grand total)[\s:]*\$?([\d,]+\.?\d*)",
            "subtotal": r"(?:subtotal|sub-total)[\s:]*\$?([\d,]+\.?\d*)",
            "tax": r"(?:tax|vat|gst)[\s:]*\$?([\d,]+\.?\d*)",
            "due_date": r"(?:due date|payment due|due by)[\s:]*(\d{1,2}[\s/-]\w+[\s/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
        },
        "receipt": {
            "store_name": r"^([A-Z][A-Za-z\s&]+)(?:\n|$)",
            "date": r"(?:date)[\s:]*(\d{1,2}[\s/-]\d{1,2}[\s/-]\d{2,4})",
            "total": r"(?:total)[\s:]*\$?([\d,]+\.?\d*)",
            "payment_method": r"(?:paid by|payment|card)[\s:]*(\w+)",
        },
        "contract": {
            "effective_date": r"(?:effective date|dated)[\s:]*(\w+\s+\d{1,2},?\s+\d{4})",
            "parties": r"(?:between|party)[\s:]*([A-Z][A-Za-z\s,]+)(?:and|&)",
            "term": r"(?:term|duration)[\s:]*(\d+\s*(?:year|month|day)s?)",
        },
    }

    extracted_data = {}
    confidence = {}
    method = "pattern"

    # Try pattern-based extraction first
    if data_type and data_type in patterns:
        for field, pattern in patterns[data_type].items():
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_data[field] = match.group(1).strip()
                confidence[field] = 0.7  # Pattern match confidence

    # Try custom schema
    if schema:
        for field, field_type in schema.items():
            if field not in extracted_data:
                # Generate pattern based on field name and type
                field_pattern = field.replace("_", "[\\s_]")
                pattern = r"(?:" + field_pattern + r")[\s:]*"
                if field_type == "number":
                    pattern += r"([\d,]+\.?\d*)"
                elif field_type == "currency":
                    pattern += r"\$?([\d,]+\.?\d*)"
                elif field_type == "date":
                    pattern += r"(\d{1,2}[\s/-]\w+[\s/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})"
                else:
                    pattern += r"([^\n]+)"
                
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    confidence[field] = 0.5  # Lower confidence for dynamic patterns

    # If LLM available and we have a schema or data_type, enhance with LLM
    used_backend = None
    selected_backend = backend or _get_llm_backend()
    if (not extracted_data or len(extracted_data) < 3) and selected_backend:
        target_schema = schema or patterns.get(data_type, {})
        if target_schema:
            system_prompt = """You are a document data extraction assistant. Extract structured data from the given text.
            Return ONLY a valid JSON object with the requested fields. Use null for fields you cannot find."""

            fields_to_extract = list(target_schema.keys()) if isinstance(target_schema, dict) else list(target_schema)
            prompt = f"""Extract the following fields from this document text:
            
Fields to extract: {json.dumps(fields_to_extract)}

Document text:
{full_text[:4000]}

Return a JSON object with the extracted values."""

            llm_response = _call_llm(prompt, system_prompt, model=model, backend=selected_backend)
            used_backend = selected_backend
            if llm_response:
                try:
                    json_str = llm_response
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    llm_data = json.loads(json_str.strip())
                    
                    # Merge LLM data with pattern data (patterns take precedence)
                    for key, value in llm_data.items():
                        if key not in extracted_data and value is not None:
                            extracted_data[key] = value
                            confidence[key] = 0.85  # LLM confidence
                    method = "llm+pattern" if any(c == 0.7 for c in confidence.values()) else "llm"
                except json.JSONDecodeError:
                    pass

    return {
        "data": extracted_data,
        "confidence": confidence,
        "method": method,
        "page_count": text_result.get("page_count", 0),
        "data_type": data_type,
        "backend": used_backend,
        "backend_available": selected_backend if selected_backend else None,
    }


def analyze_pdf_content(
    pdf_path: str,
    include_summary: bool = True,
    detect_entities: bool = True,
    check_completeness: bool = False,
    model: str = "auto",
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze PDF content for document type, key entities, and summary.

    Provides comprehensive document analysis including classification,
    entity extraction, and optional completeness checking.
    
    Uses local VLM by default (free, no API costs). Falls back to Ollama or OpenAI.

    Args:
        pdf_path: Path to the PDF file
        include_summary: Generate document summary (default: True)
        detect_entities: Extract key entities like dates, amounts, names (default: True)
        check_completeness: Check for missing required fields (default: False)
        model: Model to use (default: auto-select based on backend)
        backend: Force specific backend: "local", "ollama", or "openai" (default: auto)

    Returns:
        Dict with:
            - document_type: Classified type (invoice, contract, form, letter, report, other)
            - summary: Brief document summary (if requested)
            - entities: Extracted key entities (if requested)
            - completeness: Completeness analysis (if requested)
            - page_count: Number of pages
            - word_count: Approximate word count
            - backend: Which LLM backend was used (if any)

    Example:
        >>> result = analyze_pdf_content("document.pdf")
        >>> print(result["document_type"])  # "invoice"
        >>> print(result["summary"])  # "Invoice #12345 for $162.00..."
    """
    try:
        src = _ensure_file(pdf_path)
    except PdfToolError as e:
        return {"error": str(e)}

    # Extract text
    text_result = extract_text(str(src), engine="auto")
    if "error" in text_result:
        return text_result

    full_text = text_result.get("text", "")
    page_count = text_result.get("page_count", 0)
    word_count = len(full_text.split())

    # Basic document classification using patterns
    doc_type_patterns = {
        "invoice": r"(?:invoice|bill|statement)",
        "receipt": r"(?:receipt|paid|payment received)",
        "contract": r"(?:agreement|contract|terms and conditions|hereby agree)",
        "form": r"(?:please fill|form|application|submit)",
        "letter": r"(?:dear|sincerely|regards|to whom)",
        "report": r"(?:report|analysis|findings|conclusion|executive summary)",
        "resume": r"(?:experience|education|skills|employment|curriculum vitae|cv)",
    }

    document_type = "other"
    type_confidence = 0.0
    text_lower = full_text.lower()

    for doc_type, pattern in doc_type_patterns.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > 0:
            conf = min(0.9, 0.3 + matches * 0.15)
            if conf > type_confidence:
                document_type = doc_type
                type_confidence = conf

    result = {
        "document_type": document_type,
        "type_confidence": round(type_confidence, 2),
        "page_count": page_count,
        "word_count": word_count,
    }

    # Entity detection using patterns
    if detect_entities:
        entities = {}
        
        # Dates
        date_pattern = r"\b(\d{1,2}[\s/-]\w+[\s/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
        dates = re.findall(date_pattern, full_text)
        if dates:
            entities["dates"] = list(set(dates[:10]))  # Limit to 10 unique dates

        # Currency amounts
        currency_pattern = r"\$[\d,]+\.?\d*|\d+\.\d{2}\s*(?:USD|EUR|GBP)"
        amounts = re.findall(currency_pattern, full_text)
        if amounts:
            entities["amounts"] = list(set(amounts[:10]))

        # Email addresses
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = re.findall(email_pattern, full_text)
        if emails:
            entities["emails"] = list(set(emails[:5]))

        # Phone numbers
        phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        phones = re.findall(phone_pattern, full_text)
        if phones:
            entities["phones"] = list(set(phones[:5]))

        # Names (simple pattern - capitalized words)
        name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        names = re.findall(name_pattern, full_text)
        if names:
            # Filter common non-names
            filtered_names = [n for n in names if n.lower() not in 
                           ["new york", "los angeles", "united states", "january", "february"]]
            entities["names"] = list(set(filtered_names[:10]))

        result["entities"] = entities

    # LLM-based summary and enhanced analysis
    used_backend = None
    selected_backend = backend or _get_llm_backend()
    if (include_summary or check_completeness) and selected_backend:
        system_prompt = """You are a document analysis assistant. Analyze the given document and provide:
        1. A brief 1-2 sentence summary
        2. Document classification confirmation
        3. Key findings or notable items
        Return as a JSON object with keys: summary, document_type, key_findings (list)"""

        prompt = f"""Analyze this {document_type} document:

{full_text[:4000]}

Provide a JSON response with summary, document_type, and key_findings."""

        llm_response = _call_llm(prompt, system_prompt, model=model, backend=selected_backend)
        used_backend = selected_backend
        if llm_response:
            try:
                json_str = llm_response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                analysis = json.loads(json_str.strip())
                
                if include_summary and "summary" in analysis:
                    result["summary"] = analysis["summary"]
                if "document_type" in analysis:
                    result["document_type"] = analysis["document_type"]
                    result["type_confidence"] = 0.9
                if "key_findings" in analysis:
                    result["key_findings"] = analysis["key_findings"]
            except json.JSONDecodeError:
                # Fallback: use raw response as summary
                if include_summary:
                    result["summary"] = llm_response[:500]
    elif include_summary:
        # Simple extractive summary without LLM
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        result["summary"] = ". ".join(sentences[:3]) + "." if sentences else "Unable to generate summary."

    # Completeness check
    if check_completeness:
        completeness = {"score": 1.0, "missing_fields": []}
        
        # Check for common required elements based on document type
        required_elements = {
            "invoice": ["date", "total", "invoice number"],
            "contract": ["date", "signature", "parties"],
            "form": ["date", "signature"],
        }
        
        if document_type in required_elements:
            for element in required_elements[document_type]:
                if element not in text_lower:
                    completeness["missing_fields"].append(element)
                    completeness["score"] -= 0.2
            completeness["score"] = max(0, completeness["score"])
        
        result["completeness"] = completeness

    result["analysis_method"] = "llm+pattern" if used_backend else "pattern"
    result["backend"] = used_backend
    
    return result
