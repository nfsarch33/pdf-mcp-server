"""
PDF MCP Server - Model Context Protocol server for PDF operations.

This module exposes PDF tools via the MCP protocol for use with AI assistants.
Run with: python -m pdf_mcp.server

Available tool categories:
- Form handling (6 tools)
- Page operations (6 tools)
- Annotations & text (14 tools)
- Signatures & security (5 tools)
- Metadata (4 tools)
- Export (2 tools)
- OCR & text extraction (8 tools)
- Table/image extraction (3 tools)
- Form detection (1 tool)
- PII detection (1 tool)
 - PII detection (1 tool)

Version: 0.5.0
License: AGPL-3.0
"""
from __future__ import annotations

import functools
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp.server.fastmcp import FastMCP

from . import pdf_tools
from .pdf_tools import PdfToolError

mcp = FastMCP("PDF Handler")


def _wrap_result(result: Any) -> Any:
    if isinstance(result, Path):
        return str(result)
    return result


def _handle_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return _wrap_result(fn(*args, **kwargs))
        except PdfToolError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": f"Unexpected error: {exc}", "trace": traceback.format_exc()}

    return wrapper


@mcp.tool()
@_handle_errors
def get_pdf_form_fields(pdf_path: str) -> Dict[str, Any]:
    """Return available form fields in the PDF."""
    return pdf_tools.get_pdf_form_fields(pdf_path)


@mcp.tool()
@_handle_errors
def fill_pdf_form(
    input_path: str,
    output_path: str,
    data: Dict[str, str],
    flatten: bool = False,
) -> Dict[str, Any]:
    """Fill a PDF form with provided data. Optionally flatten to make non-editable."""
    return pdf_tools.fill_pdf_form(input_path, output_path, data, flatten)


@mcp.tool()
@_handle_errors
def fill_pdf_form_any(
    input_path: str,
    output_path: str,
    data: Dict[str, Any],
    flatten: bool = False,
) -> Dict[str, Any]:
    """Fill standard or non-standard forms using label detection when needed."""
    return pdf_tools.fill_pdf_form_any(input_path, output_path, data, flatten)


@mcp.tool()
@_handle_errors
def create_pdf_form(
    output_path: str,
    fields: List[Dict[str, Any]],
    page_size: Optional[Sequence[float]] = None,
    pages: int = 1,
) -> Dict[str, Any]:
    """Create a new PDF with AcroForm fields."""
    return pdf_tools.create_pdf_form(output_path, fields, page_size=page_size, pages=pages)


@mcp.tool()
@_handle_errors
def flatten_pdf(input_path: str, output_path: str) -> Dict[str, Any]:
    """Flatten a PDF (remove form fields/annotations)."""
    return pdf_tools.flatten_pdf(input_path, output_path)


@mcp.tool()
@_handle_errors
def clear_pdf_form_fields(
    input_path: str,
    output_path: str,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Clear (delete) values for PDF form fields while keeping fields fillable."""
    return pdf_tools.clear_pdf_form_fields(input_path, output_path, fields=fields)


@mcp.tool()
@_handle_errors
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
) -> Dict[str, Any]:
    """Encrypt (password-protect) a PDF using pypdf."""
    return pdf_tools.encrypt_pdf(
        input_path=input_path,
        output_path=output_path,
        user_password=user_password,
        owner_password=owner_password,
        allow_printing=allow_printing,
        allow_modifying=allow_modifying,
        allow_copying=allow_copying,
        allow_annotations=allow_annotations,
        allow_form_filling=allow_form_filling,
        use_128bit=use_128bit,
    )


@mcp.tool()
@_handle_errors
def merge_pdfs(pdf_list: List[str], output_path: str) -> Dict[str, Any]:
    """Merge multiple PDFs into a single file."""
    return pdf_tools.merge_pdfs(pdf_list, output_path)


@mcp.tool()
@_handle_errors
def extract_pages(input_path: str, pages: List[int], output_path: str) -> Dict[str, Any]:
    """Extract specific 1-based pages into a new PDF."""
    return pdf_tools.extract_pages(input_path, pages, output_path)


@mcp.tool()
@_handle_errors
def rotate_pages(
    input_path: str,
    pages: List[int],
    degrees: int,
    output_path: str,
) -> Dict[str, Any]:
    """Rotate specified 1-based pages by degrees (must be multiple of 90)."""
    return pdf_tools.rotate_pages(input_path, pages, degrees, output_path)


@mcp.tool()
@_handle_errors
def reorder_pages(input_path: str, pages: List[int], output_path: str) -> Dict[str, Any]:
    """Reorder pages in a PDF using a 1-based page list."""
    return pdf_tools.reorder_pages(input_path, pages, output_path)


@mcp.tool()
@_handle_errors
def add_text_annotation(
    input_path: str,
    page: int,
    text: str,
    output_path: str,
    rect: Optional[Sequence[float]] = None,
    annotation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a FreeText annotation to a page (managed text insertion)."""
    return pdf_tools.add_text_annotation(
        input_path, page, text, output_path, rect=rect, annotation_id=annotation_id
    )


@mcp.tool()
@_handle_errors
def update_text_annotation(
    input_path: str,
    output_path: str,
    annotation_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Update an existing annotation by annotation_id."""
    return pdf_tools.update_text_annotation(
        input_path, output_path, annotation_id, text, pages=pages
    )


@mcp.tool()
@_handle_errors
def remove_text_annotation(
    input_path: str,
    output_path: str,
    annotation_id: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Remove an existing annotation by annotation_id."""
    return pdf_tools.remove_text_annotation(input_path, output_path, annotation_id, pages=pages)


@mcp.tool()
@_handle_errors
def remove_annotations(
    input_path: str,
    output_path: str,
    pages: List[int],
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove annotations from given pages. Optionally filter by subtype (e.g., FreeText)."""
    return pdf_tools.remove_annotations(input_path, output_path, pages, subtype=subtype)


@mcp.tool()
@_handle_errors
def insert_pages(
    input_path: str,
    insert_from_path: str,
    at_page: int,
    output_path: str,
) -> Dict[str, Any]:
    """Insert pages from another PDF before at_page (1-based)."""
    return pdf_tools.insert_pages(input_path, insert_from_path, at_page, output_path)


@mcp.tool()
@_handle_errors
def remove_pages(input_path: str, pages: List[int], output_path: str) -> Dict[str, Any]:
    """Remove specified 1-based pages from a PDF."""
    return pdf_tools.remove_pages(input_path, pages, output_path)


@mcp.tool()
@_handle_errors
def redact_text_regex(
    input_path: str,
    output_path: str,
    pattern: str,
    pages: Optional[List[int]] = None,
    case_insensitive: bool = False,
    whole_words: bool = False,
    fill: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Redact text using a regex pattern."""
    return pdf_tools.redact_text_regex(
        input_path=input_path,
        output_path=output_path,
        pattern=pattern,
        pages=pages,
        case_insensitive=case_insensitive,
        whole_words=whole_words,
        fill=fill,
    )


@mcp.tool()
@_handle_errors
def export_to_json(
    pdf_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """Export PDF text and metadata to JSON."""
    return pdf_tools.export_to_json(
        pdf_path=pdf_path,
        output_path=output_path,
        pages=pages,
        engine=engine,
        dpi=dpi,
        language=language,
    )


@mcp.tool()
@_handle_errors
def export_to_markdown(
    pdf_path: str,
    output_path: str,
    pages: Optional[List[int]] = None,
    engine: str = "auto",
    dpi: int = 300,
    language: str = "eng",
) -> Dict[str, Any]:
    """Export PDF text to Markdown."""
    return pdf_tools.export_to_markdown(
        pdf_path=pdf_path,
        output_path=output_path,
        pages=pages,
        engine=engine,
        dpi=dpi,
        language=language,
    )


@mcp.tool()
@_handle_errors
def insert_text(
    input_path: str,
    page: int,
    text: str,
    output_path: str,
    rect: Optional[Sequence[float]] = None,
    text_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert text via a managed FreeText annotation."""
    return pdf_tools.insert_text(input_path, page, text, output_path, rect=rect, text_id=text_id)


@mcp.tool()
@_handle_errors
def edit_text(
    input_path: str,
    output_path: str,
    text_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Edit managed inserted text."""
    return pdf_tools.edit_text(input_path, output_path, text_id, text, pages=pages)


@mcp.tool()
@_handle_errors
def remove_text(
    input_path: str,
    output_path: str,
    text_id: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Remove managed inserted text."""
    return pdf_tools.remove_text(input_path, output_path, text_id, pages=pages)


@mcp.tool()
@_handle_errors
def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Get basic PDF document metadata."""
    return pdf_tools.get_pdf_metadata(pdf_path)


@mcp.tool()
@_handle_errors
def set_pdf_metadata(
    input_path: str,
    output_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    subject: Optional[str] = None,
    keywords: Optional[str] = None,
) -> Dict[str, Any]:
    """Set basic PDF document metadata (title, author, subject, keywords)."""
    return pdf_tools.set_pdf_metadata(
        input_path,
        output_path,
        title=title,
        author=author,
        subject=subject,
        keywords=keywords,
    )


@mcp.tool()
@_handle_errors
def get_full_metadata(pdf_path: str) -> Dict[str, Any]:
    """Get full PDF metadata and document info."""
    return pdf_tools.get_full_metadata(pdf_path)


@mcp.tool()
@_handle_errors
def sanitize_pdf_metadata(
    input_path: str,
    output_path: str,
    remove_custom: bool = True,
    remove_xmp: bool = True,
) -> Dict[str, Any]:
    """Remove metadata keys from a PDF."""
    return pdf_tools.sanitize_pdf_metadata(
        input_path=input_path,
        output_path=output_path,
        remove_custom=remove_custom,
        remove_xmp=remove_xmp,
    )


@mcp.tool()
@_handle_errors
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
    """Add page numbers as FreeText annotations."""
    return pdf_tools.add_page_numbers(
        input_path=input_path,
        output_path=output_path,
        pages=pages,
        start=start,
        position=position,
        width=width,
        height=height,
        margin=margin,
    )


@mcp.tool()
@_handle_errors
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
    """Add Bates numbering as FreeText annotations."""
    return pdf_tools.add_bates_numbering(
        input_path=input_path,
        output_path=output_path,
        prefix=prefix,
        start=start,
        width=width,
        pages=pages,
        position=position,
        margin=margin,
        box_width=box_width,
        box_height=box_height,
    )


@mcp.tool()
@_handle_errors
def verify_digital_signatures(pdf_path: str) -> Dict[str, Any]:
    """Verify digital signatures in a PDF."""
    return pdf_tools.verify_digital_signatures(pdf_path)


@mcp.tool()
@_handle_errors
def add_text_watermark(
    input_path: str,
    output_path: str,
    text: str,
    pages: Optional[List[int]] = None,
    rect: Optional[Sequence[float]] = None,
    annotation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a simple text watermark or stamp via FreeText annotations."""
    return pdf_tools.add_text_watermark(
        input_path,
        output_path,
        text,
        pages=pages,
        rect=rect,
        annotation_id=annotation_id,
    )


@mcp.tool()
@_handle_errors
def add_highlight(
    input_path: str,
    output_path: str,
    page: int,
    text: Optional[str] = None,
    rect: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Add highlight annotations by text search or rectangle."""
    return pdf_tools.add_highlight(
        input_path=input_path,
        output_path=output_path,
        page=page,
        text=text,
        rect=rect,
    )


@mcp.tool()
@_handle_errors
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
    """Add a date stamp as a FreeText annotation."""
    return pdf_tools.add_date_stamp(
        input_path=input_path,
        output_path=output_path,
        pages=pages,
        position=position,
        margin=margin,
        width=width,
        height=height,
        date_text=date_text,
    )


@mcp.tool()
@_handle_errors
def add_comment(
    input_path: str,
    output_path: str,
    page: int,
    text: str,
    pos: Sequence[float],
    comment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a PDF comment (sticky note) using PyMuPDF."""
    return pdf_tools.add_comment(
        input_path=input_path,
        output_path=output_path,
        page=page,
        text=text,
        pos=pos,
        comment_id=comment_id,
    )


@mcp.tool()
@_handle_errors
def update_comment(
    input_path: str,
    output_path: str,
    comment_id: str,
    text: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Update a PDF comment by id using PyMuPDF."""
    return pdf_tools.update_comment(
        input_path=input_path,
        output_path=output_path,
        comment_id=comment_id,
        text=text,
        pages=pages,
    )


@mcp.tool()
@_handle_errors
def remove_comment(
    input_path: str,
    output_path: str,
    comment_id: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Remove a PDF comment by id using PyMuPDF."""
    return pdf_tools.remove_comment(
        input_path=input_path,
        output_path=output_path,
        comment_id=comment_id,
        pages=pages,
    )


@mcp.tool()
@_handle_errors
def add_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    image_path: str,
    rect: Sequence[float],
) -> Dict[str, Any]:
    """Add a signature image by inserting it on a page (PyMuPDF)."""
    return pdf_tools.add_signature_image(
        input_path=input_path,
        output_path=output_path,
        page=page,
        image_path=image_path,
        rect=rect,
    )


@mcp.tool()
@_handle_errors
def update_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    signature_xref: int,
    image_path: Optional[str] = None,
    rect: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Update or resize a signature image (PyMuPDF)."""
    return pdf_tools.update_signature_image(
        input_path=input_path,
        output_path=output_path,
        page=page,
        signature_xref=signature_xref,
        image_path=image_path,
        rect=rect,
    )


@mcp.tool()
@_handle_errors
def remove_signature_image(
    input_path: str,
    output_path: str,
    page: int,
    signature_xref: int,
) -> Dict[str, Any]:
    """Remove a signature image by xref (PyMuPDF)."""
    return pdf_tools.remove_signature_image(
        input_path=input_path,
        output_path=output_path,
        page=page,
        signature_xref=signature_xref,
    )


# =============================================================================
# OCR and Text Extraction Tools
# =============================================================================


@mcp.tool()
@_handle_errors
def detect_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze a PDF to classify its content type.

    Returns:
    - classification: "searchable", "image_based", or "hybrid"
    - needs_ocr: Whether OCR is recommended for full text extraction
    - Detailed page-by-page analysis with text/image metrics
    """
    return pdf_tools.detect_pdf_type(pdf_path)


@mcp.tool()
@_handle_errors
def extract_text_native(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract text from PDF using native text layer only (no OCR).

    Fast extraction for PDFs with embedded text. Use detect_pdf_type first
    to determine if the PDF has sufficient native text.
    """
    return pdf_tools.extract_text_native(pdf_path, pages=pages)


@mcp.tool()
@_handle_errors
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
    - "auto": Try native extraction first; fall back to OCR if insufficient
    - "native": Use only native text extraction (no OCR)
    - "tesseract": Force OCR using Tesseract
    - "force_ocr": Always use OCR even if native text exists

    Requires tesseract-ocr to be installed for OCR functionality.
    """
    return pdf_tools.extract_text_ocr(
        pdf_path,
        pages=pages,
        engine=engine,
        dpi=dpi,
        language=language,
    )


@mcp.tool()
@_handle_errors
def get_pdf_text_blocks(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract text blocks with position information from PDF.

    Returns structured text blocks with bounding boxes, useful for
    understanding document layout and identifying form field locations.
    """
    return pdf_tools.get_pdf_text_blocks(pdf_path, pages=pages)


# =============================================================================
# OCR Phase 2: Enhanced OCR Tools
# =============================================================================


@mcp.tool()
@_handle_errors
def get_ocr_languages() -> Dict[str, Any]:
    """
    Get available OCR languages and Tesseract installation status.

    Returns list of installed language codes and common language mappings.
    """
    return pdf_tools.get_ocr_languages()


@mcp.tool()
@_handle_errors
def extract_text_with_confidence(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    language: str = "eng",
    dpi: int = 300,
    min_confidence: int = 0,
) -> Dict[str, Any]:
    """
    Extract text from PDF with OCR confidence scores.

    Returns word-level confidence scores (0-100), allowing filtering of
    low-confidence text. Useful for quality assessment and post-processing.

    Supports multi-language with '+' separator (e.g., "eng+fra").
    """
    return pdf_tools.extract_text_with_confidence(
        pdf_path,
        pages=pages,
        language=language,
        dpi=dpi,
        min_confidence=min_confidence,
    )


# =============================================================================
# Table Extraction Tools
# =============================================================================


@mcp.tool()
@_handle_errors
def extract_tables(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    output_format: str = "list",
) -> Dict[str, Any]:
    """
    Extract tables from PDF pages.

    Uses table detection to find and extract tabular data.

    Args:
        output_format: "list" for list of lists, "dict" for list of dicts with headers
    """
    return pdf_tools.extract_tables(pdf_path, pages=pages, output_format=output_format)


# =============================================================================
# Image Extraction Tools
# =============================================================================


@mcp.tool()
@_handle_errors
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

    Saves images to output_dir with format: page{N}_img{M}.{ext}
    """
    return pdf_tools.extract_images(
        pdf_path,
        output_dir,
        pages=pages,
        min_width=min_width,
        min_height=min_height,
        image_format=image_format,
    )


@mcp.tool()
@_handle_errors
def get_image_info(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Get information about images in a PDF without extracting them.

    Returns image metadata: dimensions, format, position, etc.
    """
    return pdf_tools.get_image_info(pdf_path, pages=pages)


# =============================================================================
# Smart/Hybrid Text Extraction
# =============================================================================


@mcp.tool()
@_handle_errors
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
    the native text content. Optimal for hybrid documents with mixed pages.

    Args:
        native_threshold: Min chars to prefer native extraction (default: 100)
    """
    return pdf_tools.extract_text_smart(
        pdf_path,
        pages=pages,
        native_threshold=native_threshold,
        ocr_dpi=ocr_dpi,
        language=language,
    )


# =============================================================================
# Form Auto-Detection
# =============================================================================


@mcp.tool()
@_handle_errors
def detect_form_fields(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Detect potential form fields in a PDF using text analysis.

    Analyzes text blocks to find patterns suggesting fillable fields:
    - Labels like "Name:", "Date:", "Address:"
    - Checkbox indicators
    - Underlines for text entry

    Useful for PDFs that appear to be forms but don't have AcroForm fields.
    """
    return pdf_tools.detect_form_fields(pdf_path, pages=pages)


@mcp.tool()
@_handle_errors
def detect_pii_patterns(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Detect common PII patterns (email, phone, SSN, credit card) in a PDF.
    """
    return pdf_tools.detect_pii_patterns(pdf_path, pages=pages)


# =============================================================================
# Phase 3: Link Extraction, Optimization, Barcodes, Splitting, Comparison, Batch
# =============================================================================


@mcp.tool()
@_handle_errors
def extract_links(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Extract links (URLs, hyperlinks, internal references) from a PDF.

    Returns all links with their types (uri, goto, external_goto, etc.)
    and positions on the page.

    Args:
        pdf_path: Path to the PDF file
        pages: Optional list of page numbers (1-indexed). None = all pages.
    """
    return pdf_tools.extract_links(pdf_path, pages=pages)


@mcp.tool()
@_handle_errors
def optimize_pdf(
    pdf_path: str,
    output_path: str,
    quality: str = "medium",
) -> Dict[str, Any]:
    """
    Optimize/compress a PDF to reduce file size.

    Applies various optimization techniques: garbage collection,
    image compression, font subsetting.

    Args:
        pdf_path: Path to the input PDF
        output_path: Path for the optimized PDF
        quality: "low" (max compression), "medium", or "high" (min compression)
    """
    return pdf_tools.optimize_pdf(pdf_path, output_path, quality=quality)


@mcp.tool()
@_handle_errors
def detect_barcodes(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Detect and decode barcodes/QR codes in a PDF.

    Requires pyzbar library. Renders each page and scans for:
    QR codes, Code128, Code39, EAN13, EAN8, UPC-A, etc.

    Args:
        pdf_path: Path to the PDF file
        pages: Optional page numbers (1-indexed). None = all pages.
        dpi: Render resolution (higher = better detection, slower)
    """
    return pdf_tools.detect_barcodes(pdf_path, pages=pages, dpi=dpi)


@mcp.tool()
@_handle_errors
def split_pdf_by_bookmarks(
    pdf_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Split a PDF into multiple files based on its bookmarks (TOC).

    Each bookmark becomes a separate PDF containing pages from that
    section to the next bookmark.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save split PDFs
    """
    return pdf_tools.split_pdf_by_bookmarks(pdf_path, output_dir)


@mcp.tool()
@_handle_errors
def split_pdf_by_pages(
    pdf_path: str,
    output_dir: str,
    pages_per_split: int = 1,
) -> Dict[str, Any]:
    """
    Split a PDF into multiple files by page count.

    Each output file contains the specified number of pages.

    Args:
        pdf_path: Path to the input PDF
        output_dir: Directory to save split PDFs
        pages_per_split: Number of pages per output file (default: 1)
    """
    return pdf_tools.split_pdf_by_pages(pdf_path, output_dir, pages_per_split=pages_per_split)


@mcp.tool()
@_handle_errors
def compare_pdfs(
    pdf1_path: str,
    pdf2_path: str,
    compare_text: bool = True,
    compare_images: bool = False,
) -> Dict[str, Any]:
    """
    Compare two PDFs and identify differences.

    Compares page count, text content, and optionally images.

    Args:
        pdf1_path: Path to the first PDF
        pdf2_path: Path to the second PDF
        compare_text: Whether to compare text content (default: True)
        compare_images: Whether to compare image counts (default: False)
    """
    return pdf_tools.compare_pdfs(
        pdf1_path, pdf2_path,
        compare_text=compare_text,
        compare_images=compare_images,
    )


@mcp.tool()
@_handle_errors
def batch_process(
    pdf_paths: List[str],
    operation: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process multiple PDFs with a single operation.

    Supported operations:
    - "get_info": Get basic PDF info (page count, metadata, size)
    - "extract_text": Extract text from each PDF
    - "extract_links": Extract links from each PDF
    - "optimize": Compress each PDF (requires output_dir)

    Args:
        pdf_paths: List of PDF file paths
        operation: Operation to perform
        output_dir: Required for "optimize" operation
    """
    return pdf_tools.batch_process(pdf_paths, operation, output_dir=output_dir)


if __name__ == "__main__":
    mcp.run(transport="stdio")

