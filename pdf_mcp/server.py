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
def flatten_pdf(input_path: str, output_path: str) -> Dict[str, Any]:
    """Flatten a PDF (remove form fields/annotations)."""
    return pdf_tools.flatten_pdf(input_path, output_path)


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


if __name__ == "__main__":
    mcp.run(transport="stdio")

