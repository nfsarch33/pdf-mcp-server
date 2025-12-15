from __future__ import annotations

import functools
import traceback
from pathlib import Path
from typing import Any, Dict, List

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


if __name__ == "__main__":
    mcp.run(transport="stdio")

