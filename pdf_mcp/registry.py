"""Single-source registry of pdf-mcp tools (TICKET-05, v1.3.0).

The registry is the sole authority on which MCP tools the server exposes
and which CLI verb groups the Typer entry-point eventually mounts. Both
``pdf_mcp.server`` and ``pdf_mcp.cli`` consume this module so that adding
or re-grouping a tool is a one-line edit instead of a 5-place duplication.

Hard contract
-------------

* Importing ``pdf_mcp.registry`` MUST NOT import ``pdf_mcp.pdf_tools``.
  ``pdf_tools`` pulls in ``pymupdf``, ``pypdf``, optional ``openai``, and
  many other heavy modules; eagerly importing it from registry would
  defeat the lazy-import property that keeps ``pdf-mcp --help`` /
  ``--version`` fast (see ``pdf_mcp/cli.py:serve``).
* Tool names are stable. Removing or renaming an entry is a breaking
  change for MCP clients and must be reviewed.

Quick add a tool
----------------

    from pdf_mcp.registry import register_tool

    register_tool(
        name="my_new_tool",
        verb="pages",
        description="What it does (one sentence).",
        import_path="pdf_mcp.pdf_tools:my_new_tool",
    )

The ``import_path`` form is ``"package.module:attr"``. ``LazyCallable``
resolves it via :func:`importlib.import_module` on first call and
memoises the resolved attribute, so the heavy import only happens when
the tool actually runs.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

JSONSchema = Mapping[str, Any]


class LazyCallable:
    """Resolve ``module:attr`` on first call, memoise, and forward args.

    The path string is split on ``:`` into ``(module, attr)``. The
    underlying function is fetched with :func:`importlib.import_module`
    plus :func:`getattr`. Subsequent calls reuse the cached resolution
    so we never re-import.

    Empty / malformed paths raise ``ValueError`` eagerly so test fixtures
    catch typos at registration time.
    """

    __slots__ = ("_path", "_resolved")

    def __init__(self, import_path: str) -> None:
        if ":" not in import_path:
            raise ValueError(f"LazyCallable import_path must be 'module:attr', got: {import_path!r}")
        module_name, _, attr = import_path.partition(":")
        if not module_name or not attr:
            raise ValueError(f"LazyCallable import_path must be 'module:attr', got: {import_path!r}")
        self._path = import_path
        self._resolved: Optional[Callable[..., Any]] = None

    @property
    def import_path(self) -> str:
        return self._path

    def _resolve(self) -> Callable[..., Any]:
        if self._resolved is None:
            module_name, _, attr = self._path.partition(":")
            module = importlib.import_module(module_name)
            self._resolved = getattr(module, attr)
        return self._resolved

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"LazyCallable({self._path!r})"


@dataclass(frozen=True)
class PdfTool:
    """Metadata for one pdf-mcp tool.

    ``callable`` is a :class:`LazyCallable` so importing the registry
    does not pull in :mod:`pdf_mcp.pdf_tools`. ``schema`` /
    ``output_schema`` are reserved for later tickets that auto-generate
    Typer parameters from the JSON schema.
    """

    name: str
    description: str
    verb: str
    callable: Callable[..., Any]
    schema: Optional[JSONSchema] = None
    output_schema: Optional[JSONSchema] = None


@dataclass(frozen=True)
class VerbGroup:
    """Collection of tools that share a CLI verb (e.g. ``form``)."""

    verb: str
    help: str
    tools: tuple[PdfTool, ...] = field(default_factory=tuple)


_TOOLS: "OrderedDict[str, PdfTool]" = OrderedDict()

_VERB_HELP: dict[str, str] = {
    "form": "PDF form discovery, filling, templates, and flattening.",
    "pages": "Page-level mutation: merge, split, extract, rotate, reorder, insert, remove.",
    "text": "Text annotations, redaction, watermarks, comments, page numbers, Bates stamps.",
    "extract": "Read-only extraction: text blocks, tables, images, links, structured data.",
    "sign": "Digital and visual signatures.",
    "metadata": "Metadata read/write/sanitisation and PDF type/feature detection.",
    "ocr": "OCR helpers and image inspection.",
    "ai": "LLM-backed extraction, auto-fill, and analysis.",
    "batch": "Multi-file processing and comparisons.",
    "security": "Encryption and PII detection.",
}


def register_tool(
    *,
    name: str,
    verb: str,
    description: str,
    import_path: str,
    schema: Optional[JSONSchema] = None,
    output_schema: Optional[JSONSchema] = None,
) -> PdfTool:
    """Register a single tool. Raises ``ValueError`` on duplicate name."""
    if name in _TOOLS:
        raise ValueError(f"Tool name {name!r} already registered (existing verb={_TOOLS[name].verb})")
    if verb not in _VERB_HELP:
        raise ValueError(
            f"Tool {name!r} uses unknown verb {verb!r}; add it to _VERB_HELP first or pick one of {sorted(_VERB_HELP)}"
        )
    tool = PdfTool(
        name=name,
        description=description,
        verb=verb,
        callable=LazyCallable(import_path),
        schema=schema,
        output_schema=output_schema,
    )
    _TOOLS[name] = tool
    return tool


def get(name: str) -> PdfTool:
    """Look up a tool by MCP name. Raises ``KeyError`` if not registered."""
    return _TOOLS[name]


def iter_all() -> Iterable[PdfTool]:
    """Yield every registered tool in insertion order."""
    return tuple(_TOOLS.values())


def verb_groups() -> tuple[VerbGroup, ...]:
    """Group registered tools by verb in insertion order of first-seen verb."""
    seen: "OrderedDict[str, list[PdfTool]]" = OrderedDict()
    for tool in _TOOLS.values():
        seen.setdefault(tool.verb, []).append(tool)
    return tuple(VerbGroup(verb=v, help=_VERB_HELP[v], tools=tuple(tools)) for v, tools in seen.items())


def _seed_default_registry() -> None:
    """Populate the registry with the canonical pdf-mcp tool set.

    The order here MUST match the order of ``@mcp.tool()`` declarations
    in :mod:`pdf_mcp.server` (verified by
    :func:`tests.test_registry.test_registry_mirrors_server_tool_names_in_order`).
    """
    pt = "pdf_mcp.pdf_tools"

    register_tool(
        name="get_pdf_form_fields",
        verb="form",
        description="Read all form fields from a PDF.",
        import_path=f"{pt}:get_pdf_form_fields",
    )
    register_tool(
        name="fill_pdf_form",
        verb="form",
        description="Fill a PDF form with provided values.",
        import_path=f"{pt}:fill_pdf_form",
    )
    register_tool(
        name="fill_pdf_form_any",
        verb="form",
        description="Fill any PDF form, tolerant of arbitrary field shapes.",
        import_path=f"{pt}:fill_pdf_form_any",
    )
    register_tool(
        name="create_pdf_form",
        verb="form",
        description="Create a new PDF form from scratch.",
        import_path=f"{pt}:create_pdf_form",
    )
    register_tool(
        name="get_form_templates",
        verb="form",
        description="List available form templates shipped with pdf-mcp.",
        import_path=f"{pt}:get_form_templates",
    )
    register_tool(
        name="create_pdf_form_from_template",
        verb="form",
        description="Materialise a PDF form from a named template.",
        import_path=f"{pt}:create_pdf_form_from_template",
    )
    register_tool(
        name="flatten_pdf",
        verb="form",
        description="Flatten interactive form fields into static page content.",
        import_path=f"{pt}:flatten_pdf",
    )
    register_tool(
        name="clear_pdf_form_fields",
        verb="form",
        description="Reset all form field values in a PDF.",
        import_path=f"{pt}:clear_pdf_form_fields",
    )

    register_tool(
        name="encrypt_pdf",
        verb="security",
        description="Encrypt a PDF with owner / user password.",
        import_path=f"{pt}:encrypt_pdf",
    )

    register_tool(
        name="merge_pdfs",
        verb="pages",
        description="Merge multiple PDFs into one document.",
        import_path=f"{pt}:merge_pdfs",
    )
    register_tool(
        name="extract_pages",
        verb="pages",
        description="Extract a subset of pages into a new PDF.",
        import_path=f"{pt}:extract_pages",
    )
    register_tool(
        name="rotate_pages",
        verb="pages",
        description="Rotate one or more pages by a multiple of 90 degrees.",
        import_path=f"{pt}:rotate_pages",
    )
    register_tool(
        name="reorder_pages",
        verb="pages",
        description="Reorder pages by an explicit page sequence.",
        import_path=f"{pt}:reorder_pages",
    )

    register_tool(
        name="add_text_annotation",
        verb="text",
        description="Add a free-text annotation at a given location.",
        import_path=f"{pt}:add_text_annotation",
    )
    register_tool(
        name="update_text_annotation",
        verb="text",
        description="Update the text or position of an existing annotation.",
        import_path=f"{pt}:update_text_annotation",
    )
    register_tool(
        name="remove_text_annotation",
        verb="text",
        description="Remove a text annotation by id.",
        import_path=f"{pt}:remove_text_annotation",
    )
    register_tool(
        name="remove_annotations",
        verb="text",
        description="Remove every annotation on the given pages.",
        import_path=f"{pt}:remove_annotations",
    )

    register_tool(
        name="insert_pages",
        verb="pages",
        description="Insert pages from another PDF at a given offset.",
        import_path=f"{pt}:insert_pages",
    )
    register_tool(
        name="remove_pages",
        verb="pages",
        description="Remove pages by index from a PDF.",
        import_path=f"{pt}:remove_pages",
    )

    register_tool(
        name="redact_text_regex",
        verb="text",
        description="Redact text matching a regex across the document.",
        import_path=f"{pt}:redact_text_regex",
    )

    register_tool(
        name="get_pdf_metadata",
        verb="metadata",
        description="Read PDF metadata (title, author, dates, etc.).",
        import_path=f"{pt}:get_pdf_metadata",
    )
    register_tool(
        name="set_pdf_metadata",
        verb="metadata",
        description="Set PDF metadata fields.",
        import_path=f"{pt}:set_pdf_metadata",
    )
    register_tool(
        name="sanitize_pdf_metadata",
        verb="metadata",
        description="Strip identifying / sensitive metadata.",
        import_path=f"{pt}:sanitize_pdf_metadata",
    )

    register_tool(
        name="add_page_numbers",
        verb="text",
        description="Add page numbers to selected pages.",
        import_path=f"{pt}:add_page_numbers",
    )
    register_tool(
        name="add_bates_numbering",
        verb="text",
        description="Apply Bates numbering across pages.",
        import_path=f"{pt}:add_bates_numbering",
    )

    register_tool(
        name="verify_digital_signatures",
        verb="sign",
        description="Verify digital signatures embedded in a PDF.",
        import_path=f"{pt}:verify_digital_signatures",
    )
    register_tool(
        name="sign_pdf",
        verb="sign",
        description="Digitally sign a PDF using a PKCS#12 keystore.",
        import_path=f"{pt}:sign_pdf",
    )
    register_tool(
        name="sign_pdf_pem",
        verb="sign",
        description="Digitally sign a PDF using PEM-encoded key/cert.",
        import_path=f"{pt}:sign_pdf_pem",
    )

    register_tool(
        name="add_text_watermark",
        verb="text",
        description="Add a textual watermark via FreeText annotations.",
        import_path=f"{pt}:add_text_watermark",
    )
    register_tool(
        name="add_highlight",
        verb="text",
        description="Highlight matched text or arbitrary rectangles.",
        import_path=f"{pt}:add_highlight",
    )
    register_tool(
        name="add_date_stamp",
        verb="text",
        description="Add a date stamp annotation.",
        import_path=f"{pt}:add_date_stamp",
    )
    register_tool(
        name="add_comment",
        verb="text",
        description="Add a sticky-note comment annotation.",
        import_path=f"{pt}:add_comment",
    )
    register_tool(
        name="update_comment",
        verb="text",
        description="Update an existing comment annotation by id.",
        import_path=f"{pt}:update_comment",
    )
    register_tool(
        name="remove_comment",
        verb="text",
        description="Remove a comment annotation by id.",
        import_path=f"{pt}:remove_comment",
    )

    register_tool(
        name="add_signature_image",
        verb="sign",
        description="Stamp a signature image onto a page.",
        import_path=f"{pt}:add_signature_image",
    )
    register_tool(
        name="update_signature_image",
        verb="sign",
        description="Resize or move an existing signature image.",
        import_path=f"{pt}:update_signature_image",
    )
    register_tool(
        name="remove_signature_image",
        verb="sign",
        description="Remove a signature image by xref.",
        import_path=f"{pt}:remove_signature_image",
    )

    register_tool(
        name="detect_pdf_type",
        verb="metadata",
        description="Classify a PDF (scanned, born-digital, hybrid, form-heavy).",
        import_path=f"{pt}:detect_pdf_type",
    )

    register_tool(
        name="get_pdf_text_blocks",
        verb="extract",
        description="Extract text blocks with positions.",
        import_path=f"{pt}:get_pdf_text_blocks",
    )

    register_tool(
        name="get_ocr_languages",
        verb="ocr",
        description="Report available OCR languages and Tesseract status.",
        import_path=f"{pt}:get_ocr_languages",
    )

    register_tool(
        name="extract_tables",
        verb="extract",
        description="Extract tabular data from PDF pages.",
        import_path=f"{pt}:extract_tables",
    )
    register_tool(
        name="extract_images",
        verb="extract",
        description="Extract embedded images.",
        import_path=f"{pt}:extract_images",
    )
    register_tool(
        name="get_image_info",
        verb="ocr",
        description="Inspect images without extracting them.",
        import_path=f"{pt}:get_image_info",
    )

    register_tool(
        name="detect_form_fields",
        verb="form",
        description="Find potential form fields by text analysis.",
        import_path=f"{pt}:detect_form_fields",
    )

    register_tool(
        name="detect_pii_patterns",
        verb="security",
        description="Detect common PII patterns (email, phone, SSN, credit card).",
        import_path=f"{pt}:detect_pii_patterns",
    )

    register_tool(
        name="extract_links",
        verb="extract",
        description="Extract URLs, internal references, and hyperlinks.",
        import_path=f"{pt}:extract_links",
    )
    register_tool(
        name="optimize_pdf",
        verb="pages",
        description="Optimise / compress a PDF to reduce file size.",
        import_path=f"{pt}:optimize_pdf",
    )
    register_tool(
        name="detect_barcodes",
        verb="extract",
        description="Detect and decode barcodes / QR codes.",
        import_path=f"{pt}:detect_barcodes",
    )
    register_tool(
        name="compare_pdfs",
        verb="batch",
        description="Compare two PDFs and identify differences.",
        import_path=f"{pt}:compare_pdfs",
    )
    register_tool(
        name="batch_process",
        verb="batch",
        description="Apply a single operation to multiple PDFs.",
        import_path=f"{pt}:batch_process",
    )

    register_tool(
        name="extract_text",
        verb="extract",
        description="Unified text extraction with selectable engine.",
        import_path=f"{pt}:extract_text",
    )
    register_tool(
        name="split_pdf", verb="pages", description="Split a PDF into multiple files.", import_path=f"{pt}:split_pdf"
    )
    register_tool(
        name="export_pdf",
        verb="extract",
        description="Export PDF content into other formats.",
        import_path=f"{pt}:export_pdf",
    )

    register_tool(
        name="get_llm_backend_info",
        verb="ai",
        description="Report which LLM backends are available.",
        import_path=f"{pt}:get_llm_backend_info",
    )
    register_tool(
        name="auto_fill_pdf_form",
        verb="ai",
        description="LLM-backed intelligent form filling.",
        import_path=f"{pt}:auto_fill_pdf_form",
    )
    register_tool(
        name="extract_structured_data",
        verb="ai",
        description="Extract structured data via patterns or LLM.",
        import_path=f"{pt}:extract_structured_data",
    )
    register_tool(
        name="analyze_pdf_content",
        verb="ai",
        description="Analyse content for type, key entities, and summary.",
        import_path=f"{pt}:analyze_pdf_content",
    )


_seed_default_registry()
