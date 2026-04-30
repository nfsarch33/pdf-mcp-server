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

    def resolve(self) -> Callable[..., Any]:
        """Return the underlying callable, importing on first access.

        The first call performs the (potentially heavy) import of the
        target module via :func:`importlib.import_module` and memoises
        the resolved attribute. Subsequent calls return the cached
        reference and do not re-import.

        Consumers that need the *function itself* (for example, to feed
        FastMCP's :func:`mcp.server.fastmcp.FastMCP.tool` decorator,
        which introspects ``__name__``, ``__doc__`` and
        ``__annotations__``) should call :meth:`resolve` rather than
        invoking the :class:`LazyCallable` instance.
        """
        if self._resolved is None:
            module_name, _, attr = self._path.partition(":")
            module = importlib.import_module(module_name)
            self._resolved = getattr(module, attr)
        return self._resolved

    # Private alias retained for backwards compatibility within the
    # package; new code should call :meth:`resolve`.
    _resolve = resolve

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve()(*args, **kwargs)

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
        description="Return available form fields in the PDF.",
        import_path=f"{pt}:get_pdf_form_fields",
    )
    register_tool(
        name="fill_pdf_form",
        verb="form",
        description="Fill a PDF form with provided data. Optionally flatten to make non-editable.",
        import_path=f"{pt}:fill_pdf_form",
    )
    register_tool(
        name="fill_pdf_form_any",
        verb="form",
        description="Fill standard or non-standard forms using label detection when needed.",
        import_path=f"{pt}:fill_pdf_form_any",
    )
    register_tool(
        name="create_pdf_form",
        verb="form",
        description="Create a new PDF with AcroForm fields.",
        import_path=f"{pt}:create_pdf_form",
    )
    register_tool(
        name="get_form_templates",
        verb="form",
        description="List built-in form templates for common workflows.",
        import_path=f"{pt}:get_form_templates",
    )
    register_tool(
        name="create_pdf_form_from_template",
        verb="form",
        description="Create a PDF form using a built-in template.",
        import_path=f"{pt}:create_pdf_form_from_template",
    )
    register_tool(
        name="flatten_pdf",
        verb="form",
        description="Flatten a PDF (remove form fields/annotations).",
        import_path=f"{pt}:flatten_pdf",
    )
    register_tool(
        name="clear_pdf_form_fields",
        verb="form",
        description="Clear (delete) values for PDF form fields while keeping fields fillable.",
        import_path=f"{pt}:clear_pdf_form_fields",
    )

    register_tool(
        name="encrypt_pdf",
        verb="security",
        description="Encrypt (password-protect) a PDF using pypdf.",
        import_path=f"{pt}:encrypt_pdf",
    )

    register_tool(
        name="merge_pdfs",
        verb="pages",
        description="Merge multiple PDFs into a single file.",
        import_path=f"{pt}:merge_pdfs",
    )
    register_tool(
        name="extract_pages",
        verb="pages",
        description="Extract specific 1-based pages into a new PDF.",
        import_path=f"{pt}:extract_pages",
    )
    register_tool(
        name="rotate_pages",
        verb="pages",
        description="Rotate specified 1-based pages by degrees (must be multiple of 90).",
        import_path=f"{pt}:rotate_pages",
    )
    register_tool(
        name="reorder_pages",
        verb="pages",
        description="Reorder pages in a PDF using a 1-based page list.",
        import_path=f"{pt}:reorder_pages",
    )

    register_tool(
        name="add_text_annotation",
        verb="text",
        description="Add a FreeText annotation to a page (managed text insertion).",
        import_path=f"{pt}:add_text_annotation",
    )
    register_tool(
        name="update_text_annotation",
        verb="text",
        description="Update an existing annotation by annotation_id.",
        import_path=f"{pt}:update_text_annotation",
    )
    register_tool(
        name="remove_text_annotation",
        verb="text",
        description="Remove an existing annotation by annotation_id.",
        import_path=f"{pt}:remove_text_annotation",
    )
    register_tool(
        name="remove_annotations",
        verb="text",
        description="Remove annotations from given pages. Optionally filter by subtype (e.g., FreeText).",
        import_path=f"{pt}:remove_annotations",
    )

    register_tool(
        name="insert_pages",
        verb="pages",
        description="Insert pages from another PDF before at_page (1-based).",
        import_path=f"{pt}:insert_pages",
    )
    register_tool(
        name="remove_pages",
        verb="pages",
        description="Remove specified 1-based pages from a PDF.",
        import_path=f"{pt}:remove_pages",
    )

    register_tool(
        name="redact_text_regex",
        verb="text",
        description="Redact text using a regex pattern.",
        import_path=f"{pt}:redact_text_regex",
    )

    register_tool(
        name="get_pdf_metadata",
        verb="metadata",
        description="Get PDF document metadata.",
        import_path=f"{pt}:get_pdf_metadata",
    )
    register_tool(
        name="set_pdf_metadata",
        verb="metadata",
        description="Set basic PDF document metadata (title, author, subject, keywords).",
        import_path=f"{pt}:set_pdf_metadata",
    )
    register_tool(
        name="sanitize_pdf_metadata",
        verb="metadata",
        description="Remove metadata keys from a PDF.",
        import_path=f"{pt}:sanitize_pdf_metadata",
    )

    register_tool(
        name="add_page_numbers",
        verb="text",
        description="Add page numbers as FreeText annotations.",
        import_path=f"{pt}:add_page_numbers",
    )
    register_tool(
        name="add_bates_numbering",
        verb="text",
        description="Add Bates numbering as FreeText annotations.",
        import_path=f"{pt}:add_bates_numbering",
    )

    register_tool(
        name="verify_digital_signatures",
        verb="sign",
        description="Verify digital signatures in a PDF.",
        import_path=f"{pt}:verify_digital_signatures",
    )
    register_tool(
        name="sign_pdf",
        verb="sign",
        description="Digitally sign a PDF using a PKCS#12/PFX certificate.",
        import_path=f"{pt}:sign_pdf",
    )
    register_tool(
        name="sign_pdf_pem",
        verb="sign",
        description="Digitally sign a PDF using PEM key + cert chain.",
        import_path=f"{pt}:sign_pdf_pem",
    )

    register_tool(
        name="add_text_watermark",
        verb="text",
        description="Add a simple text watermark or stamp via FreeText annotations.",
        import_path=f"{pt}:add_text_watermark",
    )
    register_tool(
        name="add_highlight",
        verb="text",
        description="Add highlight annotations by text search or rectangle.",
        import_path=f"{pt}:add_highlight",
    )
    register_tool(
        name="add_date_stamp",
        verb="text",
        description="Add a date stamp as a FreeText annotation.",
        import_path=f"{pt}:add_date_stamp",
    )
    register_tool(
        name="add_comment",
        verb="text",
        description="Add a PDF comment (sticky note) using PyMuPDF.",
        import_path=f"{pt}:add_comment",
    )
    register_tool(
        name="update_comment",
        verb="text",
        description="Update a PDF comment by id using PyMuPDF.",
        import_path=f"{pt}:update_comment",
    )
    register_tool(
        name="remove_comment",
        verb="text",
        description="Remove a PDF comment by id using PyMuPDF.",
        import_path=f"{pt}:remove_comment",
    )

    register_tool(
        name="add_signature_image",
        verb="sign",
        description="Add a signature image by inserting it on a page (PyMuPDF).",
        import_path=f"{pt}:add_signature_image",
    )
    register_tool(
        name="update_signature_image",
        verb="sign",
        description="Update or resize a signature image (PyMuPDF).",
        import_path=f"{pt}:update_signature_image",
    )
    register_tool(
        name="remove_signature_image",
        verb="sign",
        description="Remove a signature image by xref (PyMuPDF).",
        import_path=f"{pt}:remove_signature_image",
    )

    register_tool(
        name="detect_pdf_type",
        verb="metadata",
        description="Analyze a PDF to classify its content type.",
        import_path=f"{pt}:detect_pdf_type",
    )

    register_tool(
        name="get_pdf_text_blocks",
        verb="extract",
        description="Extract text blocks with position information from PDF.",
        import_path=f"{pt}:get_pdf_text_blocks",
    )

    register_tool(
        name="get_ocr_languages",
        verb="ocr",
        description="Get available OCR languages and Tesseract installation status.",
        import_path=f"{pt}:get_ocr_languages",
    )

    register_tool(
        name="extract_tables",
        verb="extract",
        description="Extract tables from PDF pages.",
        import_path=f"{pt}:extract_tables",
    )
    register_tool(
        name="extract_images",
        verb="extract",
        description="Extract embedded images from PDF pages.",
        import_path=f"{pt}:extract_images",
    )

    register_tool(
        name="get_image_info",
        verb="ocr",
        description="Get information about images in a PDF without extracting them.",
        import_path=f"{pt}:get_image_info",
    )

    register_tool(
        name="detect_form_fields",
        verb="form",
        description="Detect potential form fields in a PDF using text analysis.",
        import_path=f"{pt}:detect_form_fields",
    )

    register_tool(
        name="detect_pii_patterns",
        verb="security",
        description="Detect common PII patterns (email, phone, SSN, credit card) in a PDF.",
        import_path=f"{pt}:detect_pii_patterns",
    )

    register_tool(
        name="extract_links",
        verb="extract",
        description="Extract links (URLs, hyperlinks, internal references) from a PDF.",
        import_path=f"{pt}:extract_links",
    )

    register_tool(
        name="optimize_pdf",
        verb="pages",
        description="Optimize/compress a PDF to reduce file size.",
        import_path=f"{pt}:optimize_pdf",
    )

    register_tool(
        name="detect_barcodes",
        verb="extract",
        description="Detect and decode barcodes/QR codes in a PDF.",
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
        description="Process multiple PDFs with a single operation.",
        import_path=f"{pt}:batch_process",
    )

    register_tool(
        name="extract_text",
        verb="extract",
        description="Unified text extraction with multiple engine options and optional confidence scores.",
        import_path=f"{pt}:extract_text",
    )

    register_tool(
        name="split_pdf",
        verb="pages",
        description="Split a PDF into multiple files.",
        import_path=f"{pt}:split_pdf",
    )

    register_tool(
        name="export_pdf",
        verb="extract",
        description="Export PDF content to different formats.",
        import_path=f"{pt}:export_pdf",
    )

    register_tool(
        name="get_llm_backend_info",
        verb="ai",
        description="Get information about available LLM backends.",
        import_path=f"{pt}:get_llm_backend_info",
    )
    register_tool(
        name="auto_fill_pdf_form",
        verb="ai",
        description="Intelligently fill PDF form fields using LLM-powered field mapping.",
        import_path=f"{pt}:auto_fill_pdf_form",
    )
    register_tool(
        name="extract_structured_data",
        verb="ai",
        description="Extract structured data from PDF using pattern matching or LLM.",
        import_path=f"{pt}:extract_structured_data",
    )
    register_tool(
        name="analyze_pdf_content",
        verb="ai",
        description="Analyze PDF content for document type, key entities, and summary.",
        import_path=f"{pt}:analyze_pdf_content",
    )


_seed_default_registry()
