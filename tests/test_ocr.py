"""
Comprehensive OCR tests for pdf-mcp-server.

Tests cover:
- PDF type detection (searchable vs image-based vs hybrid)
- Native text extraction
- OCR text extraction with fallback chain
- Text block extraction with positions
- All PDF fixtures in tests/ directory
"""

from pathlib import Path
from typing import Dict, Any

import pytest

from pdf_mcp import pdf_tools
from pdf_mcp.pdf_tools import PdfToolError


# =============================================================================
# Test Fixtures - All PDFs in tests/
# =============================================================================

TESTS_DIR = Path(__file__).parent

# Categorize test PDFs by expected type (based on actual analysis)
# Classifications determined by text_coverage_ratio and image_coverage_ratio
PDF_FIXTURES = {
    # Form PDFs - searchable with native text layer
    "1006.pdf": {"expected_type": "searchable", "has_forms": True},
    
    # PDFs with native text layer (may have OCR layer embedded)
    "TestOCR.pdf": {"expected_type": "searchable", "has_forms": False},  # Has OCR text layer
    "scanned_example_1.pdf": {"expected_type": "searchable", "has_forms": False},  # Has OCR text layer
    
    # Image-based PDFs (no native text layer, need OCR)
    "pdf_sample2.pdf": {"expected_type": "image_based", "has_forms": False},
    "scansmpl.pdf": {"expected_type": "image_based", "has_forms": False},
    "image-based-pdf-sample.pdf": {"expected_type": "image_based", "has_forms": False},
    "non-text-searchable.pdf": {"expected_type": "image_based", "has_forms": False},
    "Sbizhub_C2219080509040.pdf": {"expected_type": "image_based", "has_forms": False},
    "PublicWaterMassMailing.pdf": {"expected_type": "image_based", "has_forms": False},
}


def get_available_fixtures() -> Dict[str, Dict[str, Any]]:
    """Return only fixtures that exist in the tests directory."""
    available = {}
    for name, info in PDF_FIXTURES.items():
        path = TESTS_DIR / name
        if path.exists():
            available[name] = {**info, "path": str(path)}
    return available


# =============================================================================
# detect_pdf_type Tests
# =============================================================================


class TestDetectPdfType:
    """Tests for the detect_pdf_type function."""

    def test_detect_type_returns_required_fields(self):
        """Verify detect_pdf_type returns all required fields."""
        fixtures = get_available_fixtures()
        if not fixtures:
            pytest.skip("No test fixtures available")
        
        # Use first available fixture
        path = list(fixtures.values())[0]["path"]
        result = pdf_tools.detect_pdf_type(path)
        
        # Check required fields
        assert "pdf_path" in result
        assert "classification" in result
        assert result["classification"] in ("searchable", "image_based", "hybrid")
        assert "total_pages" in result
        assert "pages_with_native_text" in result
        assert "pages_with_images" in result
        assert "total_native_chars" in result
        assert "total_images" in result
        assert "needs_ocr" in result
        assert "tesseract_available" in result
        assert "page_details" in result

    def test_detect_type_all_fixtures(self):
        """Test detect_pdf_type on all available fixtures."""
        fixtures = get_available_fixtures()
        results = {}
        
        for name, info in fixtures.items():
            result = pdf_tools.detect_pdf_type(info["path"])
            results[name] = result
            
            # Basic sanity checks
            assert result["total_pages"] >= 1
            assert result["classification"] in ("searchable", "image_based", "hybrid")
            assert isinstance(result["page_details"], list)
            assert len(result["page_details"]) == result["total_pages"]
        
        # Print summary for debugging
        print("\n--- PDF Type Detection Summary ---")
        for name, result in results.items():
            print(f"{name}: {result['classification']}, "
                  f"pages={result['total_pages']}, "
                  f"text_ratio={result['text_coverage_ratio']}, "
                  f"needs_ocr={result['needs_ocr']}")

    @pytest.mark.parametrize("fixture_name,expected_info", [
        (name, info) for name, info in PDF_FIXTURES.items()
    ])
    def test_detect_type_classification(self, fixture_name: str, expected_info: Dict):
        """Test that PDFs are classified as expected."""
        path = TESTS_DIR / fixture_name
        if not path.exists():
            pytest.skip(f"Fixture {fixture_name} not available")
        
        result = pdf_tools.detect_pdf_type(str(path))
        
        # For image-based PDFs, either image_based or hybrid is acceptable
        # since some might have minimal text layers
        expected = expected_info["expected_type"]
        actual = result["classification"]
        
        if expected == "image_based":
            assert actual in ("image_based", "hybrid"), \
                f"{fixture_name}: expected {expected}, got {actual}"
        else:
            # For searchable PDFs, we're more lenient
            assert actual in ("searchable", "hybrid"), \
                f"{fixture_name}: expected searchable/hybrid, got {actual}"

    def test_detect_type_file_not_found(self, tmp_path: Path):
        """Test error handling for missing file."""
        with pytest.raises(PdfToolError) as exc:
            pdf_tools.detect_pdf_type(str(tmp_path / "nonexistent.pdf"))
        assert "not found" in str(exc.value).lower()

    def test_detect_type_1006_has_text(self):
        """Test that 1006.pdf (form PDF) is detected as searchable."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.detect_pdf_type(str(path))
        assert result["classification"] == "searchable"
        assert result["total_native_chars"] > 100
        assert result["needs_ocr"] is False


# =============================================================================
# extract_text_native Tests
# =============================================================================


class TestExtractTextNative:
    """Tests for native text extraction."""

    def test_extract_native_searchable_pdf(self):
        """Test native extraction on a searchable PDF."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.extract_text_native(str(path))
        
        assert result["method"] == "native"
        assert result["total_chars"] > 0
        assert "text" in result
        assert len(result["text"]) > 0
        assert result["pages_extracted"] >= 1

    def test_extract_native_specific_pages(self):
        """Test native extraction of specific pages."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        # Extract only page 1
        result = pdf_tools.extract_text_native(str(path), pages=[1])
        
        assert result["pages_extracted"] == 1
        assert len(result["page_details"]) == 1
        assert result["page_details"][0]["page"] == 1

    def test_extract_native_image_based_pdf(self):
        """Test native extraction on image-based PDF (should return minimal text)."""
        fixtures = get_available_fixtures()
        image_pdfs = [name for name, info in fixtures.items() 
                      if info["expected_type"] == "image_based"]
        
        if not image_pdfs:
            pytest.skip("No image-based PDF fixtures available")
        
        path = fixtures[image_pdfs[0]]["path"]
        result = pdf_tools.extract_text_native(path)
        
        # Image-based PDFs should have minimal native text
        # (but might have some due to PDF metadata or embedded fonts)
        assert result["method"] == "native"
        # The text might be very short or empty
        assert "text" in result

    def test_extract_native_all_fixtures(self):
        """Test native extraction on all available fixtures."""
        fixtures = get_available_fixtures()
        
        print("\n--- Native Text Extraction Summary ---")
        for name, info in fixtures.items():
            result = pdf_tools.extract_text_native(info["path"])
            print(f"{name}: {result['total_chars']} chars from {result['pages_extracted']} pages")
            
            assert "text" in result
            assert result["pages_extracted"] >= 1


# =============================================================================
# extract_text_ocr Tests
# =============================================================================


class TestExtractTextOcr:
    """Tests for OCR text extraction."""

    def test_extract_ocr_auto_engine(self):
        """Test auto engine mode (native with OCR fallback)."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.extract_text_ocr(str(path), engine="auto")
        
        assert result["engine_requested"] == "auto"
        assert result["method_used"] in ("native", "ocr", "hybrid")
        assert "text" in result

    def test_extract_ocr_native_engine(self):
        """Test native-only engine mode (no OCR)."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.extract_text_ocr(str(path), engine="native")
        
        assert result["engine_requested"] == "native"
        assert result["method_used"] == "native"

    def test_extract_ocr_invalid_engine(self):
        """Test error handling for invalid engine."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        with pytest.raises(PdfToolError) as exc:
            pdf_tools.extract_text_ocr(str(path), engine="invalid_engine")
        assert "Invalid engine" in str(exc.value)

    def test_extract_ocr_specific_pages(self):
        """Test OCR extraction of specific pages."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.extract_text_ocr(str(path), pages=[1], engine="native")
        
        assert result["pages_extracted"] == 1
        assert len(result["page_details"]) == 1

    def test_extract_ocr_tesseract_required_but_missing(self):
        """Test error when Tesseract is required but not available."""
        # This test only runs if Tesseract is NOT installed
        if pdf_tools._HAS_TESSERACT:
            pytest.skip("Tesseract is available, skipping unavailable test")
        
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        with pytest.raises(PdfToolError) as exc:
            pdf_tools.extract_text_ocr(str(path), engine="tesseract")
        assert "Tesseract" in str(exc.value) or "not available" in str(exc.value).lower()

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_ocr_tesseract_engine(self):
        """Test Tesseract OCR engine (requires tesseract-ocr installed)."""
        fixtures = get_available_fixtures()
        image_pdfs = [name for name, info in fixtures.items() 
                      if info["expected_type"] == "image_based"]
        
        if not image_pdfs:
            pytest.skip("No image-based PDF fixtures available")
        
        path = fixtures[image_pdfs[0]]["path"]
        result = pdf_tools.extract_text_ocr(path, engine="tesseract", dpi=150)
        
        assert result["engine_requested"] == "tesseract"
        assert result["method_used"] in ("ocr", "hybrid")
        assert result["dpi"] == 150

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_ocr_force_ocr_engine(self):
        """Test force_ocr engine (OCR even on searchable PDFs)."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.extract_text_ocr(str(path), engine="force_ocr", pages=[1])
        
        assert result["engine_requested"] == "force_ocr"
        assert "ocr" in result["method_used"]

    def test_extract_ocr_all_fixtures_auto(self):
        """Test auto OCR extraction on all available fixtures."""
        fixtures = get_available_fixtures()
        
        print("\n--- OCR Text Extraction Summary (auto mode) ---")
        for name, info in fixtures.items():
            result = pdf_tools.extract_text_ocr(info["path"], engine="auto")
            print(f"{name}: method={result['method_used']}, "
                  f"{result['total_chars']} chars from {result['pages_extracted']} pages")
            
            assert "text" in result
            assert result["pages_extracted"] >= 1


# =============================================================================
# get_pdf_text_blocks Tests
# =============================================================================


class TestGetPdfTextBlocks:
    """Tests for text block extraction with positions."""

    def test_get_text_blocks_returns_structure(self):
        """Test that text blocks returns proper structure."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.get_pdf_text_blocks(str(path))
        
        assert "pdf_path" in result
        assert "total_pages" in result
        assert "pages_analyzed" in result
        assert "page_blocks" in result
        assert isinstance(result["page_blocks"], list)

    def test_get_text_blocks_has_bbox(self):
        """Test that text blocks include bounding boxes."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.get_pdf_text_blocks(str(path), pages=[1])
        
        assert len(result["page_blocks"]) == 1
        page_data = result["page_blocks"][0]
        
        assert "page" in page_data
        assert "width" in page_data
        assert "height" in page_data
        assert "blocks" in page_data
        
        # Check that blocks have proper structure
        for block in page_data["blocks"]:
            assert "type" in block
            assert block["type"] in ("text", "image")
            assert "bbox" in block

    def test_get_text_blocks_specific_pages(self):
        """Test text block extraction for specific pages."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        result = pdf_tools.get_pdf_text_blocks(str(path), pages=[1])
        
        assert result["pages_analyzed"] == 1
        assert len(result["page_blocks"]) == 1
        assert result["page_blocks"][0]["page"] == 1

    def test_get_text_blocks_all_fixtures(self):
        """Test text block extraction on all available fixtures."""
        fixtures = get_available_fixtures()
        
        print("\n--- Text Block Extraction Summary ---")
        for name, info in fixtures.items():
            result = pdf_tools.get_pdf_text_blocks(info["path"])
            
            total_blocks = sum(
                len(p["blocks"]) for p in result["page_blocks"]
            )
            text_blocks = sum(
                len([b for b in p["blocks"] if b["type"] == "text"])
                for p in result["page_blocks"]
            )
            image_blocks = sum(
                len([b for b in p["blocks"] if b["type"] == "image"])
                for p in result["page_blocks"]
            )
            
            print(f"{name}: {result['pages_analyzed']} pages, "
                  f"{total_blocks} blocks ({text_blocks} text, {image_blocks} image)")


# =============================================================================
# MCP Layer Integration Tests
# =============================================================================


class TestMcpLayerOcr:
    """Test OCR tools through the MCP layer."""

    def test_mcp_detect_pdf_type(self):
        """Test detect_pdf_type via MCP layer."""
        import asyncio
        from pdf_mcp import server

        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "detect_pdf_type", {"pdf_path": str(path)}
            )
            assert isinstance(meta, dict)
            assert "result" in meta
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert result["classification"] in ("searchable", "image_based", "hybrid")

    def test_mcp_extract_text_native(self):
        """Test extract_text_native via MCP layer."""
        import asyncio
        from pdf_mcp import server

        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "extract_text_native", {"pdf_path": str(path)}
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert result["method"] == "native"
        assert "text" in result

    def test_mcp_extract_text_ocr(self):
        """Test extract_text_ocr via MCP layer."""
        import asyncio
        from pdf_mcp import server

        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "extract_text_ocr",
                {"pdf_path": str(path), "engine": "auto"},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert result["engine_requested"] == "auto"
        assert "text" in result

    def test_mcp_get_pdf_text_blocks(self):
        """Test get_pdf_text_blocks via MCP layer."""
        import asyncio
        from pdf_mcp import server

        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "get_pdf_text_blocks", {"pdf_path": str(path), "pages": [1]}
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "page_blocks" in result
        assert len(result["page_blocks"]) == 1


# =============================================================================
# Regression Tests for Specific PDFs
# =============================================================================


class TestSpecificPdfRegression:
    """Regression tests for specific PDF fixtures."""

    def test_1006_pdf_full_workflow(self):
        """Comprehensive test on 1006.pdf (InDesign form)."""
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        # Detect type
        detect_result = pdf_tools.detect_pdf_type(str(path))
        assert detect_result["classification"] == "searchable"
        assert detect_result["needs_ocr"] is False
        
        # Native extraction
        native_result = pdf_tools.extract_text_native(str(path))
        assert native_result["total_chars"] > 100
        
        # OCR extraction (should use native)
        ocr_result = pdf_tools.extract_text_ocr(str(path), engine="auto")
        assert ocr_result["method_used"] in ("native", "hybrid")
        
        # Text blocks
        blocks_result = pdf_tools.get_pdf_text_blocks(str(path))
        assert len(blocks_result["page_blocks"]) >= 1

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_image_pdf_ocr_extraction(self):
        """Test OCR extraction on image-based PDFs."""
        fixtures = get_available_fixtures()
        image_pdfs = [name for name, info in fixtures.items() 
                      if info["expected_type"] == "image_based"]
        
        if not image_pdfs:
            pytest.skip("No image-based PDF fixtures available")
        
        # Test first available image PDF
        path = fixtures[image_pdfs[0]]["path"]
        
        # Detect type
        detect_result = pdf_tools.detect_pdf_type(path)
        assert detect_result["classification"] in ("image_based", "hybrid")
        
        # OCR extraction should produce text
        ocr_result = pdf_tools.extract_text_ocr(path, engine="tesseract", dpi=200)
        assert ocr_result["method_used"] in ("ocr", "hybrid")
        
        print(f"\n--- OCR Result for {image_pdfs[0]} ---")
        print(f"Characters extracted: {ocr_result['total_chars']}")
        if ocr_result['total_chars'] > 0:
            preview = ocr_result['text'][:200].replace('\n', ' ')
            print(f"Preview: {preview}...")

    def test_scansmpl_pdf(self):
        """Test scansmpl.pdf (scanner sample)."""
        path = TESTS_DIR / "scansmpl.pdf"
        if not path.exists():
            pytest.skip("scansmpl.pdf fixture not available")
        
        detect_result = pdf_tools.detect_pdf_type(str(path))
        print(f"\nscansmpl.pdf: {detect_result['classification']}, "
              f"native_chars={detect_result['total_native_chars']}, "
              f"images={detect_result['total_images']}")

    def test_testocr_pdf(self):
        """Test TestOCR.pdf (OCR test document)."""
        path = TESTS_DIR / "TestOCR.pdf"
        if not path.exists():
            pytest.skip("TestOCR.pdf fixture not available")
        
        detect_result = pdf_tools.detect_pdf_type(str(path))
        print(f"\nTestOCR.pdf: {detect_result['classification']}, "
              f"native_chars={detect_result['total_native_chars']}, "
              f"images={detect_result['total_images']}")

    def test_public_water_mass_mailing(self):
        """Test PublicWaterMassMailing.pdf (large multi-page document)."""
        path = TESTS_DIR / "PublicWaterMassMailing.pdf"
        if not path.exists():
            pytest.skip("PublicWaterMassMailing.pdf fixture not available")
        
        detect_result = pdf_tools.detect_pdf_type(str(path))
        print(f"\nPublicWaterMassMailing.pdf: {detect_result['classification']}, "
              f"pages={detect_result['total_pages']}, "
              f"native_chars={detect_result['total_native_chars']}")
        
        # Test extraction of first few pages only (for speed)
        native_result = pdf_tools.extract_text_native(str(path), pages=[1, 2])
        assert native_result["pages_extracted"] == 2


# =============================================================================
# Performance Tests
# =============================================================================


class TestOcrPerformance:
    """Performance-related tests for OCR operations."""

    def test_detect_type_performance(self):
        """Ensure detect_pdf_type completes in reasonable time."""
        import time
        
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        start = time.time()
        pdf_tools.detect_pdf_type(str(path))
        elapsed = time.time() - start
        
        # Should complete in under 2 seconds for a typical PDF
        assert elapsed < 2.0, f"detect_pdf_type took {elapsed:.2f}s"

    def test_native_extraction_performance(self):
        """Ensure native extraction completes quickly."""
        import time
        
        path = TESTS_DIR / "1006.pdf"
        if not path.exists():
            pytest.skip("1006.pdf fixture not available")
        
        start = time.time()
        pdf_tools.extract_text_native(str(path))
        elapsed = time.time() - start
        
        # Native extraction should be very fast
        assert elapsed < 1.0, f"extract_text_native took {elapsed:.2f}s"

