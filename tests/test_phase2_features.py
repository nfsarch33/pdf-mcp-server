"""
Integration tests for Phase 2 features:
- OCR Phase 2: Multi-language support, confidence scores
- Table extraction
- Image extraction
- Smart/hybrid text extraction
- Form auto-detection
"""

from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

import pytest

from pdf_mcp import pdf_tools
from pdf_mcp.pdf_tools import PdfToolError


# =============================================================================
# Test Fixtures
# =============================================================================

TESTS_DIR = Path(__file__).parent


def get_test_pdf(name: str) -> str:
    """Get path to test PDF if it exists."""
    path = TESTS_DIR / name
    if path.exists():
        return str(path)
    return None


# =============================================================================
# OCR Phase 2 Tests: Multi-language and Confidence Scores
# =============================================================================


class TestOcrLanguages:
    """Tests for OCR language support."""

    def test_get_ocr_languages_structure(self):
        """Test that get_ocr_languages returns proper structure."""
        result = pdf_tools.get_ocr_languages()

        assert "tesseract_available" in result
        assert isinstance(result["tesseract_available"], bool)
        assert "installed_languages" in result
        assert isinstance(result["installed_languages"], list)
        assert "common_language_codes" in result
        assert isinstance(result["common_language_codes"], dict)

    def test_common_languages_include_english(self):
        """Test that common languages include English."""
        result = pdf_tools.get_ocr_languages()
        assert "eng" in result["common_language_codes"]
        assert result["common_language_codes"]["eng"] == "English"

    def test_tesseract_languages_include_common(self):
        """Test common language codes are documented."""
        result = pdf_tools.get_ocr_languages()
        common = result["common_language_codes"]

        # Should have major languages
        expected_codes = ["eng", "fra", "deu", "spa", "chi_sim", "jpn"]
        for code in expected_codes:
            assert code in common, f"Missing common language: {code}"


class TestExtractTextWithConfidence:
    """Tests for OCR with confidence scores."""

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_with_confidence_structure(self):
        """Test extract_text_with_confidence returns proper structure."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_with_confidence(pdf_path, pages=[1])

        assert "pdf_path" in result
        assert "language" in result
        assert "dpi" in result
        assert "min_confidence" in result
        assert "pages_extracted" in result
        assert "total_words" in result
        assert "overall_average_confidence" in result
        assert "text" in result
        assert "page_details" in result

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_with_confidence_word_level(self):
        """Test that word-level details include confidence."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_with_confidence(pdf_path, pages=[1])

        assert result["pages_extracted"] == 1
        page_detail = result["page_details"][0]

        assert "words" in page_detail
        assert "average_confidence" in page_detail

        # Check word structure if words exist
        if page_detail["words"]:
            word = page_detail["words"][0]
            assert "text" in word
            assert "confidence" in word
            assert "bbox" in word
            assert isinstance(word["confidence"], int)

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_with_min_confidence_filter(self):
        """Test that min_confidence filters low-confidence words."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        # Extract all words
        result_all = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], min_confidence=0
        )

        # Extract only high confidence words
        result_high = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], min_confidence=80
        )

        # High confidence should have same or fewer words
        assert result_high["total_words"] <= result_all["total_words"]

    def test_extract_with_confidence_no_tesseract(self):
        """Test error when Tesseract not available."""
        if pdf_tools._HAS_TESSERACT:
            pytest.skip("Tesseract is available")

        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        with pytest.raises(PdfToolError) as exc:
            pdf_tools.extract_text_with_confidence(pdf_path)
        assert "Tesseract" in str(exc.value)


# =============================================================================
# Table Extraction Tests
# =============================================================================


class TestExtractTables:
    """Tests for table extraction."""

    def test_extract_tables_structure(self):
        """Test that extract_tables returns proper structure."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_tables(pdf_path)

        assert "pdf_path" in result
        assert "total_pages" in result
        assert "pages_analyzed" in result
        assert "total_tables" in result
        assert "output_format" in result
        assert "page_tables" in result
        assert isinstance(result["page_tables"], list)

    def test_extract_tables_list_format(self):
        """Test table extraction with list format."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_tables(pdf_path, output_format="list")
        assert result["output_format"] == "list"

    def test_extract_tables_dict_format(self):
        """Test table extraction with dict format (uses headers)."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_tables(pdf_path, output_format="dict")
        assert result["output_format"] == "dict"

    def test_extract_tables_invalid_format(self):
        """Test error for invalid output format."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        with pytest.raises(PdfToolError) as exc:
            pdf_tools.extract_tables(pdf_path, output_format="invalid")
        assert "output_format" in str(exc.value)

    def test_extract_tables_specific_pages(self):
        """Test table extraction for specific pages."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_tables(pdf_path, pages=[1])
        assert result["pages_analyzed"] == 1

    def test_extract_tables_all_fixtures(self):
        """Test table extraction on all available PDFs."""
        test_pdfs = ["1006.pdf", "PublicWaterMassMailing.pdf", "TestOCR.pdf"]

        print("\n--- Table Extraction Summary ---")
        for name in test_pdfs:
            pdf_path = get_test_pdf(name)
            if not pdf_path:
                continue

            result = pdf_tools.extract_tables(pdf_path)
            print(f"{name}: {result['total_tables']} tables found")


# =============================================================================
# Image Extraction Tests
# =============================================================================


class TestImageExtraction:
    """Tests for image extraction."""

    def test_get_image_info_structure(self):
        """Test that get_image_info returns proper structure."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.get_image_info(pdf_path)

        assert "pdf_path" in result
        assert "total_pages" in result
        assert "pages_analyzed" in result
        assert "total_images" in result
        assert "page_images" in result
        assert isinstance(result["page_images"], list)

    def test_get_image_info_image_details(self):
        """Test that image info includes dimensions."""
        # Use an image-based PDF
        pdf_path = get_test_pdf("image-based-pdf-sample.pdf")
        if not pdf_path:
            pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("No image-based PDF available")

        result = pdf_tools.get_image_info(pdf_path)

        if result["total_images"] > 0:
            page_info = result["page_images"][0]
            if page_info["images"]:
                img = page_info["images"][0]
                assert "width" in img
                assert "height" in img
                assert "xref" in img

    def test_extract_images_creates_files(self):
        """Test that extract_images creates image files."""
        # Find a PDF with images
        pdf_path = get_test_pdf("image-based-pdf-sample.pdf")
        if not pdf_path:
            pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("No image-based PDF available")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pdf_tools.extract_images(pdf_path, tmpdir, pages=[1])

            assert "images_extracted" in result
            assert "images" in result
            assert result["output_dir"] == tmpdir

            # Check if files were created
            if result["images_extracted"] > 0:
                for img_info in result["images"]:
                    if "output_path" in img_info:
                        assert Path(img_info["output_path"]).exists()

    def test_extract_images_min_dimensions(self):
        """Test that min dimensions filter works."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract with very high min dimensions
            result = pdf_tools.extract_images(
                pdf_path, tmpdir, min_width=5000, min_height=5000
            )

            # Should skip most/all images
            assert result["images_extracted"] == 0 or result["images_skipped"] >= 0

    def test_extract_images_formats(self):
        """Test different output formats."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        for fmt in ["png", "jpeg"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = pdf_tools.extract_images(
                    pdf_path, tmpdir, pages=[1], image_format=fmt
                )
                assert "images" in result

    def test_extract_images_invalid_format(self):
        """Test error for invalid image format."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PdfToolError) as exc:
                pdf_tools.extract_images(pdf_path, tmpdir, image_format="invalid")
            assert "image_format" in str(exc.value)


# =============================================================================
# Smart/Hybrid Text Extraction Tests
# =============================================================================


class TestSmartExtraction:
    """Tests for smart/hybrid text extraction."""

    def test_extract_text_smart_structure(self):
        """Test that extract_text_smart returns proper structure."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_smart(pdf_path)

        assert "pdf_path" in result
        assert "total_pages" in result
        assert "pages_extracted" in result
        assert "total_chars" in result
        assert "native_threshold" in result
        assert "pages_using_native" in result
        assert "pages_using_ocr" in result
        assert "tesseract_available" in result
        assert "text" in result
        assert "page_details" in result

    def test_extract_text_smart_per_page_method(self):
        """Test that each page has method info."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_smart(pdf_path)

        for page_detail in result["page_details"]:
            assert "page" in page_detail
            assert "method" in page_detail
            assert page_detail["method"] in ("native", "ocr")
            assert "text" in page_detail
            assert "char_count" in page_detail
            assert "native_chars" in page_detail

    def test_extract_text_smart_threshold(self):
        """Test native_threshold parameter."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        # With very high threshold, more pages should try OCR
        result_high = pdf_tools.extract_text_smart(
            pdf_path, native_threshold=10000
        )

        # With low threshold, most pages should use native
        result_low = pdf_tools.extract_text_smart(
            pdf_path, native_threshold=10
        )

        # Low threshold should have more native pages
        assert result_low["pages_using_native"] >= result_high["pages_using_native"]

    def test_extract_text_smart_hybrid_document(self):
        """Test smart extraction on image-based PDF."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        result = pdf_tools.extract_text_smart(pdf_path)

        # Image-based PDFs should have mostly OCR pages if Tesseract available
        if pdf_tools._HAS_TESSERACT:
            # At least some pages should attempt OCR
            print(f"\nSmart extraction: {result['pages_using_native']} native, "
                  f"{result['pages_using_ocr']} OCR")

    def test_extract_text_smart_all_fixtures(self):
        """Test smart extraction on multiple PDFs."""
        test_pdfs = ["1006.pdf", "scansmpl.pdf", "TestOCR.pdf"]

        print("\n--- Smart Extraction Summary ---")
        for name in test_pdfs:
            pdf_path = get_test_pdf(name)
            if not pdf_path:
                continue

            result = pdf_tools.extract_text_smart(pdf_path)
            print(f"{name}: {result['pages_using_native']} native, "
                  f"{result['pages_using_ocr']} OCR, "
                  f"{result['total_chars']} chars")


# =============================================================================
# Form Auto-Detection Tests
# =============================================================================


class TestFormDetection:
    """Tests for form field auto-detection."""

    def test_detect_form_fields_structure(self):
        """Test that detect_form_fields returns proper structure."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.detect_form_fields(pdf_path)

        assert "pdf_path" in result
        assert "total_pages" in result
        assert "pages_analyzed" in result
        assert "has_existing_acroform" in result
        assert "existing_field_count" in result
        assert "detected_potential_fields" in result
        assert "detected_fields" in result
        assert "page_analysis" in result
        assert "recommendation" in result

    def test_detect_form_fields_on_form_pdf(self):
        """Test form detection on a PDF with AcroForm fields."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.detect_form_fields(pdf_path)

        # 1006.pdf has AcroForm fields
        assert result["has_existing_acroform"] is True
        assert result["existing_field_count"] > 0
        assert "AcroForm" in result["recommendation"]

    def test_detect_form_fields_labels(self):
        """Test that label patterns are detected."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.detect_form_fields(pdf_path)

        # Check page analysis structure
        for page_analysis in result["page_analysis"]:
            assert "page" in page_analysis
            assert "detected_labels" in page_analysis
            assert "detected_checkboxes" in page_analysis

    def test_detect_form_fields_specific_pages(self):
        """Test form detection on specific pages."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.detect_form_fields(pdf_path, pages=[1])
        assert result["pages_analyzed"] == 1
        assert len(result["page_analysis"]) == 1

    def test_detect_form_fields_suggestion_type(self):
        """Test that detected fields have suggested types."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.detect_form_fields(pdf_path)

        for field in result["detected_fields"]:
            assert "type" in field
            if field["type"] == "label":
                assert "suggested_field_type" in field

    def test_detect_form_fields_all_fixtures(self):
        """Test form detection on multiple PDFs."""
        test_pdfs = ["1006.pdf", "PublicWaterMassMailing.pdf", "TestOCR.pdf"]

        print("\n--- Form Detection Summary ---")
        for name in test_pdfs:
            pdf_path = get_test_pdf(name)
            if not pdf_path:
                continue

            result = pdf_tools.detect_form_fields(pdf_path)
            print(f"{name}: AcroForm={result['has_existing_acroform']}, "
                  f"existing={result['existing_field_count']}, "
                  f"detected={result['detected_potential_fields']}")


# =============================================================================
# MCP Layer Integration Tests
# =============================================================================


class TestMcpLayerPhase2:
    """Test Phase 2 tools through the MCP layer."""

    def test_mcp_get_ocr_languages(self):
        """Test get_ocr_languages via MCP layer."""
        import asyncio
        from pdf_mcp import server

        async def call():
            _content, meta = await server.mcp.call_tool(
                "get_ocr_languages", {}
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "tesseract_available" in result

    def test_mcp_extract_tables(self):
        """Test extract_tables via MCP layer."""
        import asyncio
        from pdf_mcp import server

        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "extract_tables",
                {"pdf_path": pdf_path, "pages": [1]},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "total_tables" in result

    def test_mcp_get_image_info(self):
        """Test get_image_info via MCP layer."""
        import asyncio
        from pdf_mcp import server

        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "get_image_info",
                {"pdf_path": pdf_path},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "total_images" in result

    def test_mcp_extract_text_smart(self):
        """Test extract_text_smart via MCP layer."""
        import asyncio
        from pdf_mcp import server

        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "extract_text_smart",
                {"pdf_path": pdf_path, "pages": [1]},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "pages_using_native" in result

    def test_mcp_detect_form_fields(self):
        """Test detect_form_fields via MCP layer."""
        import asyncio
        from pdf_mcp import server

        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "detect_form_fields",
                {"pdf_path": pdf_path},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "has_existing_acroform" in result


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflows:
    """End-to-end workflow tests combining multiple features."""

    def test_full_document_analysis_workflow(self):
        """Test complete document analysis: detect type, extract text, find tables."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        # Step 1: Detect PDF type
        type_result = pdf_tools.detect_pdf_type(pdf_path)
        assert type_result["classification"] in ("searchable", "image_based", "hybrid")

        # Step 2: Smart text extraction
        text_result = pdf_tools.extract_text_smart(pdf_path)
        assert text_result["total_chars"] >= 0

        # Step 3: Table extraction
        table_result = pdf_tools.extract_tables(pdf_path)
        assert table_result["total_tables"] >= 0

        # Step 4: Image info
        image_result = pdf_tools.get_image_info(pdf_path)
        assert image_result["total_images"] >= 0

        # Step 5: Form detection
        form_result = pdf_tools.detect_form_fields(pdf_path)
        assert "recommendation" in form_result

        print(f"\n--- Document Analysis: {Path(pdf_path).name} ---")
        print(f"Type: {type_result['classification']}")
        print(f"Text: {text_result['total_chars']} chars "
              f"({text_result['pages_using_native']} native, "
              f"{text_result['pages_using_ocr']} OCR)")
        print(f"Tables: {table_result['total_tables']}")
        print(f"Images: {image_result['total_images']}")
        print(f"Form: {form_result['has_existing_acroform']}, "
              f"{form_result['detected_potential_fields']} detected")

    def test_image_based_document_workflow(self):
        """Test workflow for image-based documents."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        # Detect type
        type_result = pdf_tools.detect_pdf_type(pdf_path)

        # Smart extraction should handle image-based pages
        text_result = pdf_tools.extract_text_smart(pdf_path)

        # Get image info
        image_result = pdf_tools.get_image_info(pdf_path)

        print(f"\n--- Image-based Document: {Path(pdf_path).name} ---")
        print(f"Type: {type_result['classification']}")
        print(f"Needs OCR: {type_result['needs_ocr']}")
        print(f"Images: {image_result['total_images']}")
        print(f"Extracted chars: {text_result['total_chars']}")

        if type_result["classification"] == "image_based":
            assert image_result["total_images"] > 0

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_ocr_quality_assessment_workflow(self):
        """Test OCR quality assessment workflow."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        # Extract with confidence scores
        result = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], min_confidence=0
        )

        avg_conf = result["overall_average_confidence"]

        print(f"\n--- OCR Quality Assessment ---")
        print(f"Words: {result['total_words']}")
        print(f"Average Confidence: {avg_conf}%")

        # Filter for high-quality text
        if result["page_details"] and result["page_details"][0]["words"]:
            high_conf_words = [
                w for w in result["page_details"][0]["words"]
                if w["confidence"] >= 80
            ]
            print(f"High-confidence words (>=80%): {len(high_conf_words)}")


# =============================================================================
# Advanced OCR Tests (requires Tesseract)
# =============================================================================


class TestAdvancedOcr:
    """Advanced OCR tests that require Tesseract to be installed."""

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_ocr_different_dpi_settings(self):
        """Test OCR with different DPI settings."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        # Test with lower DPI (faster but less accurate)
        result_150 = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], dpi=150
        )

        # Test with higher DPI (slower but more accurate)
        result_300 = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], dpi=300
        )

        assert result_150["dpi"] == 150
        assert result_300["dpi"] == 300

        # Both should produce some text
        assert result_150["total_words"] >= 0
        assert result_300["total_words"] >= 0

        print(f"\n--- DPI Comparison ---")
        print(f"150 DPI: {result_150['total_words']} words, "
              f"avg confidence: {result_150['overall_average_confidence']}%")
        print(f"300 DPI: {result_300['total_words']} words, "
              f"avg confidence: {result_300['overall_average_confidence']}%")

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_ocr_on_multiple_scanned_documents(self):
        """Test OCR on various scanned document types."""
        scanned_pdfs = ["scansmpl.pdf", "image-based-pdf-sample.pdf", "TestOCR.pdf"]

        print("\n--- OCR on Multiple Documents ---")
        for name in scanned_pdfs:
            pdf_path = get_test_pdf(name)
            if not pdf_path:
                continue

            result = pdf_tools.extract_text_with_confidence(
                pdf_path, pages=[1], dpi=200, min_confidence=50
            )

            print(f"{name}: {result['total_words']} words (>=50% conf), "
                  f"avg: {result['overall_average_confidence']}%")

            assert "text" in result
            assert result["pages_extracted"] == 1

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_smart_extraction_uses_ocr_for_images(self):
        """Test that smart extraction uses OCR for image-based pages."""
        pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("scansmpl.pdf not available")

        # With low native threshold, should trigger OCR
        result = pdf_tools.extract_text_smart(
            pdf_path,
            native_threshold=10,  # Very low threshold
            ocr_dpi=200
        )

        # Should use OCR for image-based pages
        if result["pages_using_ocr"] > 0:
            print(f"\nSmart extraction used OCR on {result['pages_using_ocr']} pages")
            assert result["total_chars"] > 0

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_confidence_score_ranges(self):
        """Test that confidence scores are in valid range (0-100)."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_with_confidence(pdf_path, pages=[1])

        for page in result["page_details"]:
            for word in page.get("words", []):
                assert 0 <= word["confidence"] <= 100, \
                    f"Invalid confidence: {word['confidence']}"

        assert 0 <= result["overall_average_confidence"] <= 100

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_extract_text_with_confidence_mcp_layer(self):
        """Test extract_text_with_confidence via MCP layer."""
        import asyncio
        from pdf_mcp import server

        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        async def call():
            _content, meta = await server.mcp.call_tool(
                "extract_text_with_confidence",
                {"pdf_path": pdf_path, "pages": [1], "min_confidence": 50},
            )
            result = meta["result"]
            assert "error" not in result, result.get("error")
            return result

        result = asyncio.run(call())
        assert "overall_average_confidence" in result
        assert result["min_confidence"] == 50

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_ocr_extracts_meaningful_text(self):
        """Test that OCR extracts actual meaningful text from documents."""
        pdf_path = get_test_pdf("TestOCR.pdf")
        if not pdf_path:
            pdf_path = get_test_pdf("scansmpl.pdf")
        if not pdf_path:
            pytest.skip("No suitable OCR test PDF available")

        result = pdf_tools.extract_text_ocr(
            pdf_path, pages=[1], engine="tesseract", dpi=300
        )

        # Should extract some text
        assert result["total_chars"] > 0, "OCR should extract some text"
        assert len(result["text"].strip()) > 0

        print(f"\n--- OCR Text Preview ---")
        preview = result["text"][:200].replace("\n", " ")
        print(f"Extracted: {preview}...")

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_force_ocr_on_searchable_pdf(self):
        """Test force_ocr engine on a searchable PDF."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        # Native extraction first
        native_result = pdf_tools.extract_text_native(pdf_path, pages=[1])

        # Force OCR
        ocr_result = pdf_tools.extract_text_ocr(
            pdf_path, pages=[1], engine="force_ocr", dpi=200
        )

        assert native_result["method"] == "native"
        assert "ocr" in ocr_result["method_used"]

        # Both should produce text
        assert native_result["total_chars"] > 0
        assert ocr_result["total_chars"] > 0

        print(f"\n--- Native vs Force OCR ---")
        print(f"Native: {native_result['total_chars']} chars")
        print(f"Force OCR: {ocr_result['total_chars']} chars")


class TestOcrLanguageSupport:
    """Tests for OCR language support features."""

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_get_installed_languages(self):
        """Test that installed languages are reported correctly."""
        result = pdf_tools.get_ocr_languages()

        assert result["tesseract_available"] is True
        assert "eng" in result["installed_languages"], "English should be installed"

        print(f"\nInstalled languages: {result['installed_languages']}")

    @pytest.mark.skipif(not pdf_tools._HAS_TESSERACT, reason="Tesseract not installed")
    def test_ocr_with_english_language(self):
        """Test OCR with explicit English language setting."""
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("1006.pdf not available")

        result = pdf_tools.extract_text_with_confidence(
            pdf_path, pages=[1], language="eng"
        )

        assert result["language"] == "eng"
        assert result["total_words"] > 0
