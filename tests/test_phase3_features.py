"""
Integration tests for Phase 3 features (v0.3.0):
- Link extraction: Extract URLs, hyperlinks, internal references
- PDF optimization: Compress/reduce PDF file size
- Barcode/QR code detection: Detect and decode barcodes and QR codes
- Page splitting: Split PDFs by bookmarks or content markers
- PDF comparison: Diff two PDFs and highlight changes
- Batch processing: Process multiple PDFs in a single call

TDD Pattern: Tests written FIRST, implementation follows.
"""

from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
import os

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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_pdf():
    """Get a sample PDF for testing."""
    return get_test_pdf("1006.pdf")


@pytest.fixture
def multiple_pdfs():
    """Get multiple PDFs for batch testing."""
    pdfs = ["1006.pdf", "pdf_sample2.pdf", "PublicWaterMassMailing.pdf"]
    return [get_test_pdf(p) for p in pdfs if get_test_pdf(p)]


# =============================================================================
# Link Extraction Tests
# =============================================================================


class TestExtractLinks:
    """Tests for link extraction functionality."""

    def test_extract_links_returns_structure(self, sample_pdf):
        """Test that extract_links returns proper structure."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.extract_links(sample_pdf)

        assert "pdf_path" in result
        assert "total_links" in result
        assert "links" in result
        assert isinstance(result["links"], list)
        assert "pages_scanned" in result

    def test_extract_links_specific_pages(self, sample_pdf):
        """Test extracting links from specific pages."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.extract_links(sample_pdf, pages=[1])

        assert result["pages_scanned"] == 1

    def test_extract_links_categories(self, sample_pdf):
        """Test that links are categorized by type."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.extract_links(sample_pdf)

        # Should have link type categories
        assert "link_types" in result
        # Common types: uri, internal, goto
        assert isinstance(result["link_types"], dict)

    def test_extract_links_invalid_pdf(self):
        """Test error handling for invalid PDF."""
        with pytest.raises(PdfToolError):
            pdf_tools.extract_links("/nonexistent/file.pdf")

    def test_extract_links_contains_url_info(self, sample_pdf):
        """Test that URL links contain proper information."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.extract_links(sample_pdf)

        if result["total_links"] > 0:
            link = result["links"][0]
            assert "page" in link
            assert "type" in link
            # URI links should have a uri field
            if link["type"] == "uri":
                assert "uri" in link


# =============================================================================
# PDF Optimization Tests
# =============================================================================


class TestOptimizePdf:
    """Tests for PDF optimization/compression."""

    def test_optimize_pdf_returns_structure(self, sample_pdf, temp_dir):
        """Test that optimize_pdf returns proper structure."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        output_path = os.path.join(temp_dir, "optimized.pdf")
        result = pdf_tools.optimize_pdf(sample_pdf, output_path)

        assert "input_path" in result
        assert "output_path" in result
        assert "original_size" in result
        assert "optimized_size" in result
        assert "compression_ratio" in result
        assert "size_reduction_percent" in result

    def test_optimize_pdf_creates_file(self, sample_pdf, temp_dir):
        """Test that optimize_pdf creates output file."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        output_path = os.path.join(temp_dir, "optimized.pdf")
        result = pdf_tools.optimize_pdf(sample_pdf, output_path)

        assert os.path.exists(output_path)
        assert result["optimized_size"] > 0

    def test_optimize_pdf_quality_settings(self, sample_pdf, temp_dir):
        """Test optimization with different quality settings."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Test with low quality (max compression)
        output_low = os.path.join(temp_dir, "low_quality.pdf")
        result_low = pdf_tools.optimize_pdf(sample_pdf, output_low, quality="low")

        # Test with high quality (min compression)
        output_high = os.path.join(temp_dir, "high_quality.pdf")
        result_high = pdf_tools.optimize_pdf(sample_pdf, output_high, quality="high")

        # Low quality should generally be smaller
        assert result_low["optimized_size"] <= result_high["optimized_size"]

    def test_optimize_pdf_invalid_input(self, temp_dir):
        """Test error handling for invalid input."""
        with pytest.raises(PdfToolError):
            pdf_tools.optimize_pdf("/nonexistent.pdf", os.path.join(temp_dir, "out.pdf"))

    def test_optimize_pdf_preserves_content(self, sample_pdf, temp_dir):
        """Test that optimization preserves page count."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        output_path = os.path.join(temp_dir, "optimized.pdf")
        pdf_tools.optimize_pdf(sample_pdf, output_path)

        # Check page count is preserved
        import pymupdf
        with pymupdf.open(sample_pdf) as orig:
            with pymupdf.open(output_path) as opt:
                assert len(opt) == len(orig)


# =============================================================================
# Barcode/QR Code Detection Tests
# =============================================================================


class TestDetectBarcodes:
    """Tests for barcode/QR code detection."""

    def test_detect_barcodes_returns_structure(self, sample_pdf):
        """Test that detect_barcodes returns proper structure."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.detect_barcodes(sample_pdf)

        assert "pdf_path" in result
        assert "total_barcodes" in result
        assert "barcodes" in result
        assert isinstance(result["barcodes"], list)
        assert "pages_scanned" in result
        assert "pyzbar_available" in result

    def test_detect_barcodes_specific_pages(self, sample_pdf):
        """Test detecting barcodes on specific pages."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.detect_barcodes(sample_pdf, pages=[1])

        assert result["pages_scanned"] == 1

    def test_detect_barcodes_barcode_info(self):
        """Test that detected barcodes contain proper info."""
        # This test uses any PDF - barcodes may or may not be present
        pdf_path = get_test_pdf("1006.pdf")
        if not pdf_path:
            pytest.skip("Test PDF not available")

        result = pdf_tools.detect_barcodes(pdf_path)

        # If barcodes found, verify structure
        if result["total_barcodes"] > 0:
            barcode = result["barcodes"][0]
            assert "page" in barcode
            assert "type" in barcode  # e.g., QRCODE, CODE128, EAN13
            assert "data" in barcode
            assert "position" in barcode

    def test_detect_barcodes_invalid_pdf(self):
        """Test error handling for invalid PDF."""
        with pytest.raises(PdfToolError):
            pdf_tools.detect_barcodes("/nonexistent/file.pdf")

    @pytest.mark.skipif(
        not hasattr(pdf_tools, "_HAS_PYZBAR") or not pdf_tools._HAS_PYZBAR,
        reason="pyzbar not installed"
    )
    def test_detect_barcodes_with_pyzbar(self, sample_pdf):
        """Test barcode detection when pyzbar is available."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.detect_barcodes(sample_pdf)
        assert result["pyzbar_available"] is True


# =============================================================================
# Page Splitting Tests
# =============================================================================


class TestSplitPdfByBookmarks:
    """Tests for splitting PDFs by bookmarks."""

    def test_split_by_bookmarks_returns_structure(self, sample_pdf, temp_dir):
        """Test that split_pdf_by_bookmarks returns proper structure."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.split_pdf_by_bookmarks(sample_pdf, temp_dir)

        assert "input_path" in result
        assert "output_dir" in result
        assert "total_bookmarks" in result
        assert "files_created" in result
        assert isinstance(result["files_created"], list)

    def test_split_by_bookmarks_creates_files(self, sample_pdf, temp_dir):
        """Test that splitting creates output files."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.split_pdf_by_bookmarks(sample_pdf, temp_dir)

        # Even if no bookmarks, should report results
        assert "files_created" in result
        for file_info in result["files_created"]:
            assert "path" in file_info
            assert "title" in file_info
            assert "page_range" in file_info

    def test_split_by_bookmarks_no_bookmarks(self, temp_dir):
        """Test handling of PDFs without bookmarks."""
        # Use a simple PDF without bookmarks
        pdf_path = get_test_pdf("pdf_sample2.pdf")
        if not pdf_path:
            pytest.skip("Test PDF not available")

        result = pdf_tools.split_pdf_by_bookmarks(pdf_path, temp_dir)

        # Should indicate no bookmarks found
        assert result["total_bookmarks"] == 0

    def test_split_by_bookmarks_invalid_pdf(self, temp_dir):
        """Test error handling for invalid PDF."""
        with pytest.raises(PdfToolError):
            pdf_tools.split_pdf_by_bookmarks("/nonexistent.pdf", temp_dir)

    def test_split_by_pages(self, sample_pdf, temp_dir):
        """Test splitting PDF by page ranges."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Split every 2 pages
        result = pdf_tools.split_pdf_by_pages(sample_pdf, temp_dir, pages_per_split=2)

        assert "files_created" in result
        assert len(result["files_created"]) > 0


# =============================================================================
# PDF Comparison Tests
# =============================================================================


class TestComparePdfs:
    """Tests for PDF comparison/diff functionality."""

    def test_compare_pdfs_returns_structure(self, sample_pdf):
        """Test that compare_pdfs returns proper structure."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Compare PDF with itself (should be identical)
        result = pdf_tools.compare_pdfs(sample_pdf, sample_pdf)

        assert "pdf1_path" in result
        assert "pdf2_path" in result
        assert "are_identical" in result
        assert "differences" in result
        assert isinstance(result["differences"], list)
        assert "summary" in result

    def test_compare_pdfs_identical(self, sample_pdf):
        """Test comparing identical PDFs."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        result = pdf_tools.compare_pdfs(sample_pdf, sample_pdf)

        assert result["are_identical"] is True
        assert len(result["differences"]) == 0

    def test_compare_pdfs_different(self, multiple_pdfs):
        """Test comparing different PDFs."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need at least 2 PDFs for comparison")

        result = pdf_tools.compare_pdfs(multiple_pdfs[0], multiple_pdfs[1])

        assert result["are_identical"] is False
        assert len(result["differences"]) > 0

    def test_compare_pdfs_text_diff(self, sample_pdf, temp_dir):
        """Test that comparison detects text differences."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Create a modified PDF
        import pymupdf
        modified_path = os.path.join(temp_dir, "modified.pdf")
        
        with pymupdf.open(sample_pdf) as doc:
            # Add some text to first page
            page = doc[0]
            page.insert_text((100, 100), "TEST MODIFICATION")
            doc.save(modified_path)

        result = pdf_tools.compare_pdfs(sample_pdf, modified_path)

        assert result["are_identical"] is False
        # Should detect text difference
        text_diffs = [d for d in result["differences"] if d.get("type") == "text"]
        assert len(text_diffs) > 0

    def test_compare_pdfs_page_count_diff(self, sample_pdf, temp_dir):
        """Test that comparison detects page count differences."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Create a PDF with different page count
        import pymupdf
        modified_path = os.path.join(temp_dir, "extra_page.pdf")
        
        with pymupdf.open(sample_pdf) as doc:
            # Add a new page
            doc.new_page()
            doc.save(modified_path)

        result = pdf_tools.compare_pdfs(sample_pdf, modified_path)

        assert result["are_identical"] is False
        # Should detect page count difference
        page_diffs = [d for d in result["differences"] if d.get("type") == "page_count"]
        assert len(page_diffs) > 0

    def test_compare_pdfs_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(PdfToolError):
            pdf_tools.compare_pdfs("/nonexistent1.pdf", "/nonexistent2.pdf")


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcess:
    """Tests for batch processing multiple PDFs."""

    def test_batch_process_returns_structure(self, multiple_pdfs):
        """Test that batch_process returns proper structure."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        result = pdf_tools.batch_process(
            pdf_paths=multiple_pdfs,
            operation="get_info"
        )

        assert "operation" in result
        assert "total_files" in result
        assert "successful" in result
        assert "failed" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_batch_process_get_info(self, multiple_pdfs):
        """Test batch getting PDF info."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        result = pdf_tools.batch_process(
            pdf_paths=multiple_pdfs,
            operation="get_info"
        )

        assert result["total_files"] == len(multiple_pdfs)
        assert result["successful"] == len(multiple_pdfs)
        assert len(result["results"]) == len(multiple_pdfs)

    def test_batch_process_extract_text(self, multiple_pdfs):
        """Test batch text extraction."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        result = pdf_tools.batch_process(
            pdf_paths=multiple_pdfs,
            operation="extract_text"
        )

        assert result["successful"] >= 1
        for r in result["results"]:
            if r["success"]:
                assert "text" in r["result"] or "error" not in r

    def test_batch_process_extract_links(self, multiple_pdfs):
        """Test batch link extraction."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        result = pdf_tools.batch_process(
            pdf_paths=multiple_pdfs,
            operation="extract_links"
        )

        assert result["total_files"] == len(multiple_pdfs)
        for r in result["results"]:
            assert "pdf_path" in r

    def test_batch_process_invalid_operation(self, sample_pdf):
        """Test error handling for invalid operation."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        with pytest.raises(PdfToolError):
            pdf_tools.batch_process([sample_pdf], operation="invalid_op")

    def test_batch_process_partial_failure(self, sample_pdf, temp_dir):
        """Test handling of partial failures in batch."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Mix valid and invalid paths
        pdf_paths = [sample_pdf, "/nonexistent/file.pdf"]

        result = pdf_tools.batch_process(
            pdf_paths=pdf_paths,
            operation="get_info"
        )

        assert result["total_files"] == 2
        assert result["successful"] == 1
        assert result["failed"] == 1

    def test_batch_process_optimize(self, multiple_pdfs, temp_dir):
        """Test batch PDF optimization."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        result = pdf_tools.batch_process(
            pdf_paths=multiple_pdfs,
            operation="optimize",
            output_dir=temp_dir
        )

        assert result["successful"] >= 1
        # Check files were created
        for r in result["results"]:
            if r["success"]:
                assert os.path.exists(r["result"]["output_path"])


# =============================================================================
# MCP Layer Tests for Phase 3
# =============================================================================


class TestMcpLayerPhase3:
    """Test MCP tool wrappers for Phase 3 features."""

    def test_mcp_extract_links_exists(self):
        """Test that MCP extract_links tool exists."""
        from pdf_mcp import server
        
        # Check tool is registered
        assert hasattr(server, "extract_links")

    def test_mcp_optimize_pdf_exists(self):
        """Test that MCP optimize_pdf tool exists."""
        from pdf_mcp import server
        
        assert hasattr(server, "optimize_pdf")

    def test_mcp_detect_barcodes_exists(self):
        """Test that MCP detect_barcodes tool exists."""
        from pdf_mcp import server
        
        assert hasattr(server, "detect_barcodes")

    def test_mcp_split_pdf_by_bookmarks_exists(self):
        """Test that MCP split_pdf_by_bookmarks tool exists."""
        from pdf_mcp import server
        
        assert hasattr(server, "split_pdf_by_bookmarks")

    def test_mcp_compare_pdfs_exists(self):
        """Test that MCP compare_pdfs tool exists."""
        from pdf_mcp import server
        
        assert hasattr(server, "compare_pdfs")

    def test_mcp_batch_process_exists(self):
        """Test that MCP batch_process tool exists."""
        from pdf_mcp import server
        
        assert hasattr(server, "batch_process")


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestPhase3Workflows:
    """End-to-end workflow tests combining Phase 3 features."""

    def test_analyze_and_optimize_workflow(self, sample_pdf, temp_dir):
        """Test workflow: extract info -> extract links -> optimize."""
        if not sample_pdf:
            pytest.skip("Sample PDF not available")

        # Step 1: Extract links
        links_result = pdf_tools.extract_links(sample_pdf)
        
        # Step 2: Optimize
        output_path = os.path.join(temp_dir, "optimized.pdf")
        opt_result = pdf_tools.optimize_pdf(sample_pdf, output_path)

        # Verify workflow completed
        assert links_result["pdf_path"] == sample_pdf
        assert os.path.exists(opt_result["output_path"])

    def test_batch_analysis_workflow(self, multiple_pdfs, temp_dir):
        """Test workflow: batch extract text -> batch extract links."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need multiple PDFs for batch testing")

        # Step 1: Batch extract links
        links_result = pdf_tools.batch_process(multiple_pdfs, "extract_links")

        # Step 2: Batch get info
        info_result = pdf_tools.batch_process(multiple_pdfs, "get_info")

        assert links_result["total_files"] == len(multiple_pdfs)
        assert info_result["total_files"] == len(multiple_pdfs)

    def test_compare_and_report_workflow(self, multiple_pdfs):
        """Test workflow: compare PDFs and generate report."""
        if len(multiple_pdfs) < 2:
            pytest.skip("Need at least 2 PDFs for comparison")

        # Compare first two PDFs
        result = pdf_tools.compare_pdfs(multiple_pdfs[0], multiple_pdfs[1])

        # Generate a summary report
        summary = result["summary"]
        assert "pages" in summary or "text" in summary or isinstance(summary, str)
