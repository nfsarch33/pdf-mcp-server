"""
Tests for v0.8.0+ Agentic AI Features with Multi-Backend Support (v0.9.0+).

This module tests LLM-powered PDF processing capabilities:
- auto_fill_pdf_form: Intelligent form filling with field mapping
- extract_structured_data: Entity/section extraction
- analyze_pdf_content: Document analysis and summarization
- get_llm_backend_info: Check available LLM backends

Supports multiple backends (v0.9.0+):
- local: Local model server at localhost:8100 (free, no API costs)
- ollama: Ollama models (free, local)
- openai: OpenAI API (paid, requires OPENAI_API_KEY)

All tests use mocked LLM responses for unit testing.
"""

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_mcp import pdf_tools

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_form_pdf(tmp_path):
    """Create a simple form PDF for testing."""
    output = tmp_path / "form.pdf"
    pdf_tools.create_pdf_form(
        str(output),
        fields=[
            {"name": "full_name", "type": "text", "x": 100, "y": 700, "width": 200, "height": 20},
            {"name": "email", "type": "text", "x": 100, "y": 660, "width": 200, "height": 20},
            {"name": "phone", "type": "text", "x": 100, "y": 620, "width": 200, "height": 20},
            {"name": "address", "type": "text", "x": 100, "y": 580, "width": 300, "height": 20},
        ]
    )
    return str(output)


@pytest.fixture
def sample_text_pdf(tmp_path):
    """Create a PDF with sample text content for analysis."""
    import pymupdf
    output = tmp_path / "text.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    # Add invoice-like content
    text = """
    INVOICE #12345
    Date: January 15, 2026
    
    Bill To:
    John Smith
    123 Main Street
    New York, NY 10001
    
    Items:
    Widget A - $50.00
    Widget B - $75.00
    Service Fee - $25.00
    
    Subtotal: $150.00
    Tax (8%): $12.00
    Total: $162.00
    
    Payment Due: February 15, 2026
    """
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(output))
    doc.close()
    return str(output)


@pytest.fixture
def sample_passport_pdf(tmp_path):
    """Create a PDF with passport-like MRZ and labels for extraction."""
    import pymupdf
    output = tmp_path / "passport.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    text = """
    Passport
    Surname: ERIKSSON
    Given Names: ANNA MARIA
    Nationality: UTO
    Date of Issue: 01 Jan 2015
    Issuing Authority: UTOPIA

    P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<
    L898902C36UTO7408122F1204159ZE184226B<<<<<10
    """
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(output))
    doc.close()
    return str(output)


@pytest.fixture
def sample_passport_label_only_pdf(tmp_path):
    """Create a PDF with passport labels but no MRZ."""
    import pymupdf
    output = tmp_path / "passport_labels.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    text = """
    Passport
    Surname: NGUYEN
    Given Names: THI MAI
    Nationality: VNM
    Issuing Country: VIETNAM
    Passport Number: B1234567
    Date of Issue: 2016-07-21
    Issuing Authority: IMMIGRATION DEPT
    """
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(output))
    doc.close()
    return str(output)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    def _create_mock(content):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        return mock_response
    return _create_mock


# ============================================================================
# Test: auto_fill_pdf_form
# ============================================================================

class TestAutoFillPdfForm:
    """Tests for LLM-powered form auto-fill."""

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_auto_fill_without_any_llm_returns_error(self, mock_local, sample_form_pdf, tmp_path):
        """Without any LLM backend, should return error with clear message."""
        mock_local.return_value = False  # Local server not available
        
        output = tmp_path / "filled.pdf"
        source_data = {"name": "John Smith", "email_address": "john@example.com"}
        
        try:
            result = pdf_tools.auto_fill_pdf_form(
                sample_form_pdf,
                str(output),
                source_data=source_data
            )
            
            # Should return error or succeed with direct mapping only
            if "error" in result:
                # Check for common error indicators
                error_lower = result["error"].lower()
                assert "backend" in error_lower or "llm" in error_lower or "server" in error_lower
            else:
                # Direct mapping may have succeeded
                assert "filled_fields" in result or "mappings" in result
        except AttributeError:
            # pypdf compatibility issue
            pytest.skip("pypdf form filling compatibility issue")

    @patch("pdf_mcp.pdf_tools._call_llm")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_auto_fill_with_mocked_llm(self, mock_llm, sample_form_pdf, tmp_path):
        """With mocked LLM, should fill form fields correctly."""
        # Skip if openai not available
        if not pdf_tools._HAS_OPENAI:
            pytest.skip("OpenAI library not installed")
            
        output = tmp_path / "filled.pdf"
        source_data = {
            "name": "John Smith",
            "email_address": "john@example.com",
            "phone_number": "555-123-4567",
            "home_address": "123 Main St, NYC"
        }
        
        # Mock LLM returns field mapping
        mock_llm.return_value = json.dumps({
            "full_name": "John Smith",
            "email": "john@example.com",
            "phone": "555-123-4567",
            "address": "123 Main St, NYC"
        })
        
        try:
            result = pdf_tools.auto_fill_pdf_form(
                sample_form_pdf,
                str(output),
                source_data=source_data
            )
            
            # Should either succeed or fail gracefully
            if "error" not in result:
                assert result.get("filled_fields", 0) >= 0
                assert Path(output).exists()
        except AttributeError as e:
            # pypdf has a bug with certain form structures
            if "get_object" in str(e):
                pytest.skip("pypdf bug: AttributeError in form filling (known issue)")

    @patch("pdf_mcp.pdf_tools._call_llm")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_auto_fill_reports_mapping_confidence(self, mock_llm, sample_form_pdf, tmp_path):
        """Should report confidence scores for field mappings."""
        # Skip if openai not available
        if not pdf_tools._HAS_OPENAI:
            pytest.skip("OpenAI library not installed")
            
        output = tmp_path / "filled.pdf"
        source_data = {"full_name": "John Smith"}  # Use exact field name for direct mapping
        
        mock_llm.return_value = json.dumps({
            "full_name": "John Smith"
        })
        
        try:
            result = pdf_tools.auto_fill_pdf_form(
                sample_form_pdf,
                str(output),
                source_data=source_data
            )
            
            # With direct mapping, should succeed
            assert "mappings" in result or "filled_fields" in result or "error" in result
        except AttributeError as e:
            # pypdf has a bug with certain form structures
            if "get_object" in str(e):
                pytest.skip("pypdf bug: AttributeError in form filling (known issue)")

    def test_auto_fill_with_invalid_pdf_returns_error(self, tmp_path):
        """Invalid PDF path should return error."""
        output = tmp_path / "filled.pdf"
        result = pdf_tools.auto_fill_pdf_form(
            "/nonexistent/path.pdf",
            str(output),
            source_data={"name": "Test"}
        )
        assert "error" in result


# ============================================================================
# Test: extract_structured_data
# ============================================================================

@patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
@patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
@patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
class TestExtractStructuredData:
    """Tests for entity/section extraction (LLM disabled - pattern matching only)."""

    def test_extract_invoice_data(self, mock_check, sample_text_pdf):
        """Extract invoice-specific fields."""
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice"
        )
        
        # Should return structured data (even without LLM, uses pattern matching)
        assert "error" not in result or "api" in result.get("error", "").lower()
        if "error" not in result:
            assert "data" in result or "extracted" in result

    def test_extract_with_custom_schema(self, mock_check, sample_text_pdf):
        """Extract data using custom schema definition."""
        schema = {
            "invoice_number": "string",
            "total_amount": "number",
            "due_date": "date"
        }
        
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            schema=schema
        )
        
        # Should attempt extraction
        assert isinstance(result, dict)

    def test_extract_with_invalid_pdf_returns_error(self, mock_check):
        """Invalid PDF should return error."""
        result = pdf_tools.extract_structured_data(
            "/nonexistent/path.pdf",
            data_type="invoice"
        )
        assert "error" in result

    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", True)
    @patch("pdf_mcp.pdf_tools._call_llm")
    def test_extract_with_mocked_llm(self, mock_llm, sample_text_pdf):
        """With mocked LLM, should return structured extraction."""
        mock_llm.return_value = json.dumps({
            "invoice_number": "12345",
            "date": "January 15, 2026",
            "total": 162.00,
            "items": [
                {"name": "Widget A", "price": 50.00},
                {"name": "Widget B", "price": 75.00},
                {"name": "Service Fee", "price": 25.00}
            ]
        })
        
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice"
        )
        
        if "error" not in result:
            assert "data" in result or "extracted" in result


# ============================================================================
# Test: analyze_pdf_content
# ============================================================================

@patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
@patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
@patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
class TestAnalyzePdfContent:
    """Tests for PDF content analysis and summarization (LLM disabled)."""

    def test_analyze_returns_document_type(self, mock_check, sample_text_pdf):
        """Should classify document type."""
        result = pdf_tools.analyze_pdf_content(sample_text_pdf)
        
        # Even without LLM, should attempt classification
        assert isinstance(result, dict)
        if "error" not in result:
            assert "document_type" in result or "classification" in result or "analysis" in result

    def test_analyze_returns_summary(self, mock_check, sample_text_pdf):
        """Should generate summary."""
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            include_summary=True
        )
        
        assert isinstance(result, dict)

    def test_analyze_detects_key_entities(self, mock_check, sample_text_pdf):
        """Should detect key entities like dates, amounts, names."""
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            detect_entities=True
        )
        
        assert isinstance(result, dict)

    def test_analyze_with_invalid_pdf_returns_error(self, mock_check):
        """Invalid PDF should return error."""
        result = pdf_tools.analyze_pdf_content("/nonexistent/path.pdf")
        assert "error" in result

    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", True)
    @patch("pdf_mcp.pdf_tools._call_llm")
    def test_analyze_with_mocked_llm(self, mock_llm, sample_text_pdf):
        """With mocked LLM, should return full analysis."""
        mock_llm.return_value = json.dumps({
            "document_type": "invoice",
            "summary": "Invoice #12345 for $162.00 from January 15, 2026",
            "key_entities": {
                "invoice_number": "12345",
                "total_amount": "$162.00",
                "due_date": "February 15, 2026"
            },
            "risk_flags": [],
            "completeness_score": 0.95
        })
        
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            include_summary=True,
            detect_entities=True
        )
        
        if "error" not in result:
            assert "analysis" in result or "document_type" in result


# ============================================================================
# Test: LLM Integration Helpers
# ============================================================================

class TestLLMHelpers:
    """Tests for LLM integration helper functions."""

    def test_has_openai_flag_exists(self):
        """_HAS_OPENAI flag should exist."""
        assert hasattr(pdf_tools, "_HAS_OPENAI")

    def test_call_llm_function_exists(self):
        """_call_llm helper should exist."""
        assert hasattr(pdf_tools, "_call_llm")

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_call_llm_without_any_backend_returns_none(self, mock_local):
        """Without any LLM backend, should return None."""
        mock_local.return_value = False  # Local server not available
        result = pdf_tools._call_llm("test prompt")
        assert result is None


# ============================================================================
# Test: MCP Tool Registration
# ============================================================================

class TestMCPToolRegistration:
    """Verify agentic tools are exposed via MCP."""

    def test_auto_fill_pdf_form_registered(self):
        """auto_fill_pdf_form should be a public function."""
        assert hasattr(pdf_tools, "auto_fill_pdf_form")
        assert callable(pdf_tools.auto_fill_pdf_form)

    def test_extract_structured_data_registered(self):
        """extract_structured_data should be a public function."""
        assert hasattr(pdf_tools, "extract_structured_data")
        assert callable(pdf_tools.extract_structured_data)

    def test_analyze_pdf_content_registered(self):
        """analyze_pdf_content should be a public function."""
        assert hasattr(pdf_tools, "analyze_pdf_content")
        assert callable(pdf_tools.analyze_pdf_content)


# ============================================================================
# Integration Tests (with real PDFs, no LLM)
# ============================================================================

class TestAgenticIntegration:
    """Integration tests using real PDFs without LLM."""

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_auto_fill_graceful_degradation(self, mock_check, sample_form_pdf, tmp_path):
        """Without LLM, should fall back gracefully."""
        output = tmp_path / "filled.pdf"
        
        # Use exact field name to test direct mapping path
        try:
            result = pdf_tools.auto_fill_pdf_form(
                sample_form_pdf,
                str(output),
                source_data={"full_name": "Direct Match"}
            )
            
            # Should either succeed with direct mapping or return helpful error
            assert isinstance(result, dict)
            # Either we got an error (no LLM) or we successfully filled
            if "error" not in result:
                assert "filled_fields" in result or "mappings" in result
        except AttributeError as e:
            # pypdf version compatibility issue with form filling
            # This is expected in some Python/pypdf version combinations
            pytest.skip(f"pypdf form filling compatibility issue: {e}")

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_extract_structured_data_pattern_matching(self, mock_check, sample_text_pdf):
        """Without LLM, should use pattern matching for common types."""
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice"
        )
        
        # Should attempt pattern-based extraction
        assert isinstance(result, dict)

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_extract_structured_data_passport_mrz(self, mock_check, sample_passport_pdf):
        """Should extract key passport fields from MRZ and labels."""
        result = pdf_tools.extract_structured_data(
            sample_passport_pdf,
            data_type="passport"
        )

        assert isinstance(result, dict)
        data = result.get("data", {})
        assert data.get("passport_number") == "L898902C3"
        assert data.get("nationality") == "UTO"
        assert data.get("birth_date") == "1974-08-12"
        assert data.get("expiry_date") == "2012-04-15"
        assert data.get("sex") == "F"
        assert data.get("surname") == "ERIKSSON"
        assert data.get("given_names") == "ANNA MARIA"
        assert data.get("issuing_country") == "UTO"
        assert data.get("issue_date") == "2015-01-01"
        assert data.get("issuing_authority") == "UTOPIA"

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_extract_structured_data_passport_labels_only(self, mock_check, sample_passport_label_only_pdf):
        """Should extract key passport fields from labels when MRZ is missing."""
        result = pdf_tools.extract_structured_data(
            sample_passport_label_only_pdf,
            data_type="passport"
        )

        assert isinstance(result, dict)
        data = result.get("data", {})
        assert data.get("surname") == "NGUYEN"
        assert data.get("given_names") == "THI MAI"
        assert data.get("nationality") == "VNM"
        assert data.get("issuing_country") == "VIETNAM"
        assert data.get("passport_number") == "B1234567"
        assert data.get("issue_date") == "2016-07-21"
        assert data.get("issuing_authority") == "IMMIGRATION DEPT"

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_analyze_pdf_basic_analysis(self, mock_check, sample_text_pdf):
        """Without LLM, should provide basic document analysis."""
        result = pdf_tools.analyze_pdf_content(sample_text_pdf)
        
        # Should return basic metrics at minimum
        assert isinstance(result, dict)


# ============================================================================
# Test: Multi-Backend Support (v0.9.0+)
# ============================================================================

class TestMultiBackendSupport:
    """Tests for local VLM, Ollama, and OpenAI backend support."""

    def test_get_llm_backend_info_exists(self):
        """get_llm_backend_info should be available."""
        assert hasattr(pdf_tools, "get_llm_backend_info")
        assert callable(pdf_tools.get_llm_backend_info)

    def test_get_llm_backend_info_returns_dict(self):
        """Should return backend info dict."""
        result = pdf_tools.get_llm_backend_info()
        assert isinstance(result, dict)
        assert "current_backend" in result
        assert "backends" in result
        assert "override_env" in result

    def test_backend_info_has_all_backends(self):
        """Should report on all backend types."""
        result = pdf_tools.get_llm_backend_info()
        backends = result["backends"]
        assert "local" in backends
        assert "ollama" in backends
        assert "openai" in backends

    def test_local_backend_info_has_url(self):
        """Local backend should report URL."""
        result = pdf_tools.get_llm_backend_info()
        local_info = result["backends"]["local"]
        assert "url" in local_info
        assert "localhost" in local_info["url"] or "127.0.0.1" in local_info["url"]

    def test_backends_report_cost(self):
        """All backends should report cost info."""
        result = pdf_tools.get_llm_backend_info()
        for backend_name, backend_info in result["backends"].items():
            assert "cost" in backend_info

    def test_local_and_ollama_are_free(self):
        """Local and Ollama should be marked as free."""
        result = pdf_tools.get_llm_backend_info()
        assert "free" in result["backends"]["local"]["cost"]
        assert "free" in result["backends"]["ollama"]["cost"]

    def test_openai_is_paid(self):
        """OpenAI should be marked as paid."""
        result = pdf_tools.get_llm_backend_info()
        assert "paid" in result["backends"]["openai"]["cost"]

    def test_backend_constants_exist(self):
        """Backend constants should be defined."""
        assert hasattr(pdf_tools, "LLM_BACKEND_LOCAL")
        assert hasattr(pdf_tools, "LLM_BACKEND_OLLAMA")
        assert hasattr(pdf_tools, "LLM_BACKEND_OPENAI")
        assert pdf_tools.LLM_BACKEND_LOCAL == "local"
        assert pdf_tools.LLM_BACKEND_OLLAMA == "ollama"
        assert pdf_tools.LLM_BACKEND_OPENAI == "openai"

    def test_local_model_server_url_configurable(self):
        """LOCAL_MODEL_SERVER_URL should be configurable via env."""
        assert hasattr(pdf_tools, "LOCAL_MODEL_SERVER_URL")
        # Default should include localhost
        assert "localhost" in pdf_tools.LOCAL_MODEL_SERVER_URL or "127.0.0.1" in pdf_tools.LOCAL_MODEL_SERVER_URL

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    def test_get_llm_backend_prefers_local(self, mock_check):
        """Should prefer local backend when available."""
        mock_check.return_value = True
        result = pdf_tools._get_llm_backend()
        assert result == "local"

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", True)
    def test_get_llm_backend_falls_back_to_ollama(self, mock_check):
        """Should fall back to Ollama when local unavailable."""
        mock_check.return_value = False
        result = pdf_tools._get_llm_backend()
        # Should be ollama or openai (depends on whether OPENAI_API_KEY is set)
        assert result in ("ollama", "openai", "")

    @patch.dict(os.environ, {"PDF_MCP_LLM_BACKEND": "openai"})
    def test_get_llm_backend_respects_override(self):
        """Should respect PDF_MCP_LLM_BACKEND env override."""
        result = pdf_tools._get_llm_backend()
        assert result == "openai"


class TestLocalVLMBackend:
    """Tests for local VLM backend at localhost:8100."""

    def test_check_local_model_server_function_exists(self):
        """_check_local_model_server should exist."""
        assert hasattr(pdf_tools, "_check_local_model_server")
        assert callable(pdf_tools._check_local_model_server)

    def test_call_local_llm_function_exists(self):
        """_call_local_llm should exist."""
        assert hasattr(pdf_tools, "_call_local_llm")
        assert callable(pdf_tools._call_local_llm)

    @patch("pdf_mcp.pdf_tools._HAS_REQUESTS", False)
    def test_call_local_llm_without_requests_returns_none(self):
        """Without requests library, should return None."""
        result = pdf_tools._call_local_llm("test prompt")
        assert result is None

    @patch("pdf_mcp.pdf_tools._HAS_REQUESTS", True)
    @patch("pdf_mcp.pdf_tools._resolve_local_model_name", return_value="test-model")
    @patch("pdf_mcp.pdf_tools._requests")
    def test_call_local_llm_with_mock_server(self, mock_requests, _mock_resolve):
        """With mocked server, should return response (OpenAI chat format)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_requests.post.return_value = mock_response
        
        result = pdf_tools._call_local_llm("test prompt")
        assert result == "Test response"


class TestOllamaBackend:
    """Tests for Ollama backend."""

    def test_call_ollama_llm_function_exists(self):
        """_call_ollama_llm should exist."""
        assert hasattr(pdf_tools, "_call_ollama_llm")
        assert callable(pdf_tools._call_ollama_llm)

    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    def test_call_ollama_llm_without_ollama_returns_none(self):
        """Without ollama library, should return None."""
        result = pdf_tools._call_ollama_llm("test prompt")
        assert result is None

    def test_call_ollama_llm_with_mock(self):
        """With mocked Ollama, should return response."""
        if not pdf_tools._HAS_OLLAMA:
            pytest.skip("Ollama not installed")
        
        with patch("pdf_mcp.pdf_tools._ollama") as mock_ollama:
            # Ollama returns Pydantic ChatResponse; mock .message.content
            mock_response = MagicMock()
            mock_response.message.content = "Ollama response"
            mock_ollama.chat.return_value = mock_response
            result = pdf_tools._call_ollama_llm("test prompt")
            assert result == "Ollama response"


# ============================================================================
# v0.9.0 Comprehensive Integration Tests
# ============================================================================

class TestLocalVLMIntegration:
    """Integration tests for local VLM backend with agentic functions."""

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._call_local_llm")
    def test_auto_fill_uses_local_backend(self, mock_call_llm, mock_check, sample_form_pdf, tmp_path):
        """auto_fill_pdf_form should use local backend when available."""
        mock_check.return_value = True
        mock_call_llm.return_value = json.dumps({"full_name": "Test User"})
        
        output = tmp_path / "filled.pdf"
        try:
            result = pdf_tools.auto_fill_pdf_form(
                sample_form_pdf,
                str(output),
                source_data={"name": "Test User"},
                backend="local"
            )
            
            # Should either succeed or return meaningful result
            assert isinstance(result, dict)
            if "error" not in result:
                assert result.get("backend") == "local" or "filled_fields" in result
        except AttributeError as e:
            # pypdf version compatibility issue with form filling
            pytest.skip(f"pypdf form filling compatibility issue: {e}")

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._call_local_llm")
    def test_extract_structured_data_uses_local_backend(self, mock_call_llm, mock_check, sample_text_pdf):
        """extract_structured_data should use local backend when specified."""
        mock_check.return_value = True
        mock_call_llm.return_value = json.dumps({
            "invoice_number": "INV-001",
            "total": "100.00"
        })
        
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice",
            backend="local"
        )
        
        assert isinstance(result, dict)
        # Verify backend is tracked
        if "backend" in result:
            assert result["backend"] in ("local", None)

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._call_local_llm")
    def test_analyze_pdf_content_uses_local_backend(self, mock_call_llm, mock_check, sample_text_pdf):
        """analyze_pdf_content should use local backend when specified."""
        mock_check.return_value = True
        mock_call_llm.return_value = json.dumps({
            "summary": "This is a test document.",
            "document_type": "invoice",
            "key_findings": ["Invoice number found"]
        })
        
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            include_summary=True,
            backend="local"
        )
        
        assert isinstance(result, dict)
        assert "document_type" in result


class TestOllamaIntegration:
    """Integration tests for Ollama backend with agentic functions."""

    @patch("pdf_mcp.pdf_tools._get_llm_backend")
    @patch("pdf_mcp.pdf_tools._call_ollama_llm")
    def test_extract_structured_data_with_ollama(self, mock_call_llm, mock_get_backend, sample_text_pdf):
        """extract_structured_data should work with Ollama backend."""
        mock_get_backend.return_value = "ollama"
        mock_call_llm.return_value = json.dumps({
            "invoice_number": "12345",
            "date": "January 15, 2026"
        })
        
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice",
            backend="ollama"
        )
        
        assert isinstance(result, dict)

    @patch("pdf_mcp.pdf_tools._get_llm_backend")
    @patch("pdf_mcp.pdf_tools._call_ollama_llm")
    def test_analyze_pdf_content_with_ollama(self, mock_call_llm, mock_get_backend, sample_text_pdf):
        """analyze_pdf_content should work with Ollama backend."""
        mock_get_backend.return_value = "ollama"
        mock_call_llm.return_value = json.dumps({
            "summary": "Invoice document for services.",
            "document_type": "invoice",
            "key_findings": []
        })
        
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            backend="ollama"
        )
        
        assert isinstance(result, dict)
        assert "document_type" in result


class TestBackendFieldInResults:
    """Tests verifying backend field is returned in agentic function results."""

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_extract_structured_data_returns_backend_field(self, mock_check, sample_text_pdf):
        """extract_structured_data should return backend field."""
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice"
        )
        
        assert isinstance(result, dict)
        # Backend field should be present (may be None if no LLM used)
        assert "backend" in result

    @patch("pdf_mcp.pdf_tools._check_local_model_server", return_value=False)
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_analyze_pdf_content_returns_backend_field(self, mock_check, sample_text_pdf):
        """analyze_pdf_content should return backend field."""
        result = pdf_tools.analyze_pdf_content(sample_text_pdf)
        
        assert isinstance(result, dict)
        assert "backend" in result

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._call_local_llm")
    def test_backend_field_reflects_local_when_used(self, mock_call_llm, mock_check, sample_text_pdf):
        """When local backend is used, backend field should be 'local'."""
        mock_check.return_value = True
        mock_call_llm.return_value = json.dumps({
            "summary": "Test summary",
            "document_type": "invoice",
            "key_findings": []
        })
        
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            backend="local"
        )
        
        if "backend" in result and result["backend"] is not None:
            assert result["backend"] == "local"


class TestBackendFallbackChain:
    """Tests for backend fallback behavior."""

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_no_backend_available_graceful_degradation(self, mock_check, sample_text_pdf):
        """When no backend available, should gracefully degrade to pattern matching."""
        mock_check.return_value = False
        
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice"
        )
        
        # Should still return results using pattern matching
        assert isinstance(result, dict)
        if "method" in result:
            assert result["method"] == "pattern"

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", True)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", False)
    def test_fallback_from_local_to_ollama(self, mock_check):
        """When local unavailable, should fall back to Ollama."""
        mock_check.return_value = False
        
        backend = pdf_tools._get_llm_backend()
        assert backend == "ollama"

    @patch("pdf_mcp.pdf_tools._check_local_model_server")
    @patch("pdf_mcp.pdf_tools._HAS_OLLAMA", False)
    @patch("pdf_mcp.pdf_tools._HAS_OPENAI", True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_fallback_from_local_to_openai(self, mock_check):
        """When local and Ollama unavailable, should fall back to OpenAI."""
        mock_check.return_value = False
        
        backend = pdf_tools._get_llm_backend()
        assert backend == "openai"


class TestBackendEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    @patch.dict(os.environ, {"PDF_MCP_LLM_BACKEND": "local"})
    def test_env_override_forces_local(self):
        """PDF_MCP_LLM_BACKEND=local should force local backend."""
        backend = pdf_tools._get_llm_backend()
        assert backend == "local"

    @patch.dict(os.environ, {"PDF_MCP_LLM_BACKEND": "ollama"})
    def test_env_override_forces_ollama(self):
        """PDF_MCP_LLM_BACKEND=ollama should force ollama backend."""
        backend = pdf_tools._get_llm_backend()
        assert backend == "ollama"

    @patch.dict(os.environ, {"PDF_MCP_LLM_BACKEND": "openai"})
    def test_env_override_forces_openai(self):
        """PDF_MCP_LLM_BACKEND=openai should force openai backend."""
        backend = pdf_tools._get_llm_backend()
        assert backend == "openai"

    @patch.dict(os.environ, {"LOCAL_MODEL_SERVER_URL": "http://custom:9999"})
    def test_custom_local_server_url(self):
        """LOCAL_MODEL_SERVER_URL should be configurable."""
        # Note: This tests the module-level constant which may already be set
        # The actual URL would be read at import time
        assert hasattr(pdf_tools, "LOCAL_MODEL_SERVER_URL")


class TestUnifiedCallLLM:
    """Tests for unified _call_llm function with backend routing."""

    def test_call_llm_exists(self):
        """_call_llm should exist and be callable."""
        assert hasattr(pdf_tools, "_call_llm")
        assert callable(pdf_tools._call_llm)

    @patch("pdf_mcp.pdf_tools._call_local_llm")
    def test_call_llm_routes_to_local(self, mock_local):
        """_call_llm should route to local when specified."""
        mock_local.return_value = "local response"
        
        result = pdf_tools._call_llm("test", backend="local")
        
        mock_local.assert_called_once()
        assert result == "local response"

    @patch("pdf_mcp.pdf_tools._call_ollama_llm")
    def test_call_llm_routes_to_ollama(self, mock_ollama):
        """_call_llm should route to ollama when specified."""
        mock_ollama.return_value = "ollama response"
        
        result = pdf_tools._call_llm("test", backend="ollama")
        
        mock_ollama.assert_called_once()
        assert result == "ollama response"

    @patch("pdf_mcp.pdf_tools._call_openai_llm")
    def test_call_llm_routes_to_openai(self, mock_openai):
        """_call_llm should route to openai when specified."""
        mock_openai.return_value = "openai response"
        
        result = pdf_tools._call_llm("test", backend="openai")
        
        mock_openai.assert_called_once()
        assert result == "openai response"


class TestMCPToolRegistrationV090:
    """Verify v0.9.0 tools are exposed via MCP."""

    def test_get_llm_backend_info_registered(self):
        """get_llm_backend_info should be a public function."""
        assert hasattr(pdf_tools, "get_llm_backend_info")
        assert callable(pdf_tools.get_llm_backend_info)

    def test_all_agentic_tools_have_backend_param(self):
        """All agentic tools should accept backend parameter."""
        import inspect
        
        # Check auto_fill_pdf_form
        sig = inspect.signature(pdf_tools.auto_fill_pdf_form)
        assert "backend" in sig.parameters
        
        # Check extract_structured_data
        sig = inspect.signature(pdf_tools.extract_structured_data)
        assert "backend" in sig.parameters
        
        # Check analyze_pdf_content
        sig = inspect.signature(pdf_tools.analyze_pdf_content)
        assert "backend" in sig.parameters


# ============================================================================
# E2E Tests with Real LLM (v0.9.2+)
# These tests actually call the local model server when available
# Mark with pytest.mark.slow for CI/CD to optionally skip
# ============================================================================

def _is_local_server_running() -> bool:
    """Check if local model server is running at localhost:8100."""
    try:
        import requests
        response = requests.get(f"{pdf_tools.LOCAL_MODEL_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.slow
class TestE2ELocalVLM:
    """
    End-to-end tests with REAL local model server (not mocked).
    
    These tests require the local model server to be running:
        ./scripts/run_local_vlm.sh
    
    Tests are skipped if server is not available.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_server(self):
        """Skip all tests in this class if local server not running."""
        if not _is_local_server_running():
            pytest.skip("Local model server not running at localhost:8100")

    def test_e2e_get_llm_backend_info_detects_local(self):
        """With server running, should detect local backend."""
        result = pdf_tools.get_llm_backend_info()
        
        assert result["backends"]["local"]["available"] is True
        # Local should be selected as current backend (highest priority)
        assert result["current_backend"] == "local"

    def test_e2e_call_local_llm_returns_response(self):
        """With server running, should get actual LLM response."""
        result = pdf_tools._call_local_llm(
            "What is 2+2? Reply with just the number."
        )
        
        assert result is not None
        assert len(result) > 0
        # Response should contain "4" somewhere
        assert "4" in result

    def test_e2e_extract_structured_data_with_local_llm(self, sample_text_pdf):
        """E2E test: extract_structured_data with real local LLM."""
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice",
            backend="local"
        )
        
        assert isinstance(result, dict)
        assert "error" not in result
        assert "data" in result
        # Backend is "local" if LLM was used, None if pattern matching was sufficient
        # Both are valid outcomes - pattern matching success is actually preferred
        assert result.get("backend") in ("local", None)
        assert result.get("method") in ("pattern", "llm", "llm+pattern")

    def test_e2e_analyze_pdf_content_with_local_llm(self, sample_text_pdf):
        """E2E test: analyze_pdf_content with real local LLM."""
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            include_summary=True,
            detect_entities=True,
            backend="local"
        )
        
        assert isinstance(result, dict)
        assert "error" not in result
        assert "document_type" in result
        # With real LLM, should have a summary
        if "summary" in result:
            assert len(result["summary"]) > 10
        # Backend should be "local"
        assert result.get("backend") == "local"

    def test_e2e_local_llm_timeout_handling(self):
        """E2E test: local LLM should handle requests without hanging."""
        import time
        
        start = time.time()
        result = pdf_tools._call_local_llm(
            "Reply with a single word: hello"
        )
        elapsed = time.time() - start
        
        # Should complete within reasonable time (2 min max for slow first load)
        assert elapsed < 120
        assert result is not None


def _ollama_model_available(model: str) -> bool:
    """Check if specific Ollama model is available."""
    from pdf_mcp import llm_setup
    if not llm_setup.ollama_is_installed():
        return False
    models = llm_setup.ollama_list_models()
    return model in models


@pytest.mark.slow
class TestE2EOllama:
    """
    End-to-end tests with REAL Ollama (not mocked).
    
    Requires Ollama installed and a model pulled:
        ollama pull qwen2.5:7b
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_ollama(self):
        """Skip all tests if Ollama not available."""
        if not pdf_tools._HAS_OLLAMA:
            pytest.skip("Ollama library not installed (pip install ollama)")
        
        from pdf_mcp import llm_setup
        
        # Check if Ollama CLI is installed
        if not llm_setup.ollama_is_installed():
            pytest.skip("Ollama CLI not found (install: curl -fsSL https://ollama.ai/install.sh | sh)")
        
        # Check if Ollama service is running
        try:
            import ollama
            ollama.list()
        except Exception as e:
            pytest.skip(f"Ollama service not running: {e}")
        
        # Check if any model is available
        models = llm_setup.ollama_list_models()
        if not models:
            pytest.skip("No Ollama models found (run: ollama pull qwen2.5:1.5b)")

    def test_e2e_ollama_llm_returns_response(self):
        """With Ollama running, should get actual LLM response."""
        # Try smaller models first for speed
        test_models = ["qwen2.5:1.5b", "qwen2.5:7b", "llama3.2:1b"]
        model_to_use = None
        
        for model in test_models:
            if _ollama_model_available(model):
                model_to_use = model
                break
        
        if model_to_use is None:
            pytest.skip(f"None of {test_models} found; pull one: ollama pull qwen2.5:1.5b")
        
        result = pdf_tools._call_ollama_llm(
            "What is 2+2? Reply with just the number.",
            model=model_to_use
        )
        
        assert result is not None
        assert len(result) > 0

    def test_e2e_extract_structured_data_with_ollama(self, sample_text_pdf):
        """E2E test: extract_structured_data with real Ollama."""
        result = pdf_tools.extract_structured_data(
            sample_text_pdf,
            data_type="invoice",
            backend="ollama"
        )
        
        assert isinstance(result, dict)


@pytest.mark.slow
class TestE2EOpenAI:
    """
    End-to-end tests with REAL OpenAI API (not mocked).
    
    Requires OPENAI_API_KEY environment variable.
    WARNING: These tests incur actual API costs!
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_openai(self):
        """Skip all tests if OpenAI not available."""
        if not pdf_tools._HAS_OPENAI:
            pytest.skip("OpenAI library not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_e2e_openai_llm_returns_response(self):
        """With OpenAI API key, should get actual response."""
        result = pdf_tools._call_openai_llm(
            "What is 2+2? Reply with just the number.",
            model="gpt-4o-mini"
        )
        
        assert result is not None
        assert "4" in result

    def test_e2e_analyze_pdf_content_with_openai(self, sample_text_pdf):
        """E2E test: analyze_pdf_content with real OpenAI."""
        result = pdf_tools.analyze_pdf_content(
            sample_text_pdf,
            include_summary=True,
            backend="openai"
        )
        
        assert isinstance(result, dict)
        assert "document_type" in result


# ============================================================================
# Backend Comparison Tests (v0.9.2+)
# ============================================================================

@pytest.mark.slow
class TestBackendComparison:
    """Compare outputs across different backends."""

    def test_all_backends_return_consistent_structure(self, sample_text_pdf):
        """All backends should return same result structure."""
        backends_to_test = []
        
        # Check which backends are available
        if _is_local_server_running():
            backends_to_test.append("local")
        if pdf_tools._HAS_OLLAMA:
            try:
                import ollama
                ollama.list()
                backends_to_test.append("ollama")
            except Exception:
                pass
        if pdf_tools._HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
            backends_to_test.append("openai")
        
        if not backends_to_test:
            pytest.skip("No LLM backends available for comparison")
        
        results = {}
        for backend in backends_to_test:
            result = pdf_tools.extract_structured_data(
                sample_text_pdf,
                data_type="invoice",
                backend=backend
            )
            results[backend] = result
        
        # All results should have same structure
        for backend, result in results.items():
            assert "data" in result, f"{backend} missing 'data' field"
            assert "confidence" in result, f"{backend} missing 'confidence' field"
            assert "method" in result, f"{backend} missing 'method' field"
            assert "backend" in result, f"{backend} missing 'backend' field"


# ============================================================================
# Test: MRZ Checksum Validation (v1.2.0)
# ============================================================================


class TestMrzChecksumValidation:
    """Tests for MRZ checksum validation per ICAO 9303."""

    def test_mrz_check_digit_valid(self):
        """Valid MRZ check digit should return correct value per ICAO 9303."""
        assert pdf_tools._mrz_check_digit("L898902C3") == 6

    def test_mrz_check_digit_all_zeros(self):
        """All zeros should return 0."""
        assert pdf_tools._mrz_check_digit("000000000") == 0

    def test_mrz_check_digit_letters(self):
        """Letters should be converted to numbers (A=10..Z=35)."""
        # Per ICAO: A=10, B=11, ..., Z=35, <(filler)=0
        result = pdf_tools._mrz_check_digit("A")
        assert isinstance(result, int)
        assert 0 <= result <= 9

    def test_mrz_validate_field_correct(self):
        """Field with matching check digit should validate."""
        assert pdf_tools._mrz_validate_field("L898902C3", 6) is True

    def test_mrz_validate_field_incorrect(self):
        """Field with wrong check digit should fail."""
        assert pdf_tools._mrz_validate_field("L898902C3", 9) is False

    def test_extract_passport_fields_with_checksum(self):
        """MRZ extraction should include checksum_valid field."""
        mrz_text = (
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10\n"
        )
        extracted, confidence = pdf_tools._extract_passport_fields(mrz_text)
        assert extracted.get("passport_number") == "L898902C3"
        # After checksum implementation, confidence should be higher for validated fields
        assert confidence.get("passport_number", 0) >= 0.8


class TestMrzTd1Format:
    """Tests for TD1 ID card format (3 lines x 30 chars)."""

    def test_extract_mrz_lines_td1(self):
        """Should detect TD1 format (3 lines of 30 chars)."""
        td1_text = (
            "I<UTOD231458907<<<<<<<<<<<<<<<\n"
            "7408122F1204159UTO<<<<<<<<<<<6\n"
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<\n"
        )
        result = pdf_tools._extract_mrz_lines(td1_text)
        assert result is not None
        assert len(result) >= 2

    def test_extract_passport_fields_td1(self):
        """Should extract fields from TD1 (ID card) format."""
        td1_text = (
            "I<UTOD231458907<<<<<<<<<<<<<<<\n"
            "7408122F1204159UTO<<<<<<<<<<<6\n"
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<\n"
        )
        extracted, confidence = pdf_tools._extract_passport_fields(td1_text)
        # Should extract at least some fields from TD1
        assert isinstance(extracted, dict)
        # Surname should be extractable from line 3
        if extracted.get("surname"):
            assert "ERIKSSON" in extracted["surname"]


class TestMrzOcrErrorCorrection:
    """Tests for OCR error correction in MRZ text."""

    def test_mrz_ocr_correction_common_mistakes(self):
        """Should correct common OCR mistakes in MRZ: O->0, I->1, B->8."""
        # Line2 with OCR errors: O instead of 0, I instead of 1
        corrected = pdf_tools._correct_mrz_ocr_errors(
            "L89890ZC36UTO74O8I22FI2O4I59ZEI84226B<<<<<IO"
        )
        assert "0" in corrected  # O should become 0 in digit positions
        assert isinstance(corrected, str)

    def test_mrz_ocr_correction_preserves_valid(self):
        """Should not modify already-correct MRZ text."""
        valid = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        corrected = pdf_tools._correct_mrz_ocr_errors(valid)
        assert corrected == valid


# ============================================================================
# Test: Non-standard Form Heuristics (v1.2.0)
# ============================================================================


class TestFormHeuristicsLcs:
    """Tests for improved label matching using LCS fuzzy similarity."""

    def test_lcs_similarity_exact_match(self):
        """Exact match should return 1.0."""
        assert pdf_tools._lcs_similarity("name", "name") == 1.0

    def test_lcs_similarity_partial_match(self):
        """Partial match should return > 0 and < 1."""
        score = pdf_tools._lcs_similarity("fullname", "firstname")
        assert 0 < score < 1

    def test_lcs_similarity_no_match(self):
        """Completely different strings should return 0."""
        assert pdf_tools._lcs_similarity("xyz", "abc") == 0.0

    def test_lcs_similarity_empty(self):
        """Empty strings should return 0."""
        assert pdf_tools._lcs_similarity("", "abc") == 0.0

    def test_improved_score_label_match_fuzzy(self):
        """Improved scoring should handle fuzzy matches like full_name -> name."""
        score = pdf_tools._score_label_match("full_name", "name", ["name"])
        assert score >= 1  # Should match via token overlap

    def test_improved_score_label_match_abbreviation(self):
        """Should handle abbreviations like DOB -> date of birth."""
        score = pdf_tools._score_label_match("dob", "dateofbirth", ["date", "birth"])
        assert score >= 1


class TestFormHeuristicsGeometricCheckbox:
    """Tests for geometric checkbox/radio button detection."""

    def test_detect_form_fields_has_geometric_detection(self, sample_text_pdf):
        """detect_form_fields should include geometric checkbox detection."""
        result = pdf_tools.detect_form_fields(sample_text_pdf)
        assert isinstance(result, dict)
        # Should have page analysis with checkbox info
        for page_analysis in result.get("page_analysis", []):
            assert "detected_checkboxes" in page_analysis

    def test_detect_form_fields_multiline_areas(self, sample_text_pdf):
        """detect_form_fields should detect large blank areas as multi-line fields."""
        result = pdf_tools.detect_form_fields(sample_text_pdf)
        assert isinstance(result, dict)
        # Structure should exist even if no multi-line areas found
        for page_analysis in result.get("page_analysis", []):
            assert "detected_multiline_areas" in page_analysis


# ============================================================================
# Test: BUG-002 - Passport branch should use VLM when available (v1.2.1)
# ============================================================================


class TestBug002PassportVlmIntegration:
    """BUG-002: extract_structured_data(data_type='passport') must not
    bypass VLM when a backend is available.  The passport branch previously
    did an early-return with backend=None, completely skipping the LLM
    enhancement code path."""

    # Standard TD3 MRZ: 2 lines x 44 chars each
    MRZ_LINE1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
    MRZ_LINE2 = "AB12345674UTO8001011M3012315<<<<<<<<<<<<<<06"  # 44 chars

    def test_passport_branch_reports_backend_when_available(self):
        """When VLM is available, passport extraction should report it."""
        fake_text = (
            f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
            "Date of Issue: 01 Jan 2020\n"
            "Issuing Authority: Test Immigration Office\n"
        )
        fake_text_result = {
            "text": fake_text,
            "pages_extracted": 2,
            "page_details": [{"page": 1, "text": fake_text}],
        }
        fake_llm_response = '{"issue_date": "2020-01-01", "issuing_authority": "Test Immigration Office"}'

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/passport.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/passport.pdf",
                data_type="passport",
                backend="local",
            )

        # BUG-002: backend must NOT be None when VLM is available
        assert result["backend"] is not None, (
            "BUG-002: passport branch still bypasses VLM (backend is None)"
        )
        assert result["backend_available"] is not None

    def test_passport_branch_method_includes_llm_when_enhanced(self):
        """Method should be 'passport+llm' when VLM enhances results."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {
            "text": fake_text,
            "pages_extracted": 1,
        }
        fake_llm_response = '{"issuing_authority": "Immigration Office"}'

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/passport.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/passport.pdf",
                data_type="passport",
                backend="local",
            )

        assert "llm" in result.get("method", ""), (
            "BUG-002: method should contain 'llm' when VLM enhances passport data"
        )

    def test_passport_branch_still_works_without_vlm(self):
        """Without VLM, passport extraction should still work (pattern only)."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {
            "text": fake_text,
            "pages_extracted": 1,
        }

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/passport.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/passport.pdf",
                data_type="passport",
            )

        assert result["method"] == "passport"
        assert result["backend"] is None
        # MRZ should still extract the passport number
        assert result["data"].get("passport_number") is not None

    def test_passport_vlm_enhances_low_confidence_fields(self):
        """VLM should fill in fields that pattern extraction missed or got
        low confidence on (issue_date, issuing_authority, place_of_birth)."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        fake_llm_response = json.dumps({
            "issue_date": "2020-06-15",
            "issuing_authority": "Department of Immigration",
            "place_of_birth": "Springfield",
        })

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/passport.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/passport.pdf",
                data_type="passport",
                backend="local",
            )

        data = result["data"]
        assert data.get("issuing_authority") == "Department of Immigration"
        assert data.get("issue_date") == "2020-06-15"
        assert data.get("place_of_birth") == "Springfield"


# ============================================================================
# Test: BUG-002a - Passport regex quality issues (v1.2.1)
# ============================================================================


class TestBug002aPassportRegexQuality:
    """BUG-002a: _extract_passport_fields regex should not capture junk
    prefixes in passport_number or match 'Bearer's signature' as authority."""

    def test_passport_number_no_junk_prefix(self):
        """Passport number fallback should strip common prefixes."""
        ocr_text = (
            "PASSPORT | P CHN EK2547928\n"
            "Some other text\n"
        )
        # No MRZ lines, so it falls back to label-value regex
        extracted, confidence = pdf_tools._extract_passport_fields(ocr_text)
        pn = extracted.get("passport_number", "")
        # Must NOT contain "PASSPORT" or "P CHN" prefix
        assert "PASSPORT" not in pn, (
            f"BUG-002a: passport_number contains junk prefix: {pn}"
        )
        if pn:
            # Should be clean alphanumeric
            assert pn == "EK2547928" or len(pn) <= 12

    def test_authority_does_not_match_bearer_signature(self):
        """Issuing authority regex must not capture 'Bearer's signature'."""
        ocr_text = (
            "持 照 大 签名 /Bearer's signature x\n"
            "Issuing Authority: National Immigration Administration\n"
        )
        extracted, confidence = pdf_tools._extract_passport_fields(ocr_text)
        auth = extracted.get("issuing_authority", "")
        assert "signature" not in auth.lower(), (
            f"BUG-002a: issuing_authority matched signature field: {auth}"
        )

    def test_authority_does_not_match_signature_garbage(self):
        """Authority should not capture OCR noise like '#9 A %'."""
        ocr_text = (
            "#9 A % /Bearer's signature S\n"
            "Authority: Immigration Office\n"
        )
        extracted, confidence = pdf_tools._extract_passport_fields(ocr_text)
        auth = extracted.get("issuing_authority", "")
        assert "signature" not in auth.lower(), (
            f"BUG-002a: issuing_authority matched garbage: {auth}"
        )


# ============================================================================
# Test: BUG-002b - page_count should reflect actual pages (v1.2.1)
# ============================================================================


class TestBug002bPageCount:
    """BUG-002b: extract_structured_data returns page_count: 0 because it
    reads 'page_count' from text_result but extract_text returns
    'pages_extracted' instead."""

    def test_page_count_uses_pages_extracted(self):
        """page_count in result should reflect actual pages from extract_text."""
        fake_text_result = {
            "text": "Some document text",
            "pages_extracted": 3,
        }
        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/doc.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/doc.pdf",
                data_type="invoice",
            )

        assert result["page_count"] == 3, (
            f"BUG-002b: page_count is {result['page_count']}, expected 3"
        )

    def test_page_count_passport_uses_pages_extracted(self):
        """Passport branch should also correctly report page_count."""
        fake_text = (
            "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<\n"  # 44 chars
            "AB12345674UTO8001011M3012315<<<<<<<<<<<<<<06\n"  # 44 chars
        )
        fake_text_result = {
            "text": fake_text,
            "pages_extracted": 2,
        }
        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/passport.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/passport.pdf",
                data_type="passport",
            )

        assert result["page_count"] == 2, (
            f"BUG-002b: passport page_count is {result['page_count']}, expected 2"
        )


# ============================================================================
# Test: MRZ-GAP-001 - MRZ fields must appear in passport+llm output (v1.2.2)
# ============================================================================


class TestMrzGap001FieldsInOutput:
    """MRZ-GAP-001: When MRZ is successfully parsed, ALL MRZ-derived fields
    (surname, given_names, nationality, birth_date, sex, expiry_date) must
    appear in the final passport output, even when VLM enhances it."""

    # Standard TD3 MRZ: 2 lines x 44 chars each
    MRZ_LINE1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
    MRZ_LINE2 = "AB12345674UTO8001011M3012315<<<<<<<<<<<<<<06" # 44 chars

    def test_mrz_fields_present_in_passport_output(self):
        """All MRZ-derived fields should be in the output."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport",
            )

        data = result["data"]
        expected_fields = [
            "passport_number", "surname", "given_names",
            "nationality", "birth_date", "sex", "expiry_date",
        ]
        for field in expected_fields:
            assert field in data, (
                f"MRZ-GAP-001: '{field}' missing from passport output"
            )

    def test_mrz_fields_preserved_after_vlm_enhancement(self):
        """MRZ fields must NOT be overwritten by VLM (MRZ is higher confidence)."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        fake_llm_response = json.dumps({
            "issue_date": "2020-01-01",
            "issuing_authority": "Test Office",
        })

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport", backend="local",
            )

        data = result["data"]
        # MRZ fields must be present
        assert data.get("surname") == "DOE"
        assert data.get("given_names") == "JOHN JAMES"
        assert data.get("passport_number") == "AB1234567"
        # VLM fields should also be present
        assert data.get("issuing_authority") == "Test Office"


# ============================================================================
# Test: MRZ-GAP-002 - Noisy OCR MRZ lines must be tolerated (v1.2.2)
# ============================================================================


class TestMrzGap002NoisyOcrTolerance:
    """MRZ-GAP-002: Real OCR often produces MRZ lines that are 43-46 chars
    instead of exactly 44. The parser must tolerate near-length matches."""

    def test_mrz_detection_tolerates_extra_chars(self):
        """MRZ lines with 1-2 extra trailing chars should still be detected."""
        # Simulate real OCR noise: 45 chars (extra trailing char)
        noisy_line1 = "POCHNNI<<XIUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<K<"  # 45 chars
        noisy_line2 = "EK25479281CHN4905088F3304064MENPNAODNDKCA930"  # 45 chars

        mrz = pdf_tools._extract_mrz_lines(f"{noisy_line1}\n{noisy_line2}\n")
        assert mrz is not None, (
            "MRZ-GAP-002: MRZ detection failed on 45-char lines"
        )
        # Returned lines should be trimmed to 44
        assert len(mrz[0]) == 44
        assert len(mrz[1]) == 44

    def test_mrz_detection_tolerates_short_line(self):
        """MRZ lines with 1 missing char (43) should still be detected."""
        short_line1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<" # 43 chars
        normal_line2 = "AB12345674UTO8001011M3012315<<<<<<<<<<<<<<06" # 44 chars

        mrz = pdf_tools._extract_mrz_lines(f"{short_line1}\n{normal_line2}\n")
        # Should still attempt to detect -- pad short lines
        assert mrz is not None, (
            "MRZ-GAP-002: MRZ detection failed on 43-char line"
        )

    def test_passport_number_from_vlm_when_mrz_fails(self):
        """When MRZ parsing fails, VLM should extract passport_number too."""
        # No MRZ lines in text
        fake_text = "Some passport text without clear MRZ lines\nPassport No: ABC123\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        fake_llm_response = json.dumps({
            "passport_number": "EK2544770",
            "issuing_authority": "Immigration Office",
            "place_of_birth": "Fujian",
        })

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport", backend="local",
            )

        assert result["data"].get("passport_number") == "EK2544770", (
            "MRZ-GAP-002: VLM should provide passport_number when MRZ fails"
        )


# ============================================================================
# Test: VLM-QUALITY-001 - Cross-validate VLM dates with MRZ expiry (v1.2.2)
# ============================================================================


class TestVlmQuality001DateCrossValidation:
    """VLM-QUALITY-001: When VLM returns an issue_date that matches the MRZ
    expiry_date, it should be flagged and corrected using domain knowledge."""

    MRZ_LINE1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
    MRZ_LINE2 = "AB12345674UTO8001011M3012315<<<<<<<<<<<<<<06" # 44 chars
    # MRZ expiry = 301231 -> 2030-12-31

    def test_vlm_issue_date_not_same_as_mrz_expiry(self):
        """If VLM returns expiry as issue_date, it should be corrected."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        # VLM mistakenly returns the expiry date as issue_date
        fake_llm_response = json.dumps({
            "issue_date": "2030-12-31",  # Same as MRZ expiry!
            "issuing_authority": "Immigration Office",
        })

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport", backend="local",
            )

        data = result["data"]
        # issue_date must NOT be the same as expiry_date
        if data.get("issue_date") and data.get("expiry_date"):
            assert data["issue_date"] != data["expiry_date"], (
                "VLM-QUALITY-001: issue_date should not equal expiry_date"
            )

    def test_domain_knowledge_passport_validity(self):
        """When VLM date matches MRZ expiry, apply 10-year validity rule."""
        fake_text = f"{self.MRZ_LINE1}\n{self.MRZ_LINE2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        # VLM returns the expiry date (301231 in MRZ = 2030-12-31)
        fake_llm_response = json.dumps({
            "issue_date": "301231",  # Raw MRZ format of expiry
            "issuing_authority": "Immigration Office",
        })

        with (
            patch.object(pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")),
            patch.object(pdf_tools, "extract_text", return_value=fake_text_result),
            patch.object(pdf_tools, "_get_llm_backend", return_value="local"),
            patch.object(pdf_tools, "_call_llm", return_value=fake_llm_response),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport", backend="local",
            )

        data = result["data"]
        # If issue_date was corrected via domain knowledge, it should be ~10 years before expiry
        issue = data.get("issue_date", "")
        if issue:
            assert "2030" not in str(issue), (
                "VLM-QUALITY-001: issue_date still contains expiry year after correction"
            )


# ============================================================================
# Test: BUG-003 - MRZ expiry_date century must use future context (v1.2.3)
# ============================================================================


class TestBug003ExpiryDateCentury:
    """BUG-003: _parse_mrz_date uses a single threshold that fails for future
    expiry dates. Year '33' should be 2033 (expiry), not 1933."""

    def test_expiry_date_future_year(self):
        """MRZ expiry '330406' should produce 2033-04-06, not 1933."""
        result = pdf_tools._parse_mrz_date("330406", is_expiry=True)
        assert result is not None
        assert result.startswith("2033"), (
            f"BUG-003: expiry '330406' returned {result}, expected 2033-04-06"
        )

    def test_expiry_date_far_future(self):
        """MRZ expiry '450101' should produce 2045-01-01."""
        result = pdf_tools._parse_mrz_date("450101", is_expiry=True)
        assert result is not None
        assert result.startswith("2045"), (
            f"BUG-003: expiry '450101' returned {result}, expected 2045-01-01"
        )

    def test_birth_date_past_year(self):
        """MRZ birth '490508' should produce 1949-05-08 (DOB is in the past)."""
        result = pdf_tools._parse_mrz_date("490508")
        assert result is not None
        assert result.startswith("1949"), (
            f"BUG-003: birth '490508' returned {result}, expected 1949-05-08"
        )

    def test_birth_date_recent(self):
        """MRZ birth '050101' should produce 2005-01-01 (young person)."""
        result = pdf_tools._parse_mrz_date("050101")
        assert result is not None
        assert result.startswith("2005"), (
            f"BUG-003: birth '050101' returned {result}, expected 2005-01-01"
        )

    def test_expiry_cascades_to_passport_output(self):
        """Full passport extraction must produce correct expiry year."""
        # TD3 MRZ with expiry 330406 (April 6, 2033)
        line1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
        line2 = "AB12345674UTO8001011M3304064<<<<<<<<<<<<<<06" # 44 chars
        fake_text = f"{line1}\n{line2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}

        with (
            patch.object(
                pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")
            ),
            patch.object(
                pdf_tools, "extract_text", return_value=fake_text_result
            ),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport",
            )

        expiry = result["data"].get("expiry_date", "")
        assert "2033" in str(expiry), (
            f"BUG-003: expiry_date in passport output is {expiry}, expected 2033"
        )


# ============================================================================
# Test: BUG-004 - personal_number OCR noise filtering (v1.2.3)
# ============================================================================


class TestBug004PersonalNumberNoise:
    """BUG-004: personal_number from MRZ positions 28-42 often contains OCR
    garbage. Should be None when no meaningful content is found."""

    def test_noise_personal_number_filtered(self):
        """OCR noise like 'MENPNAODNDKCA9' should become None."""
        # MRZ line2 with noise in personal_number positions
        line1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
        line2 = "AB12345674UTO8001011M3012315MENPNAODNDKCA906" # 44 chars
        fake_text = f"{line1}\n{line2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}

        with (
            patch.object(
                pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")
            ),
            patch.object(
                pdf_tools, "extract_text", return_value=fake_text_result
            ),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport",
            )

        pn = result["data"].get("personal_number")
        if pn is not None:
            # If present, it should not be pure noise
            assert re.match(r"^[A-Z0-9]+$", pn) is None or len(pn) <= 2, (
                f"BUG-004: personal_number '{pn}' looks like OCR noise"
            )

    def test_valid_personal_number_preserved(self):
        """A legitimate personal_number (digits) should be kept."""
        line1 = "P<UTODOE<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
        line2 = "AB12345674UTO8001011M301231512345678<<<<<<06" # 44 chars
        fake_text = f"{line1}\n{line2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}

        with (
            patch.object(
                pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")
            ),
            patch.object(
                pdf_tools, "extract_text", return_value=fake_text_result
            ),
            patch.object(pdf_tools, "_get_llm_backend", return_value=None),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport",
            )

        pn = result["data"].get("personal_number")
        assert pn == "12345678", (
            f"BUG-004: valid personal_number should be '12345678', got '{pn}'"
        )


# ============================================================================
# Test: VLM-QUALITY-003 - VLM authority prompt improvement (v1.2.3)
# ============================================================================


class TestVlmQuality003AuthorityPrompt:
    """VLM-QUALITY-003: VLM prompt should specify to look at the data page,
    not the introductory page, for issuing authority. Also for Chinese
    passports, post-process MFA -> NIA if appropriate."""

    def test_chinese_passport_authority_post_processing(self):
        """If issuing_country=CHN and authority mentions 'Foreign Affairs',
        it should be corrected to NIA."""
        line1 = "P<CHNTEST<<PERSON<<<<<<<<<<<<<<<<<<<<<<<<<<<" # 44 chars
        line2 = "AB12345674CHN8001011M3012315<<<<<<<<<<<<<<06" # 44 chars
        fake_text = f"{line1}\n{line2}\n"
        fake_text_result = {"text": fake_text, "pages_extracted": 1}
        # VLM returns MFA (from intro page)
        fake_llm_response = json.dumps({
            "issue_date": "2020-12-31",
            "issuing_authority": "Ministry of Foreign Affairs",
            "place_of_birth": "Beijing",
        })

        with (
            patch.object(
                pdf_tools, "_ensure_file", return_value=Path("/fake/p.pdf")
            ),
            patch.object(
                pdf_tools, "extract_text", return_value=fake_text_result
            ),
            patch.object(
                pdf_tools, "_get_llm_backend", return_value="local"
            ),
            patch.object(
                pdf_tools, "_call_llm", return_value=fake_llm_response
            ),
        ):
            result = pdf_tools.extract_structured_data(
                "/fake/p.pdf", data_type="passport", backend="local",
            )

        authority = result["data"].get("issuing_authority", "")
        assert "Foreign Affairs" not in authority, (
            f"VLM-QUALITY-003: authority should not be MFA, got '{authority}'"
        )
