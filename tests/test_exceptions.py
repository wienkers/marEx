"""
Tests for MarEx Exception Hierarchy
===================================

Comprehensive tests for the marEx exception system to ensure proper
error handling, exception chaining, and informative error messages.
"""

import pytest

# Import all marEx exceptions
from marEx.exceptions import (
    ConfigurationError,
    CoordinateError,
    DataValidationError,
    DependencyError,
    MarExError,
    ProcessingError,
    TrackingError,
    VisualisationError,
    create_coordinate_error,
    create_data_validation_error,
    create_processing_error,
    wrap_exception,
)


class TestMarExError:
    """Test the base MarExError exception class."""

    def test_basic_creation(self):
        """Test basic exception creation with message only."""
        error = MarExError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details is None
        assert error.suggestions == []
        assert error.error_code is None
        assert error.context == {}

    def test_full_creation(self):
        """Test exception creation with all parameters."""
        error = MarExError(
            message="Data validation failed",
            details="Array shape mismatch",
            suggestions=["Check input dimensions", "Verify data format"],
            error_code="DATA_001",
            context={"expected_shape": (100, 200), "actual_shape": (100, 150)},
        )

        assert error.message == "Data validation failed"
        assert error.details == "Array shape mismatch"
        assert error.suggestions == ["Check input dimensions", "Verify data format"]
        assert error.error_code == "DATA_001"
        assert error.context["expected_shape"] == (100, 200)

    def test_formatted_message(self):
        """Test that the formatted error message includes all components."""
        error = MarExError(
            message="Test error",
            details="Additional information",
            suggestions=["Try this", "Or this"],
            error_code="TEST_001",
            context={"param": "value"},
        )

        full_message = str(error)
        assert "Test error" in full_message
        assert "Details: Additional information" in full_message
        assert "Context: param=value" in full_message
        assert "Suggestions:" in full_message
        assert "- Try this" in full_message
        assert "- Or this" in full_message
        assert "Error Code: TEST_001" in full_message

    def test_add_suggestion(self):
        """Test adding suggestions after creation."""
        error = MarExError("Test error")
        assert len(error.suggestions) == 0

        error.add_suggestion("New suggestion")
        assert len(error.suggestions) == 1
        assert error.suggestions[0] == "New suggestion"

    def test_add_context(self):
        """Test adding context after creation."""
        error = MarExError("Test error")
        assert len(error.context) == 0

        error.add_context("key", "value")
        assert error.context["key"] == "value"


class TestSpecificExceptions:
    """Test specific exception classes inherit properly from MarExError."""

    def test_data_validation_error(self):
        """Test DataValidationError creation and default error code."""
        error = DataValidationError("Invalid data format")
        assert isinstance(error, MarExError)
        assert error.error_code == "DATA_VALIDATION"

    def test_coordinate_error(self):
        """Test CoordinateError creation."""
        error = CoordinateError("Invalid coordinates")
        assert isinstance(error, MarExError)
        assert error.error_code == "COORDINATE_ERROR"

    def test_processing_error(self):
        """Test ProcessingError creation."""
        error = ProcessingError("Computation failed")
        assert isinstance(error, MarExError)
        assert error.error_code == "PROCESSING_ERROR"

    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError("Invalid parameter")
        assert isinstance(error, MarExError)
        assert error.error_code == "CONFIGURATION_ERROR"

    def test_dependency_error(self):
        """Test DependencyError creation."""
        error = DependencyError("Missing package")
        assert isinstance(error, MarExError)
        assert error.error_code == "DEPENDENCY_ERROR"

    def test_tracking_error(self):
        """Test TrackingError creation."""
        error = TrackingError("Tracking failed")
        assert isinstance(error, MarExError)
        assert error.error_code == "TRACKING_ERROR"

    def test_visualisation_error(self):
        """Test VisualisationError creation."""
        error = VisualisationError("Plotting failed")
        assert isinstance(error, MarExError)
        assert error.error_code == "VISUALISATION_ERROR"


class TestConvenienceConstructors:
    """Test convenience constructor functions."""

    def test_create_data_validation_error(self):
        """Test data validation error constructor."""
        data_info = {"dtype": "float32", "shape": (100, 200)}
        error = create_data_validation_error("Invalid data type", data_info=data_info, details="Expected boolean array")

        assert isinstance(error, DataValidationError)
        assert error.message == "Invalid data type"
        assert error.details == "Expected boolean array"
        assert error.context["dtype"] == "float32"
        assert error.context["shape"] == (100, 200)

    def test_create_coordinate_error(self):
        """Test coordinate error constructor."""
        ranges = {"lat": (-90, 90), "lon": (0, 360)}
        error = create_coordinate_error(
            "Invalid coordinate ranges",
            coordinate_ranges=ranges,
            detected_system="degrees",
        )

        assert isinstance(error, CoordinateError)
        assert error.context["coordinate_ranges"] == ranges
        assert error.context["detected_system"] == "degrees"

    def test_create_processing_error(self):
        """Test processing error constructor."""
        comp_info = {"memory_gb": 16, "chunk_size": "1GB"}
        error = create_processing_error("Out of memory", computation_info=comp_info)

        assert isinstance(error, ProcessingError)
        assert error.context["memory_gb"] == 16
        assert error.context["chunk_size"] == "1GB"


class TestExceptionWrapping:
    """Test exception wrapping functionality."""

    def test_wrap_value_error(self):
        """Test wrapping a ValueError."""
        original = ValueError("Original error message")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, DataValidationError)
        assert wrapped.message == "Original error message"
        assert wrapped.__cause__ is original
        assert "Original ValueError" in wrapped.details

    def test_wrap_runtime_error(self):
        """Test wrapping a RuntimeError."""
        original = RuntimeError("Runtime issue")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, ProcessingError)
        assert wrapped.message == "Runtime issue"
        assert wrapped.__cause__ is original

    def test_wrap_with_custom_message(self):
        """Test wrapping with custom message."""
        original = KeyError("missing_key")
        wrapped = wrap_exception(original, "Custom message")

        assert isinstance(wrapped, ConfigurationError)
        assert wrapped.message == "Custom message"
        assert wrapped.__cause__ is original

    def test_wrap_with_custom_type(self):
        """Test wrapping with specific exception type."""
        original = ImportError("module not found")
        wrapped = wrap_exception(original, marex_exception_type=DependencyError)

        assert isinstance(wrapped, DependencyError)
        assert wrapped.__cause__ is original


class TestExceptionWrapper:
    """Test exception wrapping functionality for backward compatibility."""

    def test_wrap_value_error_creates_data_validation_error(self):
        """Test that ValueError gets wrapped as DataValidationError."""
        original = ValueError("Invalid value")
        wrapped = wrap_exception(original)
        assert isinstance(wrapped, DataValidationError)
        assert isinstance(wrapped, MarExError)
        assert wrapped.__cause__ is original

    def test_wrap_runtime_error_creates_processing_error(self):
        """Test that RuntimeError gets wrapped as ProcessingError."""
        original = RuntimeError("Processing failed")
        wrapped = wrap_exception(original)
        assert isinstance(wrapped, ProcessingError)
        assert isinstance(wrapped, MarExError)
        assert wrapped.__cause__ is original


class TestExceptionChaining:
    """Test proper exception chaining behaviour."""

    def test_exception_chaining(self):
        """Test that exceptions can be properly chained."""
        try:
            # Simulate nested exception scenario
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DataValidationError("Validation failed", details="Caused by value error") from e
        except DataValidationError as caught:
            assert caught.__cause__ is not None
            assert isinstance(caught.__cause__, ValueError)
            assert str(caught.__cause__) == "Original error"


class TestErrorMessageQuality:
    """Test that error messages are informative and actionable."""

    def test_suggestions_are_actionable(self):
        """Test that error suggestions provide actionable guidance."""
        error = DataValidationError(
            "Data must be Dask-backed",
            suggestions=[
                "Convert to Dask: data.chunk({'time': 30})",
                "Load with chunking: xr.open_dataset('file.nc').chunk()",
            ],
        )

        message = str(error)
        assert "Convert to Dask" in message
        assert "chunk({'time': 30})" in message
        assert "xr.open_dataset" in message

    def test_context_provides_debugging_info(self):
        """Test that context provides useful debugging information."""
        error = ProcessingError(
            "Memory limit exceeded",
            context={
                "required_memory_gb": 32,
                "available_memory_gb": 16,
                "data_shape": (1000, 2000, 3000),
            },
        )

        message = str(error)
        assert "required_memory_gb=32" in message
        assert "available_memory_gb=16" in message
        assert "data_shape=(1000, 2000, 3000)" in message


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_marex_error(self):
        """Test that all marEx exceptions inherit from MarExError."""
        exceptions = [
            DataValidationError("test"),
            CoordinateError("test"),
            ProcessingError("test"),
            ConfigurationError("test"),
            DependencyError("test"),
            TrackingError("test"),
            VisualisationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, MarExError)
            assert isinstance(exc, Exception)

    def test_marex_error_inherits_from_exception(self):
        """Test that MarExError inherits from Python's Exception."""
        error = MarExError("test")
        assert isinstance(error, Exception)


class TestExceptionCatching:
    """Test that exceptions can be caught at appropriate levels."""

    def test_catch_specific_exception(self):
        """Test catching specific exception types."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Validation failed")

    def test_catch_base_marex_error(self):
        """Test catching any marEx error via base class."""
        with pytest.raises(MarExError):
            raise CoordinateError("Coordinate issue")

    def test_catch_python_exception(self):
        """Test catching marEx errors as standard exceptions."""
        with pytest.raises(ProcessingError):
            raise ProcessingError("Processing issue")


class TestExceptionAttributes:
    """Test that exception attributes are properly maintained."""

    def test_attributes_persist_through_inheritance(self):
        """Test that custom attributes work in subclasses."""
        error = TrackingError(
            "Tracking failed",
            details="Object merge conflict",
            suggestions=["Increase overlap threshold"],
            context={"n_objects": 1000, "merge_limit": 500},
        )

        # All attributes should be accessible
        assert error.message == "Tracking failed"
        assert error.details == "Object merge conflict"
        assert len(error.suggestions) == 1
        assert error.context["n_objects"] == 1000
        assert error.error_code == "TRACKING_ERROR"

    def test_empty_attributes_handled_gracefully(self):
        """Test that empty attributes don't break formatting."""
        error = VisualisationError("Plot failed")

        # Should not contain empty sections
        message = str(error)
        assert "Details:" not in message  # No details provided
        assert "Context:" not in message  # No context provided
        assert "Suggestions:" not in message  # No suggestions provided


if __name__ == "__main__":
    pytest.main([__file__])
