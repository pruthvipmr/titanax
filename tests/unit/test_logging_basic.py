"""Unit tests for titanax.logging.basic module."""

import io
import sys
from unittest.mock import patch


from src.titanax.logging.basic import Basic, CompactBasic


class TestBasic:
    """Test cases for Basic logger."""

    def test_init_default(self):
        """Test Basic logger initialization with defaults."""
        logger = Basic()
        assert logger.name == "titanax"
        assert logger.output == sys.stdout
        assert logger.show_timestamp is True
        assert logger.show_elapsed is True
        assert logger._start_time > 0

    def test_init_custom(self):
        """Test Basic logger initialization with custom parameters."""
        output = io.StringIO()
        logger = Basic(
            name="test", output=output, show_timestamp=False, show_elapsed=False
        )
        assert logger.name == "test"
        assert logger.output == output
        assert logger.show_timestamp is False
        assert logger.show_elapsed is False

    def test_log_scalar_basic(self):
        """Test logging a single scalar value."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=False, show_elapsed=False)

        logger.log_scalar("loss", 0.123456, step=100)
        result = output.getvalue().strip()

        assert "Step    100" in result
        assert "loss=0.123456" in result

    def test_log_scalar_formatting(self):
        """Test scalar value formatting in different ranges."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=False, show_elapsed=False)

        # Small value (scientific notation)
        logger.log_scalar("small", 1e-5, step=1)

        # Large value (scientific notation)
        logger.log_scalar("large", 1e5, step=2)

        # Regular value (fixed point)
        logger.log_scalar("normal", 0.123456, step=3)

        # Integer value
        logger.log_scalar("int", 42, step=4)

        # String value
        logger.log_scalar("str", "test", step=5)

        output_lines = output.getvalue().strip().split("\n")

        assert "1.000e-05" in output_lines[0]
        assert "1.000e+05" in output_lines[1]
        assert "0.123456" in output_lines[2]
        assert "42" in output_lines[3]
        assert "test" in output_lines[4]

    def test_log_dict_basic(self):
        """Test logging a dictionary of metrics."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=False, show_elapsed=False)

        metrics = {"loss": 0.5, "accuracy": 0.95, "epoch": 10}
        logger.log_dict(metrics, step=200)

        result = output.getvalue().strip()
        assert "Step    200" in result
        assert "loss=0.500000" in result
        assert "accuracy=0.950000" in result
        assert "epoch=10" in result

    def test_log_dict_empty(self):
        """Test logging empty metrics dict."""
        output = io.StringIO()
        logger = Basic(output=output)

        logger.log_dict({}, step=1)

        # Should produce no output
        assert output.getvalue() == ""

    def test_timestamp_formatting(self):
        """Test timestamp inclusion."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=True, show_elapsed=False)

        with patch("time.strftime") as mock_strftime:
            mock_strftime.return_value = "2024-01-15 10:30:45"
            logger.log_scalar("loss", 0.1, step=1)

        result = output.getvalue().strip()
        assert "[2024-01-15 10:30:45]" in result
        mock_strftime.assert_called_once()

    def test_elapsed_time_formatting(self):
        """Test elapsed time inclusion."""
        output = io.StringIO()

        # Mock the start time to control elapsed time calculation
        with patch("time.time") as mock_time:
            # Start time
            mock_time.return_value = 1000.0
            logger = Basic(output=output, show_timestamp=False, show_elapsed=True)

            # Current time (5.5 seconds later)
            mock_time.return_value = 1005.5
            logger.log_scalar("loss", 0.1, step=1)

        result = output.getvalue().strip()
        assert "5.50s" in result

    def test_full_format(self):
        """Test complete log format with all components."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=True, show_elapsed=True)

        with (
            patch("time.strftime") as mock_strftime,
            patch.object(logger, "_get_elapsed_time") as mock_elapsed,
        ):
            mock_strftime.return_value = "2024-01-15 10:30:45"
            mock_elapsed.return_value = 123.45

            logger.log_dict({"loss": 0.1, "acc": 0.95}, step=500)

        result = output.getvalue().strip()
        expected_parts = [
            "[2024-01-15 10:30:45]",
            "Step    500",
            "123.45s",
            "loss=0.100000",
            "acc=0.950000",
        ]

        for part in expected_parts:
            assert part in result

    def test_flush(self):
        """Test flush functionality."""
        output = io.StringIO()
        logger = Basic(output=output)

        # Mock flush to verify it's called
        with patch.object(output, "flush") as mock_flush:
            logger.flush()
            mock_flush.assert_called_once()

    def test_close_custom_output(self):
        """Test close functionality with custom output stream."""
        output = io.StringIO()
        logger = Basic(output=output)

        with patch.object(output, "close") as mock_close:
            logger.close()
            mock_close.assert_called_once()

    def test_close_stdout(self):
        """Test close functionality doesn't close stdout."""
        logger = Basic(output=sys.stdout)

        # Should not raise any errors
        logger.close()

    def test_multiple_log_calls(self):
        """Test multiple sequential log calls."""
        output = io.StringIO()
        logger = Basic(output=output, show_timestamp=False, show_elapsed=False)

        logger.log_scalar("loss", 1.0, step=1)
        logger.log_scalar("loss", 0.5, step=2)
        logger.log_scalar("loss", 0.1, step=3)

        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 3
        assert "Step      1" in lines[0]
        assert "Step      2" in lines[1]
        assert "Step      3" in lines[2]


class TestCompactBasic:
    """Test cases for CompactBasic logger."""

    def test_init_default(self):
        """Test CompactBasic logger initialization with defaults."""
        logger = CompactBasic()
        assert logger.name == "titanax"
        assert logger.output == sys.stdout

    def test_init_custom(self):
        """Test CompactBasic logger initialization with custom output."""
        output = io.StringIO()
        logger = CompactBasic(name="test", output=output)
        assert logger.name == "test"
        assert logger.output == output

    def test_log_scalar_compact(self):
        """Test compact scalar logging format."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        logger.log_scalar("loss", 0.123456, step=100)
        result = output.getvalue().strip()

        # Should be format: "100: loss=0.123456"
        assert result == "100: loss=0.123456"

    def test_log_dict_compact(self):
        """Test compact dict logging format."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        metrics = {"loss": 0.5, "acc": 0.95}
        logger.log_dict(metrics, step=200)

        result = output.getvalue().strip()
        assert result.startswith("200: ")
        assert "loss=0.500000" in result
        assert "acc=0.950000" in result

    def test_log_dict_empty_compact(self):
        """Test compact logging with empty metrics."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        logger.log_dict({}, step=1)

        # Should produce no output
        assert output.getvalue() == ""

    def test_value_formatting_compact(self):
        """Test value formatting in compact logger."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        logger.log_scalar("small", 1e-5, step=1)
        logger.log_scalar("large", 1e5, step=2)
        logger.log_scalar("normal", 0.123, step=3)

        lines = output.getvalue().strip().split("\n")
        assert "1.000e-05" in lines[0]
        assert "1.000e+05" in lines[1]
        assert "0.123000" in lines[2]

    def test_flush_compact(self):
        """Test flush functionality in compact logger."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        with patch.object(output, "flush") as mock_flush:
            logger.flush()
            mock_flush.assert_called_once()

    def test_close_compact(self):
        """Test close functionality in compact logger."""
        output = io.StringIO()
        logger = CompactBasic(output=output)

        with patch.object(output, "close") as mock_close:
            logger.close()
            mock_close.assert_called_once()


class TestLoggerIntegration:
    """Test integration between different logger components."""

    def test_basic_with_multilogger(self):
        """Test Basic logger used with MultiLogger."""
        from src.titanax.logging import MultiLogger

        output1 = io.StringIO()
        output2 = io.StringIO()

        logger1 = Basic(output=output1, show_timestamp=False, show_elapsed=False)
        logger2 = CompactBasic(output=output2)

        multi_logger = MultiLogger([logger1, logger2])
        multi_logger.log_scalar("loss", 0.5, step=10)

        # Both loggers should have received the log
        result1 = output1.getvalue().strip()
        result2 = output2.getvalue().strip()

        assert "Step     10" in result1
        assert "loss=0.500000" in result1
        assert "10: loss=0.500000" in result2

    def test_basic_logger_protocol_compliance(self):
        """Test that Basic logger implements Logger protocol correctly."""
        from src.titanax.types import Logger

        logger = Basic()

        # Should implement all required methods
        assert hasattr(logger, "log_scalar")
        assert hasattr(logger, "log_dict")
        assert callable(logger.log_scalar)
        assert callable(logger.log_dict)

        # Should be usable where Logger protocol is expected
        def use_logger(logger_instance: Logger) -> None:
            logger_instance.log_scalar("test", 1.0, 0)

        # Should not raise any type/protocol errors
        use_logger(logger)
