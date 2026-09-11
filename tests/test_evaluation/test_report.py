"""Tests for ReportGenerator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scinfer.evaluation.report import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def report_generator(tmp_path):
    """ReportGenerator instance with temporary output directory."""
    return ReportGenerator(output_dir=tmp_path / "reports", format="markdown")


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark result dictionary."""
    return {
        "results": {
            "model_a": {
                "throughput_ips": 1000.0,
                "latency_median_ms": 10.0,
                "peak_gpu_memory_mb": 2048.0,
            },
            "model_b": {
                "throughput_ips": 2000.0,
                "latency_median_ms": 5.0,
                "peak_gpu_memory_mb": 1024.0,
            },
        },
        "config": {"warmup_runs": 5, "benchmark_runs": 20},
        "metadata": {"platform": "test", "torch_version": "2.0.0"},
    }


@pytest.fixture
def mock_benchmark_result(sample_benchmark_data):
    """Mock BenchmarkResult with to_dict() method."""
    result = MagicMock()
    result.to_dict.return_value = sample_benchmark_data
    return result


# ---------------------------------------------------------------------------
# Tests — generate() Markdown / JSON / HTML
# ---------------------------------------------------------------------------


class TestReportGeneratorMarkdown:
    """test_report_generator_markdown: Markdown format output"""

    def test_generate_markdown(self, report_generator, sample_benchmark_data):
        report_generator.format = "markdown"
        out_path = report_generator.generate(sample_benchmark_data, "test_report.md")
        assert out_path.exists()
        assert out_path.suffix == ".md"
        content = out_path.read_text()
        assert "scInfer Benchmark Report" in content
        assert "model_a" in content
        assert "model_b" in content


class TestReportGeneratorJSON:
    """test_report_generator_json: JSON format output"""

    def test_generate_json(self, report_generator, sample_benchmark_data):
        report_generator.format = "json"
        out_path = report_generator.generate(sample_benchmark_data, "test_report.json")
        assert out_path.exists()
        assert out_path.suffix == ".json"
        with open(out_path) as f:
            data = json.load(f)
        assert "results" in data
        assert "model_a" in data["results"]


class TestReportGeneratorHTML:
    """test_report_generator_html: HTML format output"""

    def test_generate_html(self, report_generator, sample_benchmark_data):
        report_generator.format = "html"
        out_path = report_generator.generate(sample_benchmark_data, "test_report.html")
        assert out_path.exists()
        assert out_path.suffix == ".html"
        content = out_path.read_text()
        assert "<html" in content
        assert "scInfer Benchmark Report" in content
        assert "model_a" in content


# ---------------------------------------------------------------------------
# Tests — generate_comparison()
# ---------------------------------------------------------------------------


class TestReportGeneratorComparison:
    """test_report_generator_comparison: comparison report"""

    def test_generate_comparison(self, report_generator):
        results = {
            "model_a": {"throughput": 100.0, "latency": 10.0},
            "model_b": {"throughput": 200.0, "latency": 5.0},
        }
        out_path = report_generator.generate_comparison(results, "comparison.md")
        assert out_path.exists()
        content = out_path.read_text()
        assert "model_a" in content
        assert "model_b" in content
        assert "throughput" in content


# ---------------------------------------------------------------------------
# Tests — export_figure_data()
# ---------------------------------------------------------------------------


class TestReportGeneratorFigureData:
    """test_report_generator_figure_data: CSV export for figures"""

    def test_export_figure_data(self, report_generator, mock_benchmark_result):
        generated = report_generator.export_figure_data(mock_benchmark_result)
        assert isinstance(generated, list)
        # Should generate at least one CSV file
        assert len(generated) > 0
        for csv_path in generated:
            assert csv_path.exists()
            assert csv_path.suffix == ".csv"

    def test_export_figure_data_custom_dir(self, report_generator, mock_benchmark_result, tmp_path):
        custom_dir = tmp_path / "custom_figures"
        generated = report_generator.export_figure_data(mock_benchmark_result, output_dir=custom_dir)
        for csv_path in generated:
            assert csv_path.parent == custom_dir


# ---------------------------------------------------------------------------
# Tests — export_latex_table()
# ---------------------------------------------------------------------------


class TestReportGeneratorLatexTable:
    """test_report_generator_latex_table: LaTeX table export"""

    def test_export_latex_table(self, report_generator, mock_benchmark_result):
        latex = report_generator.export_latex_table(mock_benchmark_result)
        assert isinstance(latex, str)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "model_a" in latex
        assert "model_b" in latex

    def test_export_latex_table_empty(self, report_generator):
        empty_result = MagicMock()
        empty_result.to_dict.return_value = {"results": {}}
        latex = report_generator.export_latex_table(empty_result)
        assert "No data available" in latex

    def test_export_latex_table_custom_caption(self, report_generator, mock_benchmark_result):
        latex = report_generator.export_latex_table(
            mock_benchmark_result, caption="Custom caption"
        )
        assert "Custom caption" in latex
