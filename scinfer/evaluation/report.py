"""Report generator for benchmark and profiling results.

Produces human-readable reports in multiple formats (Markdown, JSON, HTML)
summarising inference performance and biological fidelity metrics.  Also
exports figure-ready CSV data and LaTeX tables suitable for publication.

Provides :class:`ReportGenerator` as the main entry point.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class ReportGenerator:
    """Generate reports from benchmark and profiling results.

    Parameters
    ----------
    output_dir : str or Path
        Directory where reports will be saved.
    format : str
        Default output format: ``"markdown"``, ``"json"``, or ``"html"``.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./results",
        format: str = "markdown",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()

    # ------------------------------------------------------------------
    # Main report generation
    # ------------------------------------------------------------------

    def generate(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate and save a report.

        Parameters
        ----------
        data : dict
            Report data — typically a ``BenchmarkResult.to_dict()`` or
            ``ProfileResult`` dictionary.
        filename : str, optional
            Output filename (auto-generated if ``None``).

        Returns
        -------
        Path
            Path to the generated report file.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            ext = {"markdown": "md", "json": "json", "html": "html"}.get(self.format, "txt")
            filename = f"report_{ts}.{ext}"

        out_path = self.output_dir / filename

        if self.format == "json":
            self._write_json(data, out_path)
        elif self.format == "html":
            self._write_html(data, out_path)
        else:
            self._write_markdown(data, out_path)

        logger.info(f"Report saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # generate_report (BenchmarkResult-aware)
    # ------------------------------------------------------------------

    def generate_report(
        self,
        benchmark_result: Any,
        format: str = "markdown",
        filename: Optional[str] = None,
    ) -> Path:
        """Generate a complete benchmark report.

        Parameters
        ----------
        benchmark_result : BenchmarkResult
            The benchmark result object.
        format : str
            ``"markdown"``, ``"json"``, or ``"html"``.
        filename : str, optional
            Output filename.

        Returns
        -------
        Path
            Path to the generated report.
        """
        data = benchmark_result.to_dict() if hasattr(benchmark_result, "to_dict") else dict(benchmark_result)

        if format == "json":
            return self._write_json_report(data, filename)
        elif format == "html":
            return self._write_html_report(data, filename)
        else:
            return self._write_markdown_report(data, filename)

    # ------------------------------------------------------------------
    # Comparison report
    # ------------------------------------------------------------------

    def generate_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate a comparison report across multiple models/configs.

        Parameters
        ----------
        results : dict
            Mapping of ``model_name → metric_name → value``.
        filename : str, optional
            Output filename.

        Returns
        -------
        Path
            Path to the generated comparison report.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{ts}.md"

        out_path = self.output_dir / filename
        lines = self._build_comparison_markdown(results)

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Comparison report saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Figure data export
    # ------------------------------------------------------------------

    def export_figure_data(
        self,
        benchmark_result: Any,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """Export data files for plotting.

        Generates one CSV per logical figure:
        - ``fig1_throughput.csv`` — throughput comparison
        - ``fig2_latency.csv`` — latency breakdown
        - ``fig3_memory.csv`` — memory usage
        - ``fig4_biological.csv`` — biological metrics

        Parameters
        ----------
        benchmark_result : BenchmarkResult
            Benchmark results.
        output_dir : str or Path, optional
            Output directory (defaults to ``self.output_dir / "figures"``).

        Returns
        -------
        list of Path
            Paths to generated CSV files.
        """
        import csv

        output_dir = Path(output_dir) if output_dir else self.output_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        data = benchmark_result.to_dict() if hasattr(benchmark_result, "to_dict") else dict(benchmark_result)
        results = data.get("results", {})
        generated: List[Path] = []

        # Categorise metrics into figure groups
        fig_groups: Dict[str, List[str]] = {
            "fig1_throughput": ["throughput_ips", "throughput_std"],
            "fig2_latency": ["latency_median_ms", "latency_std_ms",
                             "latency_p50_ms", "latency_p90_ms",
                             "latency_p95_ms", "latency_p99_ms"],
            "fig3_memory": ["peak_gpu_memory_mb", "cpu_memory_mb", "model_params_mb"],
            "fig4_biological": ["ari", "nmi", "f1_macro", "pcc",
                                "knn_accuracy", "trustworthiness", "continuity"],
        }

        for fig_name, metric_keys in fig_groups.items():
            # Check if any model has any of these metrics
            has_data = False
            for model_metrics in results.values():
                if any(k in model_metrics for k in metric_keys):
                    has_data = True
                    break
            if not has_data:
                continue

            csv_path = output_dir / f"{fig_name}.csv"
            available_keys = [k for k in metric_keys if any(
                k in m for m in results.values()
            )]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model"] + available_keys)
                for model_name, model_metrics in results.items():
                    row = [model_name] + [model_metrics.get(k, "") for k in available_keys]
                    writer.writerow(row)
            generated.append(csv_path)

        logger.info(f"Exported {len(generated)} figure data files to {output_dir}")
        return generated

    # ------------------------------------------------------------------
    # LaTeX table export
    # ------------------------------------------------------------------

    def export_latex_table(
        self,
        benchmark_result: Any,
        table_type: str = "main",
        caption: Optional[str] = None,
    ) -> str:
        """Export a LaTeX-formatted table.

        Parameters
        ----------
        benchmark_result : BenchmarkResult
            Benchmark results.
        table_type : str
            One of ``"main"`` (full results), ``"ablation"`` (ablation study),
            ``"hardware"`` (hardware comparison).
        caption : str, optional
            Table caption.

        Returns
        -------
        str
            LaTeX table string.
        """
        data = benchmark_result.to_dict() if hasattr(benchmark_result, "to_dict") else dict(benchmark_result)
        results = data.get("results", {})

        if not results:
            return "% No data available\n"

        # Collect all metric columns
        all_metrics: List[str] = []
        for model_metrics in results.values():
            for m in model_metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        # Filter metrics by table type
        if table_type == "hardware":
            hw_keys = ["throughput_ips", "latency_median_ms", "peak_gpu_memory_mb"]
            all_metrics = [m for m in all_metrics if m in hw_keys] or all_metrics
        elif table_type == "ablation":
            # Keep all metrics for ablation
            pass

        n_cols = len(all_metrics) + 1
        col_spec = "l" + "r" * (n_cols - 1)

        if caption is None:
            captions = {
                "main": "Benchmark results comparing inference performance.",
                "ablation": "Ablation study results.",
                "hardware": "Hardware comparison results.",
            }
            caption = captions.get(table_type, "Benchmark results.")

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]

        # Header
        header_parts = ["Model"]
        for m in all_metrics:
            header_parts.append(m.replace("_", r"\_"))
        lines.append(" & ".join(header_parts) + r" \\")
        lines.append(r"\midrule")

        # Rows
        for model_name, model_metrics in results.items():
            parts = [model_name]
            for m in all_metrics:
                val = model_metrics.get(m, float("nan"))
                if isinstance(val, float):
                    parts.append(f"{val:.4f}")
                else:
                    parts.append(str(val))
            lines.append(" & ".join(parts) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private: Markdown report
    # ------------------------------------------------------------------

    def _write_markdown(self, data: Dict[str, Any], path: Path) -> None:
        lines = [
            "# scInfer Benchmark Report",
            "",
            f"*Generated: {datetime.utcnow().isoformat()}*",
            "",
        ]
        lines += self._build_comparison_markdown(data.get("results", {}))

        # Metadata section
        meta = data.get("metadata", {})
        if meta:
            lines += ["", "## Environment", ""]
            for k, v in meta.items():
                lines.append(f"- **{k}**: {v}")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_markdown_report(self, data: Dict[str, Any], filename: Optional[str]) -> Path:
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{ts}.md"
        out_path = self.output_dir / filename

        lines = [
            "# scInfer Benchmark Report",
            "",
            f"*Generated: {datetime.utcnow().isoformat()}*",
            "",
        ]

        # Configuration
        config = data.get("config", {})
        if config:
            lines += ["## Configuration", ""]
            for k, v in config.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        # Results table
        results = data.get("results", {})
        lines += self._build_comparison_markdown(results)

        # Metadata
        meta = data.get("metadata", {})
        if meta:
            lines += ["", "## Environment", ""]
            for k, v in meta.items():
                lines.append(f"- **{k}**: {v}")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report saved to {out_path}")
        return out_path

    @staticmethod
    def _build_comparison_markdown(results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Build a Markdown comparison table."""
        if not results:
            return ["*(no results)*"]

        all_metrics: List[str] = []
        for model_metrics in results.values():
            for m in model_metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        lines = [
            "## Results",
            "",
            "| Model | " + " | ".join(all_metrics) + " |",
            "|-------|" + "|".join(["-------"] * len(all_metrics)) + "|",
        ]
        for model_name, model_metrics in results.items():
            parts = [model_name]
            for m in all_metrics:
                val = model_metrics.get(m, float("nan"))
                if isinstance(val, float):
                    parts.append(f"{val:.4f}")
                else:
                    parts.append(str(val))
            lines.append("| " + " | ".join(parts) + " |")

        return lines

    # ------------------------------------------------------------------
    # Private: JSON report
    # ------------------------------------------------------------------

    def _write_json(self, data: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_json_report(self, data: Dict[str, Any], filename: Optional[str]) -> Path:
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{ts}.json"
        out_path = self.output_dir / filename
        self._write_json(data, out_path)
        logger.info(f"JSON report saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Private: HTML report
    # ------------------------------------------------------------------

    def _write_html(self, data: Dict[str, Any], path: Path) -> None:
        html = self._build_html(data)
        path.write_text(html, encoding="utf-8")

    def _write_html_report(self, data: Dict[str, Any], filename: Optional[str]) -> Path:
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{ts}.html"
        out_path = self.output_dir / filename
        self._write_html(data, out_path)
        logger.info(f"HTML report saved to {out_path}")
        return out_path

    @staticmethod
    def _build_html(data: Dict[str, Any]) -> str:
        results = data.get("results", {})
        meta = data.get("metadata", {})
        config = data.get("config", {})

        # Collect metrics
        all_metrics: List[str] = []
        for model_metrics in results.values():
            for m in model_metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        # Build table rows
        table_rows = ""
        for model_name, model_metrics in results.items():
            cells = f"<td>{model_name}</td>"
            for m in all_metrics:
                val = model_metrics.get(m, float("nan"))
                if isinstance(val, float):
                    cells += f"<td>{val:.4f}</td>"
                else:
                    cells += f"<td>{val}</td>"
            table_rows += f"<tr>{cells}</tr>\n"

        header_cells = "".join(f"<th>{m}</th>" for m in all_metrics)

        meta_items = "".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in meta.items())
        config_items = "".join(f"<li><strong>{k}</strong>: {v}</li>" for k, v in config.items())

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>scInfer Benchmark Report</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a2e; }}
h1 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: .5rem; }}
h2 {{ color: #0f3460; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #0f3460; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
tr:hover {{ background: #e9ecef; }}
td:first-child {{ text-align: left; font-weight: 600; }}
.meta {{ color: #666; font-size: .9rem; }}
</style>
</head>
<body>
<h1>scInfer Benchmark Report</h1>
<p class="meta">Generated: {datetime.utcnow().isoformat()}</p>

<h2>Configuration</h2>
<ul>{config_items}</ul>

<h2>Results</h2>
<table>
<thead><tr><th>Model</th>{header_cells}</tr></thead>
<tbody>{table_rows}</tbody>
</table>

<h2>Environment</h2>
<ul>{meta_items}</ul>
</body>
</html>"""
        return html
