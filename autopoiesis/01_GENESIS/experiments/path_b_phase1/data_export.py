"""
GENESIS Phase 4: Data Export Utilities

Export experiment results to various formats:
- CSV (for spreadsheets)
- JSON (for web/APIs)
- HDF5 (for large datasets)
- LaTeX tables (for papers)
- Markdown (for reports)

Usage:
    from data_export import DataExporter

    exporter = DataExporter('results/long_term/20260104_120000')
    exporter.to_csv('output.csv')
    exporter.to_latex('table.tex')
    exporter.to_markdown('report.md')
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class DataExporter:
    """
    Export experiment data to multiple formats
    """

    def __init__(self, experiment_path: str):
        """
        Initialize exporter

        Args:
            experiment_path: Path to experiment results
        """
        from experiment_utils import load_experiment

        self.experiment_path = Path(experiment_path)
        self.experiment = load_experiment(str(experiment_path))
        self.summary = self.experiment.get_summary()

    def to_csv(self, output_path: str, include_time_series: bool = True):
        """
        Export to CSV format

        Args:
            output_path: Output CSV file path
            include_time_series: Include full time series data
        """
        output_file = Path(output_path)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Summary statistics
            writer.writerow(['EXPERIMENT SUMMARY'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow([])

            writer.writerow(['Timestamp', self.summary['timestamp']])
            writer.writerow(['Total Steps', self.summary['n_steps']])
            writer.writerow([])

            writer.writerow(['Initial Population', self.summary['initial_population']])
            writer.writerow(['Final Population', self.summary['final_population']])
            writer.writerow(['Population Growth (%)', f"{self.summary['population_growth']:.2f}"])
            writer.writerow([])

            writer.writerow(['Initial Coherence', f"{self.summary['initial_coherence']:.4f}"])
            writer.writerow(['Final Coherence', f"{self.summary['final_coherence']:.4f}"])
            writer.writerow(['Coherence Improvement', f"{self.summary['coherence_improvement']:.4f}"])
            writer.writerow([])

            # Phase 4B
            if self.summary['novelty_search']:
                writer.writerow(['Unique Behaviors', self.summary['novelty_search']['unique_behaviors']])
                writer.writerow(['Archive Size', self.summary['novelty_search']['archive_size']])
                writer.writerow([])

            if self.summary['map_elites']:
                writer.writerow(['MAP-Elites Coverage (%)', f"{self.summary['map_elites']['coverage']:.2f}"])
                writer.writerow(['MAP-Elites Size', self.summary['map_elites']['size']])
                writer.writerow([])

            # Phase 4C
            if self.summary['communication']:
                writer.writerow(['Total Messages', self.summary['communication']['total_messages']])
                if 'signal_diversity' in self.summary['communication']:
                    writer.writerow(['Signal Diversity', f"{self.summary['communication']['signal_diversity']:.4f}"])
                writer.writerow([])

            # Time series data
            if include_time_series and self.experiment.statistics:
                writer.writerow([])
                writer.writerow(['TIME SERIES DATA'])
                writer.writerow([])

                # Headers
                headers = ['Step', 'Population', 'Coherence']
                writer.writerow(headers)

                # Data
                for i, stats in enumerate(self.experiment.statistics):
                    row = [
                        i,
                        stats.get('population_size', 0),
                        f"{stats.get('avg_coherence', 0):.4f}"
                    ]
                    writer.writerow(row)

        print(f"✓ Exported to CSV: {output_file}")

    def to_json(self, output_path: str, pretty: bool = True):
        """
        Export to JSON format

        Args:
            output_path: Output JSON file path
            pretty: Pretty-print JSON
        """
        output_file = Path(output_path)

        data = {
            'experiment': {
                'timestamp': self.summary['timestamp'],
                'path': str(self.experiment_path),
                'total_steps': self.summary['n_steps']
            },
            'summary': self.summary,
            'time_series': self.experiment.statistics if hasattr(self.experiment, 'statistics') else []
        }

        with open(output_file, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

        print(f"✓ Exported to JSON: {output_file}")

    def to_latex(self, output_path: str, caption: str = "Experiment Results"):
        """
        Export to LaTeX table format

        Args:
            output_path: Output .tex file path
            caption: Table caption
        """
        output_file = Path(output_path)

        with open(output_file, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\midrule\n")

            # Write metrics
            metrics = [
                ("Total Steps", f"{self.summary['n_steps']:,}"),
                ("Final Population", f"{self.summary['final_population']}"),
                ("Population Growth", f"{self.summary['population_growth']:+.1f}\\%"),
                ("Final Coherence", f"{self.summary['final_coherence']:.3f}"),
                ("Coherence Improvement", f"{self.summary['coherence_improvement']:+.3f}"),
            ]

            if self.summary['novelty_search']:
                metrics.append(("Unique Behaviors", f"{self.summary['novelty_search']['unique_behaviors']:,}"))

            if self.summary['map_elites']:
                metrics.append(("MAP-Elites Coverage", f"{self.summary['map_elites']['coverage']:.1f}\\%"))

            if self.summary['communication']:
                metrics.append(("Total Messages", f"{self.summary['communication']['total_messages']:,}"))

            for metric, value in metrics:
                f.write(f"{metric} & {value} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"✓ Exported to LaTeX: {output_file}")

    def to_markdown(self, output_path: str):
        """
        Export to Markdown format

        Args:
            output_path: Output .md file path
        """
        output_file = Path(output_path)

        with open(output_file, 'w') as f:
            f.write(f"# Experiment Results: {self.summary['timestamp']}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Total Steps**: {self.summary['n_steps']:,}\n")
            f.write(f"- **Duration**: {self.summary['timestamp']}\n\n")

            f.write("## Population Dynamics\n\n")
            f.write(f"- Initial: {self.summary['initial_population']}\n")
            f.write(f"- Final: {self.summary['final_population']}\n")
            f.write(f"- Growth: {self.summary['population_growth']:+.1f}%\n\n")

            f.write("## Coherence Evolution\n\n")
            f.write(f"- Initial: {self.summary['initial_coherence']:.3f}\n")
            f.write(f"- Final: {self.summary['final_coherence']:.3f}\n")
            f.write(f"- Improvement: {self.summary['coherence_improvement']:+.3f}\n\n")

            if self.summary['novelty_search']:
                f.write("## Novelty Search\n\n")
                f.write(f"- Unique Behaviors: {self.summary['novelty_search']['unique_behaviors']:,}\n")
                f.write(f"- Archive Size: {self.summary['novelty_search']['archive_size']:,}\n\n")

            if self.summary['map_elites']:
                f.write("## MAP-Elites\n\n")
                f.write(f"- Coverage: {self.summary['map_elites']['coverage']:.1f}%\n")
                f.write(f"- Archive Size: {self.summary['map_elites']['size']:,}\n\n")

            if self.summary['communication']:
                f.write("## Communication\n\n")
                f.write(f"- Total Messages: {self.summary['communication']['total_messages']:,}\n")
                if 'signal_diversity' in self.summary['communication']:
                    f.write(f"- Signal Diversity: {self.summary['communication']['signal_diversity']:.3f}\n")
                f.write("\n")

            # Metrics table
            f.write("## Detailed Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Steps | {self.summary['n_steps']:,} |\n")
            f.write(f"| Population Growth | {self.summary['population_growth']:+.1f}% |\n")
            f.write(f"| Coherence Improvement | {self.summary['coherence_improvement']:+.3f} |\n")

        print(f"✓ Exported to Markdown: {output_file}")

    def to_hdf5(self, output_path: str):
        """
        Export to HDF5 format (requires h5py)

        Args:
            output_path: Output .h5 file path
        """
        try:
            import h5py
        except ImportError:
            print("⚠️  h5py not installed. Install with: pip install h5py")
            return

        output_file = Path(output_path)

        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['timestamp'] = self.summary['timestamp']
            f.attrs['total_steps'] = self.summary['n_steps']

            # Summary
            summary_group = f.create_group('summary')
            summary_group.attrs['initial_population'] = self.summary['initial_population']
            summary_group.attrs['final_population'] = self.summary['final_population']
            summary_group.attrs['population_growth'] = self.summary['population_growth']
            summary_group.attrs['initial_coherence'] = self.summary['initial_coherence']
            summary_group.attrs['final_coherence'] = self.summary['final_coherence']
            summary_group.attrs['coherence_improvement'] = self.summary['coherence_improvement']

            # Time series
            if self.experiment.statistics:
                ts_group = f.create_group('time_series')

                population = np.array([s['population_size'] for s in self.experiment.statistics])
                coherence = np.array([s['avg_coherence'] for s in self.experiment.statistics])

                ts_group.create_dataset('population', data=population)
                ts_group.create_dataset('coherence', data=coherence)

        print(f"✓ Exported to HDF5: {output_file}")


class BatchExporter:
    """
    Export multiple experiments at once
    """

    def __init__(self, experiment_paths: List[str]):
        """
        Initialize batch exporter

        Args:
            experiment_paths: List of experiment paths
        """
        self.experiments = [DataExporter(path) for path in experiment_paths]

    def to_comparison_csv(self, output_path: str):
        """
        Export comparison table to CSV

        Args:
            output_path: Output CSV path
        """
        output_file = Path(output_path)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Headers
            headers = [
                'Experiment', 'Steps', 'Final Population', 'Population Growth (%)',
                'Final Coherence', 'Coherence Improvement'
            ]
            writer.writerow(headers)

            # Data
            for exp in self.experiments:
                summary = exp.summary
                row = [
                    summary['timestamp'],
                    summary['n_steps'],
                    summary['final_population'],
                    f"{summary['population_growth']:.2f}",
                    f"{summary['final_coherence']:.4f}",
                    f"{summary['coherence_improvement']:.4f}"
                ]
                writer.writerow(row)

        print(f"✓ Exported comparison to CSV: {output_file}")

    def to_comparison_latex(self, output_path: str):
        """
        Export comparison table to LaTeX

        Args:
            output_path: Output .tex path
        """
        output_file = Path(output_path)

        with open(output_file, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Experiment Comparison}\n")
            f.write("\\label{tab:comparison}\n")
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\toprule\n")
            f.write("Experiment & Steps & Pop. Growth & Coherence & Improvement \\\\\n")
            f.write("\\midrule\n")

            for exp in self.experiments:
                summary = exp.summary
                f.write(f"{summary['timestamp']} & ")
                f.write(f"{summary['n_steps']:,} & ")
                f.write(f"{summary['population_growth']:+.1f}\\% & ")
                f.write(f"{summary['final_coherence']:.3f} & ")
                f.write(f"{summary['coherence_improvement']:+.3f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"✓ Exported comparison to LaTeX: {output_file}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENESIS Phase 4: Data Export Utilities Demo")
    print("="*70)
    print()

    # Demo message
    print("Usage example:")
    print()
    print("from data_export import DataExporter")
    print()
    print("# Single experiment")
    print("exporter = DataExporter('results/long_term/20260104_120000')")
    print("exporter.to_csv('output.csv')")
    print("exporter.to_json('output.json')")
    print("exporter.to_latex('table.tex')")
    print("exporter.to_markdown('report.md')")
    print()
    print("# Multiple experiments")
    print("from data_export import BatchExporter")
    print()
    print("batch = BatchExporter(['exp1', 'exp2', 'exp3'])")
    print("batch.to_comparison_csv('comparison.csv')")
    print("batch.to_comparison_latex('comparison.tex')")
    print()
    print("✓ Data export utilities ready")
