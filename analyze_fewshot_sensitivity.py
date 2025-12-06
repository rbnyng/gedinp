"""
Analyze and visualize few-shot sensitivity sweep results.

This script creates comprehensive visualizations showing:
- Learning curves: Performance vs. number of shots
- Region-specific sensitivity analysis
- Comparison of zero-shot vs. few-shot improvements
- Optimal shot count identification

Usage:
    python analyze_fewshot_sensitivity.py --results_dir ./fewshot_sensitivity_results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REGIONS = {
    'maine': 'Maine (Temperate)',
    'tolima': 'Tolima (Tropical)',
    'hokkaido': 'Hokkaido (Boreal)',
    'sudtirol': 'Sudtirol (Alpine)'
}

REGION_ORDER = ['maine', 'tolima', 'hokkaido', 'sudtirol']


class FewShotSensitivityAnalyzer:
    def __init__(self, results_dir: Path, output_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir or self.results_dir / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        metadata_path = self.results_dir / 'sweep_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded sweep metadata: {self.metadata}")
        else:
            self.metadata = {}
            logger.warning(f"No metadata found at {metadata_path}")

    def load_results(self) -> pd.DataFrame:
        """Load aggregated results."""
        results_file = self.results_dir / 'fewshot_sensitivity_all_results.csv'

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        df = pd.read_csv(results_file)
        logger.info(f"Loaded {len(df)} result rows from {results_file}")

        # Add derived columns
        df['is_diagonal'] = df['train_region'] == df['test_region']
        df['split'] = df['is_diagonal'].map({True: 'In-Distribution', False: 'Out-of-Distribution'})

        return df

    def create_learning_curves(
        self,
        df: pd.DataFrame,
        metric: str = 'log_r2',
        metric_name: str = 'R²',
        higher_is_better: bool = True
    ):
        """Create learning curves showing performance vs. number of shots."""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Filter for ANP few-shot results
        df_fewshot = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'few-shot') &
            (df['seed_id'] == 'mean')
        ].copy()

        if len(df_fewshot) == 0:
            logger.warning(f"No few-shot results found for {metric}")
            plt.close(fig)
            return

        # Left plot: In-distribution
        df_in = df_fewshot[df_fewshot['is_diagonal']].copy()
        self._plot_learning_curve_by_region(
            df_in, metric, metric_name, axes[0],
            title='In-Distribution Performance',
            higher_is_better=higher_is_better
        )

        # Right plot: Out-of-distribution
        df_out = df_fewshot[~df_fewshot['is_diagonal']].copy()
        self._plot_learning_curve_by_region(
            df_out, metric, metric_name, axes[1],
            title='Out-of-Distribution Transfer',
            higher_is_better=higher_is_better
        )

        plt.suptitle(f'Few-Shot Learning Curves: {metric_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / f'learning_curves_{metric}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved learning curves to {output_file}")

    def _plot_learning_curve_by_region(
        self,
        df: pd.DataFrame,
        metric: str,
        metric_name: str,
        ax: plt.Axes,
        title: str,
        higher_is_better: bool = True
    ):
        """Plot learning curves grouped by train region."""

        # Group by train region and n_shots, compute mean
        df_grouped = df.groupby(['train_region', 'n_shots']).agg({
            metric: 'mean',
            f'{metric}_std': 'mean'
        }).reset_index()

        colors = sns.color_palette('husl', n_colors=len(REGION_ORDER))
        region_colors = {region: colors[i] for i, region in enumerate(REGION_ORDER)}

        for train_region in REGION_ORDER:
            df_region = df_grouped[df_grouped['train_region'] == train_region].sort_values('n_shots')

            if len(df_region) == 0:
                continue

            x = df_region['n_shots'].values
            y = df_region[metric].values

            # Plot line with markers
            ax.plot(
                x, y,
                marker='o',
                linewidth=2,
                markersize=8,
                label=REGIONS[train_region],
                color=region_colors[train_region]
            )

            # Add error bars if std is available
            std_col = f'{metric}_std'
            if std_col in df_region.columns and df_region[std_col].notna().any():
                y_std = df_region[std_col].values
                ax.fill_between(
                    x,
                    y - y_std,
                    y + y_std,
                    alpha=0.2,
                    color=region_colors[train_region]
                )

        ax.set_xlabel('Number of Few-Shot Training Tiles', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show actual shot counts
        if len(df_grouped['n_shots'].unique()) > 0:
            ax.set_xticks(sorted(df_grouped['n_shots'].unique()))

    def create_improvement_analysis(self, df: pd.DataFrame):
        """Analyze improvement from zero-shot to few-shot."""

        # Get zero-shot baseline
        df_zero = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'zero-shot') &
            (df['seed_id'] == 'mean')
        ].copy()

        # Get few-shot results
        df_few = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'few-shot') &
            (df['seed_id'] == 'mean')
        ].copy()

        if len(df_zero) == 0 or len(df_few) == 0:
            logger.warning("Missing zero-shot or few-shot results for improvement analysis")
            return

        # Merge on train/test region
        df_merged = df_few.merge(
            df_zero[['train_region', 'test_region', 'log_r2', 'log_rmse']],
            on=['train_region', 'test_region'],
            suffixes=('_few', '_zero')
        )

        # Compute improvements
        df_merged['r2_improvement'] = df_merged['log_r2_few'] - df_merged['log_r2_zero']
        df_merged['rmse_improvement'] = df_merged['log_rmse_zero'] - df_merged['log_rmse_few']  # lower is better

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R² improvement by shot count (OOD only)
        df_ood = df_merged[~df_merged['is_diagonal']].copy()

        # Top-left: R² improvement vs shots
        self._plot_improvement_by_shots(
            df_ood, 'r2_improvement', 'R² Improvement from Zero-Shot',
            axes[0, 0], annotate_best=True
        )

        # Top-right: RMSE improvement vs shots
        self._plot_improvement_by_shots(
            df_ood, 'rmse_improvement', 'RMSE Improvement from Zero-Shot (lower RMSE)',
            axes[0, 1], annotate_best=True
        )

        # Bottom-left: Heatmap of R² improvement by region and shot count
        self._plot_improvement_heatmap(
            df_ood, 'r2_improvement', 'R² Improvement',
            axes[1, 0]
        )

        # Bottom-right: Best shot count by train-test pair
        self._plot_best_shot_counts(
            df_ood, 'r2_improvement',
            axes[1, 1]
        )

        plt.suptitle('Zero-Shot to Few-Shot Improvement Analysis (OOD Transfer Only)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        output_file = self.output_dir / 'improvement_analysis.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved improvement analysis to {output_file}")

        # Save improvement statistics
        self._save_improvement_stats(df_merged)

    def _plot_improvement_by_shots(
        self,
        df: pd.DataFrame,
        metric: str,
        ylabel: str,
        ax: plt.Axes,
        annotate_best: bool = False
    ):
        """Plot improvement metric vs. number of shots."""

        df_grouped = df.groupby(['train_region', 'n_shots'])[metric].mean().reset_index()

        colors = sns.color_palette('husl', n_colors=len(REGION_ORDER))
        region_colors = {region: colors[i] for i, region in enumerate(REGION_ORDER)}

        for train_region in REGION_ORDER:
            df_region = df_grouped[df_grouped['train_region'] == train_region].sort_values('n_shots')

            if len(df_region) == 0:
                continue

            ax.plot(
                df_region['n_shots'],
                df_region[metric],
                marker='o',
                linewidth=2,
                markersize=8,
                label=REGIONS[train_region],
                color=region_colors[train_region]
            )

            # Annotate best value
            if annotate_best and len(df_region) > 0:
                best_idx = df_region[metric].idxmax()
                best_row = df_region.loc[best_idx]
                ax.annotate(
                    f'{best_row["n_shots"]:.0f}',
                    xy=(best_row['n_shots'], best_row[metric]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color=region_colors[train_region],
                    fontweight='bold'
                )

        ax.set_xlabel('Number of Few-Shot Training Tiles', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        if len(df_grouped['n_shots'].unique()) > 0:
            ax.set_xticks(sorted(df_grouped['n_shots'].unique()))

    def _plot_improvement_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        metric_name: str,
        ax: plt.Axes
    ):
        """Create heatmap of improvement by train region and shot count."""

        df_pivot = df.groupby(['train_region', 'n_shots'])[metric].mean().reset_index()
        matrix = df_pivot.pivot(index='train_region', columns='n_shots', values=metric)
        matrix = matrix.reindex(index=REGION_ORDER)

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': metric_name},
            ax=ax,
            linewidths=0.5
        )

        ax.set_xlabel('Number of Few-Shot Tiles', fontsize=11)
        ax.set_ylabel('Train Region', fontsize=11)
        ax.set_title('Average Improvement by Region and Shot Count', fontsize=12, fontweight='bold')
        ax.set_yticklabels([REGIONS[r] for r in REGION_ORDER], rotation=0)

    def _plot_best_shot_counts(
        self,
        df: pd.DataFrame,
        metric: str,
        ax: plt.Axes
    ):
        """Show distribution of best shot counts."""

        # Find best shot count for each train-test pair
        best_shots = df.loc[df.groupby(['train_region', 'test_region'])[metric].idxmax()]

        shot_counts = best_shots['n_shots'].value_counts().sort_index()

        ax.bar(shot_counts.index, shot_counts.values, color='skyblue', edgecolor='black')
        ax.set_xlabel('Number of Few-Shot Tiles', fontsize=11)
        ax.set_ylabel('Count of Train-Test Pairs', fontsize=11)
        ax.set_title('Optimal Shot Count Distribution\n(Across All OOD Region Pairs)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate bars
        for i, (n_shots, count) in enumerate(shot_counts.items()):
            ax.text(n_shots, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')

    def _save_improvement_stats(self, df_merged: pd.DataFrame):
        """Save detailed improvement statistics."""

        stats_rows = []

        for n_shots in sorted(df_merged['n_shots'].unique()):
            df_shots = df_merged[df_merged['n_shots'] == n_shots]

            # Overall stats
            stats_rows.append({
                'n_shots': n_shots,
                'split': 'All',
                'train_region': 'All',
                'mean_r2_improvement': df_shots['r2_improvement'].mean(),
                'median_r2_improvement': df_shots['r2_improvement'].median(),
                'std_r2_improvement': df_shots['r2_improvement'].std(),
                'mean_rmse_improvement': df_shots['rmse_improvement'].mean(),
                'pct_improved_r2': (df_shots['r2_improvement'] > 0).mean() * 100
            })

            # Stats by split
            for split in ['In-Distribution', 'Out-of-Distribution']:
                df_split = df_shots[df_shots['split'] == split]
                if len(df_split) == 0:
                    continue

                stats_rows.append({
                    'n_shots': n_shots,
                    'split': split,
                    'train_region': 'All',
                    'mean_r2_improvement': df_split['r2_improvement'].mean(),
                    'median_r2_improvement': df_split['r2_improvement'].median(),
                    'std_r2_improvement': df_split['r2_improvement'].std(),
                    'mean_rmse_improvement': df_split['rmse_improvement'].mean(),
                    'pct_improved_r2': (df_split['r2_improvement'] > 0).mean() * 100
                })

            # Stats by train region (OOD only)
            df_ood = df_shots[df_shots['split'] == 'Out-of-Distribution']
            for train_region in REGION_ORDER:
                df_region = df_ood[df_ood['train_region'] == train_region]
                if len(df_region) == 0:
                    continue

                stats_rows.append({
                    'n_shots': n_shots,
                    'split': 'Out-of-Distribution',
                    'train_region': train_region,
                    'mean_r2_improvement': df_region['r2_improvement'].mean(),
                    'median_r2_improvement': df_region['r2_improvement'].median(),
                    'std_r2_improvement': df_region['r2_improvement'].std(),
                    'mean_rmse_improvement': df_region['rmse_improvement'].mean(),
                    'pct_improved_r2': (df_region['r2_improvement'] > 0).mean() * 100
                })

        df_stats = pd.DataFrame(stats_rows)
        output_file = self.output_dir / 'improvement_statistics.csv'
        df_stats.to_csv(output_file, index=False)
        logger.info(f"Saved improvement statistics to {output_file}")

        # Print summary
        print("\n" + "="*80)
        print("IMPROVEMENT STATISTICS SUMMARY (OOD Transfer)")
        print("="*80)
        df_ood_all = df_stats[
            (df_stats['split'] == 'Out-of-Distribution') &
            (df_stats['train_region'] == 'All')
        ].sort_values('n_shots')
        print(df_ood_all.to_string(index=False))
        print("="*80 + "\n")

    def create_region_specific_analysis(self, df: pd.DataFrame):
        """Create detailed analysis for each train-test region pair."""

        df_few = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'few-shot') &
            (df['seed_id'] == 'mean') &
            (~df['is_diagonal'])  # OOD only
        ].copy()

        if len(df_few) == 0:
            logger.warning("No OOD few-shot results for region-specific analysis")
            return

        # Create grid of plots: one per train region
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, train_region in enumerate(REGION_ORDER):
            df_train = df_few[df_few['train_region'] == train_region]

            if len(df_train) == 0:
                axes[i].text(0.5, 0.5, f'No data for {REGIONS[train_region]}',
                           ha='center', va='center', fontsize=12)
                axes[i].set_title(REGIONS[train_region], fontsize=14, fontweight='bold')
                continue

            # Plot R² for each test region
            test_regions = [r for r in REGION_ORDER if r != train_region]
            colors = sns.color_palette('Set2', n_colors=len(test_regions))

            for j, test_region in enumerate(test_regions):
                df_pair = df_train[df_train['test_region'] == test_region].sort_values('n_shots')

                if len(df_pair) == 0:
                    continue

                axes[i].plot(
                    df_pair['n_shots'],
                    df_pair['log_r2'],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=f'→ {REGIONS[test_region]}',
                    color=colors[j]
                )

            axes[i].set_xlabel('Number of Few-Shot Tiles', fontsize=10)
            axes[i].set_ylabel('R²', fontsize=10)
            axes[i].set_title(f'Trained on {REGIONS[train_region]}', fontsize=12, fontweight='bold')
            axes[i].legend(loc='best', fontsize=9)
            axes[i].grid(True, alpha=0.3)

            if len(df_train['n_shots'].unique()) > 0:
                axes[i].set_xticks(sorted(df_train['n_shots'].unique()))

        plt.suptitle('Region-Specific Transfer Learning Curves',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        output_file = self.output_dir / 'region_specific_analysis.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved region-specific analysis to {output_file}")

    def create_summary_report(self, df: pd.DataFrame):
        """Create a summary report of findings."""

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("FEW-SHOT SENSITIVITY ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")

        # Metadata
        if self.metadata:
            report_lines.append("Experiment Configuration:")
            report_lines.append(f"  Shot counts tested: {self.metadata.get('shot_counts', 'N/A')}")
            report_lines.append(f"  Few-shot epochs: {self.metadata.get('few_shot_epochs', 'N/A')}")
            report_lines.append(f"  Learning rate: {self.metadata.get('few_shot_lr', 'N/A')}")
            report_lines.append("")

        # Overall performance
        df_few = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'few-shot') &
            (df['seed_id'] == 'mean') &
            (~df['is_diagonal'])
        ]

        if len(df_few) > 0:
            report_lines.append("Overall Results (OOD Transfer):")

            for n_shots in sorted(df_few['n_shots'].unique()):
                df_shots = df_few[df_few['n_shots'] == n_shots]
                mean_r2 = df_shots['log_r2'].mean()
                mean_rmse = df_shots['log_rmse'].mean()

                report_lines.append(f"  {n_shots:2d} shots: R² = {mean_r2:.3f}, RMSE = {mean_rmse:.3f}")

            report_lines.append("")

            # Best shot count
            shot_performance = df_few.groupby('n_shots')['log_r2'].mean()
            best_n_shots = shot_performance.idxmax()
            best_r2 = shot_performance.max()

            report_lines.append(f"Best average performance: {best_n_shots} shots (R² = {best_r2:.3f})")
            report_lines.append("")

        # Zero-shot comparison
        df_zero = df[
            (df['model_type'] == 'anp') &
            (df['transfer_type'] == 'zero-shot') &
            (df['seed_id'] == 'mean') &
            (~df['is_diagonal'])
        ]

        if len(df_zero) > 0 and len(df_few) > 0:
            zero_shot_r2 = df_zero['log_r2'].mean()
            report_lines.append(f"Zero-shot baseline: R² = {zero_shot_r2:.3f}")

            for n_shots in sorted(df_few['n_shots'].unique()):
                df_shots = df_few[df_few['n_shots'] == n_shots]
                few_shot_r2 = df_shots['log_r2'].mean()
                improvement = few_shot_r2 - zero_shot_r2
                improvement_pct = (improvement / abs(zero_shot_r2)) * 100 if zero_shot_r2 != 0 else 0

                report_lines.append(f"  {n_shots:2d} shots: +{improvement:.3f} ({improvement_pct:+.1f}%)")

            report_lines.append("")

        report_lines.append("="*80)

        # Save report
        report_file = self.output_dir / 'summary_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Saved summary report to {report_file}")

        # Print to console
        print('\n'.join(report_lines))

    def run_full_analysis(self):
        """Run complete analysis pipeline."""

        logger.info("\n" + "="*80)
        logger.info("STARTING FEW-SHOT SENSITIVITY ANALYSIS")
        logger.info("="*80 + "\n")

        # Load results
        df = self.load_results()

        # Create all visualizations
        logger.info("\nCreating learning curves...")
        self.create_learning_curves(df, metric='log_r2', metric_name='R²', higher_is_better=True)
        self.create_learning_curves(df, metric='log_rmse', metric_name='RMSE', higher_is_better=False)

        if 'coverage_1sigma' in df.columns:
            self.create_learning_curves(df, metric='coverage_1sigma', metric_name='1-Sigma Coverage', higher_is_better=True)

        logger.info("\nCreating improvement analysis...")
        self.create_improvement_analysis(df)

        logger.info("\nCreating region-specific analysis...")
        self.create_region_specific_analysis(df)

        logger.info("\nGenerating summary report...")
        self.create_summary_report(df)

        logger.info("\n" + "="*80)
        logger.info(f"ANALYSIS COMPLETE - All outputs saved to: {self.output_dir}")
        logger.info("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze few-shot sensitivity sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing fewshot_sensitivity_all_results.csv'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for analysis (default: {results_dir}/analysis)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = FewShotSensitivityAnalyzer(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
