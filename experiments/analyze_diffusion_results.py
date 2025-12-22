"""
MLflow-Based Analysis and Reporting for Diffusion Studies
Generates comprehensive reports, statistical analyses, and visualizations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from scipy import stats
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available")


class DiffusionStudyAnalyzer:
    """Analyzes diffusion study results from MLflow"""
    
    def __init__(self, experiment_name: str = "HyperGRAND_Diffusion_Studies", tracking_uri: str = None):
        """Initialize analyzer"""
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for analysis")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Get experiment
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        self.experiment_id = experiment.experiment_id
    
    def get_parent_runs(self, study_dimension: str = None) -> List:
        """Get all parent runs, optionally filtered by study dimension"""
        filter_string = "tags.run_type = 'parent'"
        if study_dimension:
            filter_string += f" and tags.study_dimension = '{study_dimension}'"
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )
        
        return runs
    
    def get_child_runs(self, parent_run_id: str) -> List:
        """Get all child runs for a parent run"""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.parent_run_id = '{parent_run_id}'"
        )
        
        return runs
    
    def generate_study_summary(self, study_dimension: str) -> pd.DataFrame:
        """
        Generate summary table for a study dimension
        
        Returns:
            DataFrame with mean ± std for each configuration variant
        """
        parent_runs = self.get_parent_runs(study_dimension)
        
        if not parent_runs:
            print(f"No runs found for study dimension: {study_dimension}")
            return pd.DataFrame()
        
        summary_data = []
        
        for parent_run in parent_runs:
            # Get aggregated metrics from parent run
            metrics = parent_run.data.metrics
            params = parent_run.data.params
            
            row = {
                'run_id': parent_run.info.run_id,
                'study_name': params.get('study_name', 'unknown'),
            }
            
            # Extract mean and std for key metrics
            for metric_base in ['final_test_accuracy', 'final_test_loss', 'test_nmi', 'test_ari']:
                if f'{metric_base}_mean' in metrics:
                    row[f'{metric_base}_mean'] = metrics[f'{metric_base}_mean']
                    row[f'{metric_base}_std'] = metrics.get(f'{metric_base}_std', 0)
                    row[f'{metric_base}_ci_lower'] = metrics.get(f'{metric_base}_ci_lower', 0)
                    row[f'{metric_base}_ci_upper'] = metrics.get(f'{metric_base}_ci_upper', 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def generate_comparison_matrix(self, study_dimension: str, metric: str = 'final_test_accuracy') -> pd.DataFrame:
        """
        Generate comparison matrix (config_variant × dataset)
        
        Args:
            study_dimension: Study dimension to analyze
            metric: Metric to compare
        
        Returns:
            DataFrame with config_variants as rows, datasets as columns
        """
        parent_runs = self.get_parent_runs(study_dimension)
        
        # Collect data from all child runs
        data = []
        
        for parent_run in parent_runs:
            child_runs = self.get_child_runs(parent_run.info.run_id)
            
            for child_run in child_runs:
                if metric in child_run.data.metrics:
                    data.append({
                        'config_variant': child_run.data.tags.get('config_variant', 'unknown'),
                        'dataset': child_run.data.params.get('dataset', 'unknown'),
                        metric: child_run.data.metrics[metric]
                    })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Pivot to matrix form
        matrix = df.pivot_table(
            values=metric,
            index='config_variant',
            columns='dataset',
            aggfunc='mean'
        )
        
        return matrix
    
    def generate_hypothesis_validation_report(self) -> Dict[str, Any]:
        """
        Generate report validating paper hypotheses
        
        Hypotheses:
        H1: Clustering excellence (>90% accuracy/NMI)
        H2: Classification struggle (<75% accuracy)
        H3: Integration scheme impact (explicit vs implicit)
        """
        report = {}
        
        # Get all child runs grouped by task type
        all_runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.run_type = 'child'"
        )
        
        # Group by task type
        by_task = {'classification': [], 'clustering': [], 'partitioning': []}
        
        for run in all_runs:
            task_type = run.data.tags.get('task_type', 'unknown')
            if task_type in by_task:
                by_task[task_type].append(run)
        
        # H1: Clustering excellence
        if by_task['clustering']:
            nmi_values = [r.data.metrics.get('test_nmi', 0) for r in by_task['clustering'] 
                         if 'test_nmi' in r.data.metrics]
            if nmi_values:
                mean_nmi = np.mean(nmi_values)
                report['h1_clustering_excellence'] = {
                    'hypothesis': 'HyperGRAND excels at clustering (NMI > 0.9)',
                    'mean_nmi': mean_nmi,
                    'std_nmi': np.std(nmi_values),
                    'validated': mean_nmi > 0.9,
                    'num_experiments': len(nmi_values)
                }
        
        # H2: Classification struggle
        if by_task['classification']:
            acc_values = [r.data.metrics.get('final_test_accuracy', 0) for r in by_task['classification']
                         if 'final_test_accuracy' in r.data.metrics]
            if acc_values:
                mean_acc = np.mean(acc_values)
                report['h2_classification_struggle'] = {
                    'hypothesis': 'HyperGRAND struggles at classification (Acc < 0.75)',
                    'mean_accuracy': mean_acc,
                    'std_accuracy': np.std(acc_values),
                    'validated': mean_acc < 0.75,
                    'num_experiments': len(acc_values)
                }
        
        # H3: Integration scheme impact
        explicit_runs = [r for r in all_runs if r.data.params.get('integration_scheme') == 'explicit']
        implicit_runs = [r for r in all_runs if r.data.params.get('integration_scheme') == 'implicit']
        
        if explicit_runs and implicit_runs:
            explicit_acc = [r.data.metrics.get('final_test_accuracy', 0) for r in explicit_runs
                          if 'final_test_accuracy' in r.data.metrics]
            implicit_acc = [r.data.metrics.get('final_test_accuracy', 0) for r in implicit_runs
                          if 'final_test_accuracy' in r.data.metrics]
            
            if explicit_acc and implicit_acc:
                t_stat, p_value = stats.ttest_ind(explicit_acc, implicit_acc)
                report['h3_integration_scheme_impact'] = {
                    'hypothesis': 'Integration scheme significantly affects performance',
                    'explicit_mean': np.mean(explicit_acc),
                    'implicit_mean': np.mean(implicit_acc),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'num_explicit': len(explicit_acc),
                    'num_implicit': len(implicit_acc)
                }
        
        return report
    
    def generate_latex_table(self, study_dimension: str, metric: str = 'final_test_accuracy') -> str:
        """
        Generate LaTeX table for paper inclusion
        
        Args:
            study_dimension: Study dimension to tabulate
            metric: Metric to include
        
        Returns:
            LaTeX table string
        """
        df = self.generate_study_summary(study_dimension)
        
        if df.empty:
            return "% No data available"
        
        # Format table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{" + f"Results for {study_dimension.replace('_', ' ').title()} Study" + "}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\hline\n"
        latex += "Configuration & Mean $\\pm$ Std & 95\\% CI \\\\\n"
        latex += "\\hline\n"
        
        metric_base = metric
        for _, row in df.iterrows():
            if f'{metric_base}_mean' in row:
                config = row['study_name'].replace('_', '\\_')
                mean = row[f'{metric_base}_mean']
                std = row[f'{metric_base}_std']
                ci_lower = row[f'{metric_base}_ci_lower']
                ci_upper = row[f'{metric_base}_ci_upper']
                
                latex += f"{config} & ${mean:.4f} \\pm {std:.4f}$ & $[{ci_lower:.4f}, {ci_upper:.4f}]$ \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def plot_comparison_heatmap(self, study_dimension: str, metric: str = 'final_test_accuracy', 
                               output_path: str = None):
        """
        Generate heatmap comparing configurations across datasets
        
        Args:
            study_dimension: Study dimension
            metric: Metric to visualize
            output_path: Path to save plot (if None, displays plot)
        """
        matrix = self.generate_comparison_matrix(study_dimension, metric)
        
        if matrix.empty:
            print(f"No data available for {study_dimension}")
            return
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=matrix.mean().mean())
        plt.title(f'{study_dimension.replace("_", " ").title()} Comparison: {metric}')
        plt.xlabel('Dataset')
        plt.ylabel('Configuration Variant')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_curves(self, run_id: str, output_path: str = None):
        """
        Plot learning curves for a specific run
        
        Args:
            run_id: MLflow run ID
            output_path: Path to save plot
        """
        run = self.client.get_run(run_id)
        
        # Get metrics history
        train_loss = self.client.get_metric_history(run_id, 'train_loss')
        val_loss = self.client.get_metric_history(run_id, 'val_loss')
        val_metric = self.client.get_metric_history(run_id, 'val_metric')
        
        if not train_loss:
            print(f"No learning curve data for run {run_id}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot([m.step for m in train_loss], [m.value for m in train_loss], label='Train Loss')
        ax1.plot([m.step for m in val_loss], [m.value for m in val_loss], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation metric
        ax2.plot([m.step for m in val_metric], [m.value for m in val_metric], label='Val Metric')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric')
        ax2.set_title('Validation Metric')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """CLI for analyzing diffusion study results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze HyperGRAND diffusion study results')
    parser.add_argument('--experiment', type=str, default='HyperGRAND_Diffusion_Studies',
                       help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None,
                       help='MLflow tracking URI')
    parser.add_argument('--study-dimension', type=str, default='integration_scheme',
                       help='Study dimension to analyze')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--generate-latex', action='store_true',
                       help='Generate LaTeX tables')
    parser.add_argument('--plot-heatmap', action='store_true',
                       help='Generate comparison heatmaps')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Output directory for reports and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = DiffusionStudyAnalyzer(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri
    )
    
    # Generate reports
    if args.generate_report:
        print("\n" + "="*100)
        print("HYPOTHESIS VALIDATION REPORT")
        print("="*100 + "\n")
        
        report = analyzer.generate_hypothesis_validation_report()
        
        import json
        report_path = output_dir / 'hypothesis_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to {report_path}")
    
    # Generate LaTeX tables
    if args.generate_latex:
        latex = analyzer.generate_latex_table(args.study_dimension)
        latex_path = output_dir / f'{args.study_dimension}_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_path}")
    
    # Generate heatmaps
    if args.plot_heatmap:
        heatmap_path = output_dir / f'{args.study_dimension}_heatmap.png'
        analyzer.plot_comparison_heatmap(
            args.study_dimension,
            output_path=str(heatmap_path)
        )


if __name__ == '__main__':
    main()
