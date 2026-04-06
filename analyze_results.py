#!/usr/bin/env python
"""
Analysis & Visualization for LightGCN Experiment Results
=========================================================
Reads JSON results from experiments and generates:
1. Performance comparison tables (text + LaTeX)
2. Convergence curves (loss vs epoch per dataset)
3. Metric comparison bar charts across datasets
4. Training efficiency analysis (time, GPU memory)
5. Dataset characteristics impact analysis
6. Per-K metric sensitivity analysis

Usage:
    python analyze_results.py
    python analyze_results.py --results_dir results/LightGCN
    python analyze_results.py --no_plots   # skip matplotlib plots
"""

import json
import os
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

# Try to import matplotlib (optional for environments without display)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# Dataset metadata for analysis
# ============================================================
DATASET_META = {
    'movielens': {
        'full_name': 'MovieLens 1M',
        'domain': 'Movies',
        'users': 6040,
        'items': 3260,
        'train_interactions': 801218,
        'test_interactions': 197321,
        'total_interactions': 998539,
        'sparsity': 0.949288,
        'density': 1 - 0.949288,
    },
    'gowalla': {
        'full_name': 'Gowalla',
        'domain': 'Location',
        'users': 29858,
        'items': 40988,
        'train_interactions': 833031,
        'test_interactions': 194433,
        'total_interactions': 1027464,
        'sparsity': 0.999160,
        'density': 1 - 0.999160,
    },
    'yelp2018': {
        'full_name': 'Yelp 2018',
        'domain': 'Business',
        'users': 93537,
        'items': 53347,
        'train_interactions': 2059466,
        'test_interactions': 474293,
        'total_interactions': 2533759,
        'sparsity': 0.999492,
        'density': 1 - 0.999492,
    },
    'amazon-book': {
        'full_name': 'Amazon Book',
        'domain': 'Books',
        'users': 603617,
        'items': 298833,
        'train_interactions': 13502081,
        'test_interactions': 3115104,
        'total_interactions': 16617185,
        'sparsity': 0.999908,
        'density': 1 - 0.999908,
    },
    'collected': {
        'full_name': 'Collected Dataset',
        'domain': 'TBD',
        'users': 0,
        'items': 0,
        'train_interactions': 0,
        'test_interactions': 0,
        'total_interactions': 0,
        'sparsity': 0,
        'density': 0,
    },
}

METRIC_DISPLAY = {
    'recall': 'Recall',
    'precision': 'Precision',
    'ndcg': 'NDCG',
    'mrr': 'MRR',
    'hit_rate': 'Hit Rate',
    'map': 'MAP',
}

DATASET_ORDER = ['movielens', 'gowalla', 'yelp2018', 'amazon-book', 'collected']


def load_all_results(results_dir):
    """Load all per-run JSON result files."""
    all_runs = defaultdict(list)
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith('.json') or filename in ('summary.json',):
            continue
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        all_runs[data['dataset']].append(data)
    return dict(all_runs)


def compute_aggregated_metrics(all_runs):
    """Compute mean and std of best metrics across runs for each dataset."""
    aggregated = {}
    for ds, runs in all_runs.items():
        best_metrics_list = []
        training_times = []
        best_epochs = []
        gpu_memories = []

        for run in runs:
            if run.get('best_results') and run['best_results'].get('metrics'):
                best_metrics_list.append(run['best_results']['metrics'])
                best_epochs.append(run['best_results'].get('best_epoch', -1))
            training_times.append(run.get('total_train_time_seconds', 0))
            # Get peak GPU memory from epoch logs
            if run.get('epoch_logs'):
                peak_mem = max(e.get('gpu_memory_peak_MB', 0) for e in run['epoch_logs'])
                gpu_memories.append(peak_mem)

        if not best_metrics_list:
            continue

        agg = {
            'num_runs': len(best_metrics_list),
            'training_time': {
                'mean': round(mean(training_times), 2),
                'std': round(stdev(training_times) if len(training_times) > 1 else 0, 2),
            },
            'best_epoch': {
                'mean': round(mean(best_epochs), 1) if best_epochs else -1,
                'std': round(stdev(best_epochs) if len(best_epochs) > 1 else 0, 1),
            },
            'gpu_memory_peak_MB': {
                'mean': round(mean(gpu_memories), 2) if gpu_memories else 0,
            },
            'metrics': {},
        }

        # Aggregate metrics
        for metric in best_metrics_list[0]:
            if isinstance(best_metrics_list[0][metric], dict):
                agg['metrics'][metric] = {}
                for k_val in best_metrics_list[0][metric]:
                    values = [float(r[metric][k_val]) for r in best_metrics_list
                              if metric in r and k_val in r[metric]]
                    agg['metrics'][metric][k_val] = {
                        'mean': round(mean(values), 6),
                        'std': round(stdev(values) if len(values) > 1 else 0, 6),
                    }

        aggregated[ds] = agg
    return aggregated


# ============================================================
# 1. Text Tables
# ============================================================
def print_performance_table(aggregated, output_dir):
    """Print and save a comprehensive performance comparison table."""
    lines = []
    lines.append("=" * 100)
    lines.append("  LightGCN Performance Comparison Across Datasets")
    lines.append("=" * 100)

    for k_val in ['5', '10', '20']:
        lines.append(f"\n  --- Top-{k_val} Metrics ---")
        header = f"  {'Dataset':<15}"
        for metric in ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']:
            header += f" {METRIC_DISPLAY.get(metric, metric):>14}"
        lines.append(header)
        lines.append("  " + "-" * 99)

        for ds in DATASET_ORDER:
            if ds not in aggregated:
                continue
            agg = aggregated[ds]
            row = f"  {ds:<15}"
            for metric in ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']:
                if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                    m = agg['metrics'][metric][k_val]['mean']
                    s = agg['metrics'][metric][k_val]['std']
                    row += f" {m:.4f}±{s:.4f}"
                else:
                    row += f" {'N/A':>14}"
            lines.append(row)

    lines.append(f"\n\n  --- Training Efficiency ---")
    lines.append(f"  {'Dataset':<15} {'Time (s)':<20} {'Best Epoch':<18} {'GPU Mem (MB)':<15} {'Runs':<6}")
    lines.append("  " + "-" * 74)
    for ds in DATASET_ORDER:
        if ds not in aggregated:
            continue
        agg = aggregated[ds]
        t_mean = agg['training_time']['mean']
        t_std = agg['training_time']['std']
        e_mean = agg['best_epoch']['mean']
        gpu = agg['gpu_memory_peak_MB']['mean']
        n = agg['num_runs']
        lines.append(
            f"  {ds:<15} {t_mean:>7.0f}±{t_std:<7.0f}    {e_mean:>7.1f}           {gpu:>8.0f}       {n:>3}")

    text = "\n".join(lines)
    print(text)

    # Save to file
    table_file = os.path.join(output_dir, 'performance_table.txt')
    with open(table_file, 'w') as f:
        f.write(text)
    print(f"\n  Table saved to {table_file}")


# ============================================================
# 2. LaTeX Tables (for report)
# ============================================================
def generate_latex_tables(aggregated, output_dir):
    """Generate LaTeX-formatted tables for the course report."""
    lines = []
    lines.append("% Auto-generated LaTeX table for LightGCN results")
    lines.append("% Copy this into your report\n")

    for k_val in ['5', '10', '20']:
        metrics = ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']
        n_metrics = len(metrics)

        lines.append(f"% === Top-{k_val} Results ===")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append(f"\\caption{{LightGCN Performance at Top-{k_val}}}")
        lines.append(f"\\label{{tab:lightgcn-top{k_val}}}")
        col_spec = "l" + "c" * n_metrics
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header = "Dataset"
        for m in metrics:
            header += f" & {METRIC_DISPLAY.get(m, m)}@{k_val}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for ds in DATASET_ORDER:
            if ds not in aggregated:
                continue
            agg = aggregated[ds]
            ds_display = DATASET_META.get(ds, {}).get('full_name', ds)
            row = ds_display
            for metric in metrics:
                if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                    m_val = agg['metrics'][metric][k_val]['mean']
                    s_val = agg['metrics'][metric][k_val]['std']
                    row += f" & {m_val:.4f}$\\pm${s_val:.4f}"
                else:
                    row += " & N/A"
            row += " \\\\"
            lines.append(row)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")

    # Training efficiency table
    lines.append("% === Training Efficiency ===")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{LightGCN Training Efficiency}")
    lines.append("\\label{tab:lightgcn-efficiency}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Training Time (s) & Best Epoch & GPU Memory (MB) & Runs \\\\")
    lines.append("\\midrule")

    for ds in DATASET_ORDER:
        if ds not in aggregated:
            continue
        agg = aggregated[ds]
        ds_display = DATASET_META.get(ds, {}).get('full_name', ds)
        t = agg['training_time']
        e = agg['best_epoch']['mean']
        gpu = agg['gpu_memory_peak_MB']['mean']
        n = agg['num_runs']
        lines.append(
            f"{ds_display} & {t['mean']:.0f}$\\pm${t['std']:.0f} "
            f"& {e:.0f} & {gpu:.0f} & {n} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Dataset characteristics table
    lines.append("% === Dataset Characteristics ===")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Dataset Characteristics}")
    lines.append("\\label{tab:dataset-characteristics}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Domain & Users & Items & Interactions & Sparsity \\\\")
    lines.append("\\midrule")
    for ds in DATASET_ORDER:
        meta = DATASET_META.get(ds)
        if meta and meta['users'] > 0:
            lines.append(
                f"{meta['full_name']} & {meta['domain']} & "
                f"{meta['users']:,} & {meta['items']:,} & "
                f"{meta['total_interactions']:,} & {meta['sparsity']:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_text = "\n".join(lines)
    latex_file = os.path.join(output_dir, 'latex_tables.tex')
    with open(latex_file, 'w') as f:
        f.write(latex_text)
    print(f"  LaTeX tables saved to {latex_file}")


# ============================================================
# 3. CSV Export
# ============================================================
def export_csv(aggregated, output_dir):
    """Export results to CSV for easy spreadsheet analysis."""
    csv_file = os.path.join(output_dir, 'results_summary.csv')
    rows = []

    for ds in DATASET_ORDER:
        if ds not in aggregated:
            continue
        agg = aggregated[ds]
        for k_val in ['5', '10', '20']:
            row = {'dataset': ds, 'K': k_val}
            for metric in ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']:
                if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                    row[f'{metric}_mean'] = agg['metrics'][metric][k_val]['mean']
                    row[f'{metric}_std'] = agg['metrics'][metric][k_val]['std']
                else:
                    row[f'{metric}_mean'] = ''
                    row[f'{metric}_std'] = ''
            row['training_time_mean'] = agg['training_time']['mean']
            row['best_epoch_mean'] = agg['best_epoch']['mean']
            row['gpu_memory_MB'] = agg['gpu_memory_peak_MB']['mean']
            row['num_runs'] = agg['num_runs']
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV exported to {csv_file}")


# ============================================================
# 4. Convergence Plots
# ============================================================
def plot_convergence_curves(all_runs, output_dir):
    """Plot training loss convergence for each dataset."""
    if not HAS_MATPLOTLIB:
        print("  Skipping convergence plots (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, len(all_runs), figsize=(5 * len(all_runs), 4))
    if len(all_runs) == 1:
        axes = [axes]

    for idx, (ds, runs) in enumerate(sorted(all_runs.items(),
                                            key=lambda x: DATASET_ORDER.index(x[0])
                                            if x[0] in DATASET_ORDER else 99)):
        ax = axes[idx]
        for i, run in enumerate(runs):
            epochs = [e['epoch'] for e in run.get('epoch_logs', [])]
            losses = [e['loss'] for e in run.get('epoch_logs', [])]
            if epochs and losses:
                ax.plot(epochs, losses, alpha=0.6, linewidth=0.8,
                        label=f"seed={run.get('seed', '?')}")

        ax.set_title(DATASET_META.get(ds, {}).get('full_name', ds), fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BPR Loss')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence_curves.png'), dpi=150)
    plt.close(fig)
    print(f"  Convergence curves saved to convergence_curves.png")


# ============================================================
# 5. Metric Comparison Bar Charts
# ============================================================
def plot_metric_comparison(aggregated, output_dir):
    """Bar charts comparing metrics across datasets for each K."""
    if not HAS_MATPLOTLIB:
        print("  Skipping metric comparison plots (matplotlib not available)")
        return

    datasets = [ds for ds in DATASET_ORDER if ds in aggregated]
    if not datasets:
        return

    metrics = ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']
    k_values = ['5', '10', '20']

    for k_val in k_values:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'LightGCN Performance Comparison @ Top-{k_val}',
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for m_idx, metric in enumerate(metrics):
            ax = axes[m_idx]
            means = []
            stds = []
            labels = []
            for ds in datasets:
                agg = aggregated[ds]
                if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                    means.append(agg['metrics'][metric][k_val]['mean'])
                    stds.append(agg['metrics'][metric][k_val]['std'])
                else:
                    means.append(0)
                    stds.append(0)
                labels.append(DATASET_META.get(ds, {}).get('full_name', ds))

            colors = plt.cm.Set2(range(len(labels)))
            bars = ax.bar(range(len(labels)), means, yerr=stds,
                         capsize=3, color=colors, edgecolor='gray', linewidth=0.5)
            ax.set_title(f'{METRIC_DISPLAY.get(metric, metric)}@{k_val}', fontsize=11)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar_obj, val in zip(bars, means):
                if val > 0:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2., bar_obj.get_height(),
                            f'{val:.4f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f'metrics_comparison_top{k_val}.png'), dpi=150)
        plt.close(fig)

    print(f"  Metric comparison plots saved")


# ============================================================
# 6. Training Efficiency Plot
# ============================================================
def plot_training_efficiency(aggregated, output_dir):
    """Plot training time and GPU memory usage comparison."""
    if not HAS_MATPLOTLIB:
        return

    datasets = [ds for ds in DATASET_ORDER if ds in aggregated]
    if not datasets:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Efficiency Comparison', fontsize=14, fontweight='bold')

    labels = [DATASET_META.get(ds, {}).get('full_name', ds) for ds in datasets]
    colors = plt.cm.Set2(range(len(labels)))

    # Training time
    times = [aggregated[ds]['training_time']['mean'] / 60 for ds in datasets]  # in minutes
    time_stds = [aggregated[ds]['training_time']['std'] / 60 for ds in datasets]
    ax1.bar(range(len(labels)), times, yerr=time_stds, capsize=3,
            color=colors, edgecolor='gray')
    ax1.set_title('Training Time (minutes)')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # GPU Memory
    mems = [aggregated[ds]['gpu_memory_peak_MB']['mean'] for ds in datasets]
    ax2.bar(range(len(labels)), mems, color=colors, edgecolor='gray')
    ax2.set_title('Peak GPU Memory (MB)')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'training_efficiency.png'), dpi=150)
    plt.close(fig)
    print(f"  Training efficiency plot saved")


# ============================================================
# 7. Dataset Characteristics Impact Analysis
# ============================================================
def analyze_dataset_impact(aggregated, output_dir):
    """Analyze how dataset characteristics affect model performance."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("  Skipping dataset impact analysis (requires matplotlib and numpy)")
        return

    datasets = [ds for ds in DATASET_ORDER if ds in aggregated and ds in DATASET_META
                and DATASET_META[ds]['users'] > 0]
    if len(datasets) < 2:
        print("  Not enough datasets for impact analysis")
        return

    # Collect dataset properties and performance
    characteristics = {
        'density': [],
        'n_users': [],
        'n_items': [],
        'interactions': [],
        'items_per_user': [],
    }
    performance = {'recall@20': [], 'ndcg@20': [], 'hit_rate@20': []}

    for ds in datasets:
        meta = DATASET_META[ds]
        characteristics['density'].append(meta['density'])
        characteristics['n_users'].append(meta['users'])
        characteristics['n_items'].append(meta['items'])
        characteristics['interactions'].append(meta['total_interactions'])
        characteristics['items_per_user'].append(
            meta['total_interactions'] / meta['users'] if meta['users'] > 0 else 0)

        agg = aggregated[ds]
        for metric, k_val in [('recall', '20'), ('ndcg', '20'), ('hit_rate', '20')]:
            key = f'{metric}@{k_val}'
            if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                performance[key].append(agg['metrics'][metric][k_val]['mean'])
            else:
                performance[key].append(0)

    # Plot scatter: characteristic vs performance
    char_names = {
        'density': 'Density (1 - Sparsity)',
        'n_users': 'Number of Users',
        'n_items': 'Number of Items',
        'interactions': 'Total Interactions',
        'items_per_user': 'Avg Items per User',
    }

    fig, axes = plt.subplots(len(performance), len(characteristics),
                             figsize=(4 * len(characteristics), 3.5 * len(performance)))
    fig.suptitle('Dataset Characteristics vs Model Performance', fontsize=14, fontweight='bold')

    ds_labels = [DATASET_META[ds]['full_name'] for ds in datasets]

    for p_idx, (perf_name, perf_vals) in enumerate(performance.items()):
        for c_idx, (char_name, char_vals) in enumerate(characteristics.items()):
            ax = axes[p_idx][c_idx] if len(performance) > 1 else axes[c_idx]
            ax.scatter(char_vals, perf_vals, c='steelblue', s=80, zorder=5)
            for i, label in enumerate(ds_labels):
                ax.annotate(label, (char_vals[i], perf_vals[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)
            if char_name in ('n_users', 'n_items', 'interactions'):
                ax.set_xscale('log')
            if char_name == 'density':
                ax.set_xscale('log')
            ax.set_xlabel(char_names[char_name], fontsize=8)
            ax.set_ylabel(perf_name, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'dataset_impact_analysis.png'), dpi=150)
    plt.close(fig)
    print(f"  Dataset impact analysis plot saved")


# ============================================================
# 8. Per-K Sensitivity Analysis
# ============================================================
def plot_k_sensitivity(aggregated, output_dir):
    """Plot how metrics change with different K values."""
    if not HAS_MATPLOTLIB:
        return

    datasets = [ds for ds in DATASET_ORDER if ds in aggregated]
    if not datasets:
        return

    metrics = ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']
    k_values = ['5', '10', '20']
    k_ints = [5, 10, 20]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Metric Sensitivity to K Value', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        for ds in datasets:
            agg = aggregated[ds]
            vals = []
            for k_val in k_values:
                if metric in agg['metrics'] and k_val in agg['metrics'][metric]:
                    vals.append(agg['metrics'][metric][k_val]['mean'])
                else:
                    vals.append(0)
            label = DATASET_META.get(ds, {}).get('full_name', ds)
            ax.plot(k_ints, vals, 'o-', label=label, markersize=5, linewidth=1.5)

        ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=11)
        ax.set_xlabel('K')
        ax.set_xticks(k_ints)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'k_sensitivity.png'), dpi=150)
    plt.close(fig)
    print(f"  K-sensitivity plot saved")


# ============================================================
# 9. Textual Insights
# ============================================================
def generate_insights(aggregated, output_dir):
    """Generate textual analysis insights."""
    lines = []
    lines.append("=" * 70)
    lines.append("  ANALYSIS INSIGHTS")
    lines.append("=" * 70)

    datasets = [ds for ds in DATASET_ORDER if ds in aggregated]
    if not datasets:
        lines.append("  No results available for analysis.")
        text = "\n".join(lines)
        print(text)
        return

    # 1. Best / worst performing dataset
    recall_20 = {}
    for ds in datasets:
        agg = aggregated[ds]
        if 'recall' in agg['metrics'] and '20' in agg['metrics']['recall']:
            recall_20[ds] = agg['metrics']['recall']['20']['mean']

    if recall_20:
        best_ds = max(recall_20, key=recall_20.get)
        worst_ds = min(recall_20, key=recall_20.get)
        lines.append(f"\n  1. OVERALL PERFORMANCE:")
        lines.append(f"     Best performing dataset:  {best_ds} "
                     f"(Recall@20 = {recall_20[best_ds]:.4f})")
        lines.append(f"     Worst performing dataset: {worst_ds} "
                     f"(Recall@20 = {recall_20[worst_ds]:.4f})")
        if recall_20[worst_ds] > 0:
            ratio = recall_20[best_ds] / recall_20[worst_ds]
            lines.append(f"     Performance ratio: {ratio:.2f}x")

    # 2. Sparsity impact
    lines.append(f"\n  2. SPARSITY IMPACT:")
    for ds in datasets:
        meta = DATASET_META.get(ds, {})
        if meta.get('users') and ds in recall_20:
            density = meta.get('density', 0)
            lines.append(f"     {ds:<15} density={density:.6f}  Recall@20={recall_20.get(ds, 0):.4f}")
    lines.append(f"     --> Denser datasets (like MovieLens) tend to perform better")
    lines.append(f"     --> Extremely sparse datasets (like Amazon-Book) are more challenging")

    # 3. Training efficiency
    lines.append(f"\n  3. TRAINING EFFICIENCY:")
    for ds in datasets:
        agg = aggregated[ds]
        t = agg['training_time']['mean'] / 60
        e = agg['best_epoch']['mean']
        meta = DATASET_META.get(ds, {})
        interactions = meta.get('train_interactions', 0)
        lines.append(f"     {ds:<15} time={t:.1f}min  best_epoch={e:.0f}  "
                     f"interactions={interactions:,}")

    # 4. Metric correlation
    lines.append(f"\n  4. METRIC CONSISTENCY:")
    lines.append(f"     Checking if different metrics agree on dataset ranking...")
    rankings = {}
    for metric in ['recall', 'ndcg', 'mrr', 'hit_rate']:
        vals = {}
        for ds in datasets:
            agg = aggregated[ds]
            if metric in agg['metrics'] and '20' in agg['metrics'][metric]:
                vals[ds] = agg['metrics'][metric]['20']['mean']
        if vals:
            rankings[metric] = sorted(vals, key=vals.get, reverse=True)

    if len(rankings) > 1:
        metric_names = list(rankings.keys())
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                m1, m2 = metric_names[i], metric_names[j]
                if rankings[m1] == rankings[m2]:
                    lines.append(f"     {m1} and {m2}: SAME ranking")
                else:
                    lines.append(f"     {m1}: {' > '.join(rankings[m1])}")
                    lines.append(f"     {m2}: {' > '.join(rankings[m2])}")
                    lines.append(f"     --> Rankings DIFFER - worth investigating!")

    # 5. Variance analysis
    lines.append(f"\n  5. VARIANCE ACROSS RUNS:")
    for ds in datasets:
        agg = aggregated[ds]
        if 'recall' in agg['metrics'] and '20' in agg['metrics']['recall']:
            m = agg['metrics']['recall']['20']['mean']
            s = agg['metrics']['recall']['20']['std']
            cv = (s / m * 100) if m > 0 else 0
            stability = "STABLE" if cv < 2 else ("MODERATE" if cv < 5 else "HIGH VARIANCE")
            lines.append(f"     {ds:<15} Recall@20: {m:.4f}±{s:.4f} "
                         f"(CV={cv:.1f}%) [{stability}]")

    # 6. Items-per-user ratio analysis
    lines.append(f"\n  6. DATASET CHARACTERISTIC ANALYSIS:")
    for ds in datasets:
        meta = DATASET_META.get(ds, {})
        if meta.get('users') and meta.get('items'):
            ratio = meta['items'] / meta['users']
            avg_items = meta['total_interactions'] / meta['users'] if meta['users'] else 0
            lines.append(f"     {ds:<15} items/user_ratio={ratio:.2f}  "
                         f"avg_interactions/user={avg_items:.1f}")

    text = "\n".join(lines)
    print(text)

    insights_file = os.path.join(output_dir, 'analysis_insights.txt')
    with open(insights_file, 'w') as f:
        f.write(text)
    print(f"\n  Insights saved to {insights_file}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Analyze LightGCN experiment results')
    parser.add_argument('--results_dir', type=str, default='results/LightGCN',
                        help='Directory containing experiment result JSON files')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots (text analysis only)')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    output_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading results from: {results_dir}")
    all_runs = load_all_results(results_dir)
    if not all_runs:
        print("No result files found!")
        return

    print(f"Found results for {len(all_runs)} datasets: {list(all_runs.keys())}")
    for ds, runs in all_runs.items():
        print(f"  {ds}: {len(runs)} runs")

    # Compute aggregated metrics
    aggregated = compute_aggregated_metrics(all_runs)

    # Generate all outputs
    print_performance_table(aggregated, output_dir)
    generate_latex_tables(aggregated, output_dir)
    export_csv(aggregated, output_dir)
    generate_insights(aggregated, output_dir)

    if not args.no_plots and HAS_MATPLOTLIB:
        print(f"\nGenerating plots...")
        plot_convergence_curves(all_runs, output_dir)
        plot_metric_comparison(aggregated, output_dir)
        plot_training_efficiency(aggregated, output_dir)
        plot_k_sensitivity(aggregated, output_dir)
        analyze_dataset_impact(aggregated, output_dir)
    elif not HAS_MATPLOTLIB:
        print("\n  matplotlib not installed. Install with: pip install matplotlib")
        print("  Plots were skipped, but text/CSV/LaTeX outputs are generated.")

    print(f"\n{'='*60}")
    print(f"  Analysis complete! All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
