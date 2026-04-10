#!/usr/bin/env python
"""
Experiment Runner for DirectAU
==============================
Runs DirectAU on all 5 datasets with multiple seeds (3-5 runs each).
Saves per-run JSON results and generates an aggregated summary.

Usage:
    # Run all datasets (5 runs each):
    python run_experiments.py

    # Run specific datasets:
    python run_experiments.py --datasets movielens gowalla

    # Fewer runs for quick testing:
    python run_experiments.py --num_runs 3

    # Skip already-completed runs:
    python run_experiments.py --skip_existing

    # Custom output directory:
    python run_experiments.py --output_dir ../../results/DirectAU
"""

import subprocess
import sys
import os
import json
import time
import argparse
import glob
import re
from pathlib import Path
from statistics import mean, stdev


# ============================================================
# 5 different seeds for reproducibility
# ============================================================
SEEDS = [2020, 2021, 2022]

ANSI_ESCAPE_PATTERN = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')


# ============================================================
# Dataset-specific hyperparameters
# ============================================================
DATASET_CONFIGS = {
    'movielens': {
        'epochs': 300,
        'eval_step': 1,
        'train_batch_size': 256,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'stopping_step': 10,
        'topk': [5, 10, 20],
    },
    'gowalla': {
        'epochs': 300,
        'eval_step': 10,
        'train_batch_size': 1024,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'stopping_step': 10,
        'topk': [5, 10, 20],
    },
    'yelp2018': {
        'epochs': 300,
        'eval_step': 1,
        'train_batch_size': 1024,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'stopping_step': 10,
        'topk': [5, 10, 20],
    },
    'amazon-book': {
        'epochs': 300,
        'eval_step': 10,
        'train_batch_size': 512,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'stopping_step': 10,
        'topk': [5, 10, 20],
    },
    'collected': {
        'epochs': 300,
        'eval_step': 10,
        'train_batch_size': 256,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'stopping_step': 10,
        'topk': [5, 10, 20],
    },
}

DATASET_ALIASES = {
    'yelp18': 'yelp2018',
}

# Keep yelp18 usable when the local data folder is named yelp18.
DATASET_CONFIGS['yelp18'] = dict(DATASET_CONFIGS['yelp2018'])


def _resolve_dataset_name(dataset, script_dir):
    """Resolve dataset alias based on available data folders.

    Supports both yelp2018 and yelp18 layouts.
    """
    mapped = DATASET_ALIASES.get(dataset, dataset)
    if dataset not in ('yelp18', 'yelp2018'):
        return mapped

    data_root = os.path.join(script_dir, 'data')
    yelp2018_dir = os.path.join(data_root, 'yelp2018')
    yelp18_dir = os.path.join(data_root, 'yelp18')

    if os.path.isdir(yelp2018_dir):
        return 'yelp2018'
    if os.path.isdir(yelp18_dir):
        return 'yelp18'
    return mapped


def _load_convert_tools(script_dir):
    """Import convert_log helpers used by the experiment pipeline."""
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from convert_log import parse_recbole_log_to_json, clean_log_file
    return parse_recbole_log_to_json, clean_log_file


def _metric_name_to_group(metric_name):
    """Map flattened metric key (e.g., recall@20) to grouped metric name."""
    name = metric_name.split('@', 1)[0].strip().lower()
    if name == 'hit':
        return 'hit_rate'
    return name


def _build_nested_metrics(flat_metrics):
    """Convert {'recall@20': 0.1} -> {'recall': {'20': 0.1}}."""
    nested = {}
    for key, value in flat_metrics.items():
        if '@' not in key:
            continue
        metric_name, k_val = key.split('@', 1)
        metric_group = _metric_name_to_group(metric_name)
        metric_bucket = nested.setdefault(metric_group, {})
        metric_bucket[str(k_val)] = round(float(value), 6)
    return nested


def _to_int(value, default=-1):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_metrics(metrics):
    """Normalize metrics to LightGCN-compatible nested format and value types."""
    if not isinstance(metrics, dict):
        return {}

    # Convert flattened keys like recall@20 if needed.
    if metrics and all(isinstance(k, str) and '@' in k for k in metrics.keys()):
        metrics = _build_nested_metrics(metrics)

    normalized = {}
    for metric_name, metric_values in metrics.items():
        if not isinstance(metric_values, dict):
            continue
        bucket = {}
        for k_val, score in metric_values.items():
            bucket[str(k_val)] = round(_to_float(score, 0.0), 6)
        if bucket:
            normalized[str(metric_name)] = bucket
    return normalized


def _normalize_directau_run(parsed, dataset, seed, run_id, elapsed, config, data_dir):
    """Force run output into LightGCN-like schema for analyze_results.py compatibility."""
    parsed = parsed if isinstance(parsed, dict) else {}

    epoch_logs_raw = parsed.get('epoch_logs', [])
    epoch_logs = []
    if isinstance(epoch_logs_raw, list):
        for idx, e in enumerate(epoch_logs_raw):
            if not isinstance(e, dict):
                continue
            epoch_logs.append({
                'epoch': _to_int(e.get('epoch', idx + 1), idx + 1),
                'loss': round(_to_float(e.get('loss', 0.0), 0.0), 6),
                'epoch_time_seconds': round(_to_float(e.get('epoch_time_seconds', 0.0), 0.0), 2),
                # Keep numeric 0.0 when unavailable so max()/mean() in analyze_results is safe.
                'gpu_memory_peak_MB': round(_to_float(e.get('gpu_memory_peak_MB', 0.0), 0.0), 2),
            })

    test_results_raw = parsed.get('test_results', [])
    test_results = []
    if isinstance(test_results_raw, list):
        for t in test_results_raw:
            if not isinstance(t, dict):
                continue
            test_results.append({
                'epoch': _to_int(t.get('epoch', 0), 0),
                'test_time_seconds': round(_to_float(t.get('test_time_seconds', 0.0), 0.0), 2),
                'metrics': _normalize_metrics(t.get('metrics', {})),
            })

    best_results_raw = parsed.get('best_results', {})
    best_results_raw = best_results_raw if isinstance(best_results_raw, dict) else {}
    best_metrics = _normalize_metrics(best_results_raw.get('metrics', {}))
    if not best_metrics and test_results:
        # Fallback to final available test metrics if best metrics not found in logs.
        best_metrics = _normalize_metrics(test_results[-1].get('metrics', {}))

    best_epoch = _to_int(best_results_raw.get('best_epoch', best_results_raw.get('epoch', -1)), -1)
    if best_epoch < 0 and epoch_logs:
        best_epoch = _to_int(epoch_logs[-1].get('epoch', -1), -1)

    normalized = {
        'config': parsed.get('config', {
            'seed': seed,
            'epochs': config['epochs'],
            'train_batch_size': config['train_batch_size'],
            'eval_batch_size': config['eval_batch_size'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'topk': config['topk'],
            'data_path': f"{Path(data_dir).as_posix()}/",
        }),
        'dataset': dataset,
        'model': parsed.get('model', 'DirectAU'),
        'seed': _to_int(parsed.get('seed', seed), seed),
        'topks': parsed.get('topks', config['topk']),
        'dataset_stats': parsed.get('dataset_stats', {}),
        'epoch_logs': epoch_logs,
        'test_results': test_results,
        'best_results': {
            'epoch': best_epoch,
            'metrics': best_metrics,
            'best_epoch': best_epoch,
        },
        'total_train_time_seconds': round(
            _to_float(parsed.get('total_train_time_seconds', elapsed), elapsed), 2
        ),
        'system_info': parsed.get('system_info', {}),
        'run_id': run_id,
    }

    return normalized


def check_data_exists(dataset_name):
    """Check if fixed RecBole benchmark split files are available."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', dataset_name)
    train_inter = os.path.join(data_dir, f'{dataset_name}.train.inter')
    valid_inter = os.path.join(data_dir, f'{dataset_name}.valid.inter')
    test_inter = os.path.join(data_dir, f'{dataset_name}.test.inter')
    return os.path.exists(train_inter) and os.path.exists(valid_inter) and os.path.exists(test_inter)


def _list_log_files(code_dir):
    """Return current RecBole log files under code/log (recursive)."""
    return set(glob.glob(os.path.join(code_dir, 'log', '**', '*.log'), recursive=True))


def _pick_new_log_file(code_dir, logs_before, run_start_ts):
    """Pick log file created/updated by the latest run."""
    logs_after = _list_log_files(code_dir)
    created = sorted(logs_after - logs_before)
    if created:
        return created[-1]

    candidates = []
    for log_file in logs_after:
        try:
            mtime = os.path.getmtime(log_file)
        except OSError:
            continue
        if mtime >= run_start_ts - 1:
            candidates.append((mtime, log_file))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def _write_override_config(code_dir, data_dir, dataset, seed, config, include_hparams=True):
    """Write a temporary YAML override config for one run.

    Epochs is always overridden; other hyperparameters are optional.
    """
    override_path = os.path.join(code_dir, f'.tmp_override_{dataset}_{seed}.yaml')
    lines = [
        f'seed: {seed}',
        'reproducibility: True',
        f"data_path: '{Path(data_dir).as_posix()}/'",
        "benchmark_filename: ['train', 'valid', 'test']",
        f'epochs: {config["epochs"]}',
        f'eval_step: {config["eval_step"]}',
    ]
    if include_hparams:
        lines.extend([
            f'train_batch_size: {config["train_batch_size"]}',
            f'eval_batch_size: {config["eval_batch_size"]}',
            f'learning_rate: {config["learning_rate"]}',
            f'weight_decay: {config["weight_decay"]}',
            f'stopping_step: {config["stopping_step"]}',
            f"topk: {config['topk']}",
            "metrics: ['Recall', 'NDCG', 'MRR', 'Hit', 'MAP', 'Precision']",
            'valid_metric: NDCG@20',
            'valid_metric_bigger: True',
        ])
    with open(override_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return override_path


def _find_dataset_custom_config(code_dir, dataset):
    """Return dataset-specific config file name if present, else None."""
    candidates = [f'{dataset}_directau.yaml']
    if dataset == 'yelp2018':
        candidates.append('yelp18_directau.yaml')

    for filename in candidates:
        path = os.path.join(code_dir, filename)
        if os.path.exists(path):
            return filename
    return None


def run_single_experiment(dataset, seed, run_id, output_dir, num_runs, config_file_list):
    """Run a single DirectAU experiment via subprocess."""
    config = DATASET_CONFIGS[dataset]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(script_dir, 'code')
    data_dir = os.path.join(script_dir, 'data')
    run_script = os.path.join(code_dir, 'run_recbole.py')
    parse_recbole_log_to_json, clean_log_file = _load_convert_tools(script_dir)

    dataset_custom_cfg = _find_dataset_custom_config(code_dir, dataset)
    include_hparams = dataset_custom_cfg is None
    override_path = _write_override_config(
        code_dir, data_dir, dataset, seed, config, include_hparams=include_hparams
    )
    # run_recbole.py splits config_files by spaces, so avoid absolute paths with spaces.
    override_rel_path = os.path.basename(override_path)
    run_config_files = list(config_file_list)
    if dataset_custom_cfg and dataset_custom_cfg not in run_config_files:
        run_config_files.append(dataset_custom_cfg)
    run_config_files.append(override_rel_path)

    cmd = [
        sys.executable,
        run_script,
        '--model=DirectAU',
        f'--dataset={dataset}',
        f"--config_files={' '.join(run_config_files)}",
        f"--eval_step={config['eval_step']}",
    ]

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset} | Seed: {seed} | Run: {run_id+1}/{num_runs}")
    print(
        f"  Config: epochs={config['epochs']}, eval_step={config['eval_step']}, batch={config['train_batch_size']}, "
        f"eval_batch={config['eval_batch_size']}, lr={config['learning_rate']}, "
        f"weight_decay={config['weight_decay']}"
    )
    if dataset_custom_cfg:
        print(f"  Using dataset custom config: {dataset_custom_cfg}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    logs_before = _list_log_files(code_dir)
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=code_dir)
    finally:
        if os.path.exists(override_path):
            try:
                os.remove(override_path)
            except OSError:
                pass
    elapsed = time.time() - start_time

    status = 'SUCCESS' if result.returncode == 0 else 'FAILED'
    print(f"\n[{status}] {dataset} seed={seed} completed in {elapsed/60:.1f} minutes")

    # Parse the latest log into LightGCN-like JSON structure.
    log_file = _pick_new_log_file(code_dir, logs_before, start_time)
    parsed = None
    if log_file:
        try:
            clean_status = clean_log_file(Path(log_file))
            if isinstance(clean_status, str) and clean_status.startswith('error:'):
                print(f"WARNING: Failed to clean log {log_file}: {clean_status}")
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                log_content = ANSI_ESCAPE_PATTERN.sub('', log_content)
                parsed = parse_recbole_log_to_json(log_content)
        except Exception as exc:
            print(f"WARNING: Failed to parse log {log_file}: {exc}")

    if parsed is None:
        parsed = {}

    parsed = _normalize_directau_run(
        parsed=parsed,
        dataset=dataset,
        seed=seed,
        run_id=run_id,
        elapsed=elapsed,
        config=config,
        data_dir=data_dir,
    )

    result_file = os.path.join(output_dir, f'{dataset}_seed{seed}_run{run_id}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, indent=2)

    return result.returncode


def aggregate_results(output_dir):
    """Aggregate per-run JSON results into a summary with mean and std."""
    all_data = {}

    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith('.json') or filename in ('summary.json',):
            continue
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ds = data.get('dataset')
        if not ds:
            continue

        if ds not in all_data:
            all_data[ds] = {
                'runs': [],
                'training_times': [],
                'best_epochs': [],
                'dataset_stats': data.get('dataset_stats', {}),
                'system_info': data.get('system_info', {}),
                'config': data.get('config', {}),
            }

        if data.get('best_results') and data['best_results'].get('metrics'):
            all_data[ds]['runs'].append(data['best_results']['metrics'])
            all_data[ds]['best_epochs'].append(data['best_results'].get('best_epoch', -1))
        if data.get('total_train_time_seconds') is not None:
            all_data[ds]['training_times'].append(float(data.get('total_train_time_seconds', 0)))

    summary = {}
    for ds, info in all_data.items():
        runs = info['runs']
        if not runs:
            continue

        summary[ds] = {
            'num_runs': len(runs),
            'dataset_stats': info['dataset_stats'],
            'system_info': info['system_info'],
            'config': info['config'],
            'training_time_mean_seconds': round(mean(info['training_times']), 2)
            if info['training_times'] else 0,
            'training_time_std_seconds': round(
                stdev(info['training_times']) if len(info['training_times']) > 1 else 0,
                2,
            ),
            'best_epoch_mean': round(mean(info['best_epochs']), 1) if info['best_epochs'] else -1,
            'metrics': {},
        }

        for metric in runs[0]:
            if isinstance(runs[0][metric], dict):
                summary[ds]['metrics'][metric] = {}
                for k_val in runs[0][metric]:
                    values = [
                        float(r[metric][k_val])
                        for r in runs
                        if metric in r and k_val in r[metric]
                    ]
                    if values:
                        summary[ds]['metrics'][metric][k_val] = {
                            'mean': round(mean(values), 6),
                            'std': round(stdev(values) if len(values) > 1 else 0, 6),
                            'values': [round(v, 6) for v in values],
                        }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*90}")
    print(f"  EXPERIMENT SUMMARY  (saved to {summary_file})")
    print(f"{'='*90}")

    for ds in summary:
        s = summary[ds]
        print(f"\n  Dataset: {ds} ({s['num_runs']} runs)")
        print(f"  Training time: {s['training_time_mean_seconds']:.0f}s "
              f"± {s['training_time_std_seconds']:.0f}s")
        print(f"  Best epoch (mean): {s['best_epoch_mean']}")
        print(f"  {'Metric':<15} {'K=5':>15} {'K=10':>15} {'K=20':>15}")
        print(f"  {'-'*60}")
        for metric in ['recall', 'precision', 'ndcg', 'mrr', 'hit_rate', 'map']:
            if metric in s['metrics']:
                vals = s['metrics'][metric]
                parts = []
                for k_val in ['5', '10', '20']:
                    if k_val in vals:
                        parts.append(f"{vals[k_val]['mean']:.4f}±{vals[k_val]['std']:.4f}")
                    else:
                        parts.append('    N/A     ')
                print(f"  {metric:<15} {parts[0]:>15} {parts[1]:>15} {parts[2]:>15}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run DirectAU experiments on all datasets')
    parser.add_argument('--datasets', nargs='+',
                        default=['movielens', 'gowalla', 'yelp2018', 'amazon-book', 'collected'],
                        help='Datasets to run experiments on')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per dataset (1-5)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs for all selected datasets (e.g., --epochs 3 for smoke test)')
    parser.add_argument('--run_ids', nargs='+', type=int, default=None,
                        help='Specific run IDs to execute (0-indexed), e.g. --run_ids 2 3 4')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments that already have result files')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Only aggregate existing results, do not run experiments')
    parser.add_argument('--config_files', type=str,
                        default='recbole/properties/overall.yaml recbole/properties/model/DirectAU.yaml',
                        help='Space-separated RecBole config files relative to models/DirectAU/code')
    args = parser.parse_args()

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError('--epochs must be a positive integer')
        for dataset_name in DATASET_CONFIGS:
            DATASET_CONFIGS[dataset_name]['epochs'] = args.epochs

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve output directory
    if args.output_dir is None:
        output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'results', 'DirectAU'))
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(output_dir)
        return

    config_file_list = args.config_files.strip().split() if args.config_files else []

    num_runs = min(args.num_runs, len(SEEDS))
    # Determine which run IDs to execute
    if args.run_ids is not None:
        run_id_list = [r for r in args.run_ids if 0 <= r < num_runs]
    else:
        run_id_list = list(range(num_runs))

    total_experiments = 0
    failed_experiments = []
    skipped_experiments = 0

    normalized_datasets = [_resolve_dataset_name(ds, script_dir) for ds in args.datasets]

    print(f"\n{'#'*70}")
    print('  DirectAU Experiment Runner')
    print(f"  Datasets: {args.datasets}")
    if normalized_datasets != args.datasets:
        print(f"  Normalized datasets: {normalized_datasets}")
    print(f"  Run IDs: {run_id_list} (seeds: {[SEEDS[i] for i in run_id_list]})")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}\n")

    overall_start = time.time()

    for dataset in normalized_datasets:
        if dataset not in DATASET_CONFIGS:
            print(f"WARNING: Unknown dataset '{dataset}', skipping.")
            continue

        if not check_data_exists(dataset):
            print(
                f"WARNING: No fixed .inter split found for '{dataset}', skipping. "
                "Run: python data/build_recbole_inter_splits.py --dataset "
                f"{dataset}"
            )
            continue

        for run_id in run_id_list:
            seed = SEEDS[run_id]
            result_file = os.path.join(
                output_dir, f'{dataset}_seed{seed}_run{run_id}.json')

            if args.skip_existing and os.path.exists(result_file):
                print(f"Skipping (exists): {result_file}")
                skipped_experiments += 1
                continue

            returncode = run_single_experiment(
                dataset, seed, run_id, output_dir, num_runs, config_file_list)
            total_experiments += 1
            if returncode != 0:
                failed_experiments.append(f"{dataset}_seed{seed}_run{run_id}")

    overall_time = time.time() - overall_start

    # Aggregate results
    print(f"\n{'#'*70}")
    print('  All experiments completed!')
    print(f"  Total: {total_experiments} run, "
          f"{len(failed_experiments)} failed, {skipped_experiments} skipped")
    print(f"  Total time: {overall_time/60:.1f} minutes ({overall_time/3600:.1f} hours)")
    if failed_experiments:
        print(f"  Failed: {failed_experiments}")
    print(f"{'#'*70}")

    aggregate_results(output_dir)


if __name__ == '__main__':
    main()
