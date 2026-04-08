import ast
import json
import re
import sys
from pathlib import Path

ANSI_ESCAPE_PATTERN = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')


def safe_console_text(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(text):
    print(safe_console_text(text))


def clean_log_file(log_file_path):
    try:
        content = log_file_path.read_text(encoding='utf-8', errors='ignore')
        cleaned_content = ANSI_ESCAPE_PATTERN.sub('', content)

        if cleaned_content != content:
            log_file_path.write_text(cleaned_content, encoding='utf-8')
            return 'cleaned'
        return 'already_clean'
    except Exception as err:
        return f'error: {err}'


def clean_log_files(log_dir):
    log_files = sorted(log_dir.rglob('*.log'))
    if not log_files:
        safe_print(f"Khong tim thay file .log trong thu muc: {log_dir}")
        return []

    safe_print(f"Tim thay {len(log_files)} file .log de clean...")
    for log_file in log_files:
        status = clean_log_file(log_file)
        relative_path = log_file.relative_to(log_dir)
        if status == 'cleaned':
            safe_print(f"Da clean: {relative_path}")
        elif status == 'already_clean':
            safe_print(f"Da sach san: {relative_path}")
        else:
            safe_print(f"Loi khi xu ly {relative_path}: {status}")
    return log_files

def parse_recbole_log_to_json(log_content):
    def to_value(raw):
        raw = raw.strip()
        if raw in ("None", "none", "null"):
            return None
        if raw in ("True", "False"):
            return raw == "True"
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if re.fullmatch(r"-?\d+\.\d+(e[+-]?\d+)?", raw, re.IGNORECASE):
            return float(raw)
        if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
            try:
                return ast.literal_eval(raw)
            except Exception:
                return raw
        return raw

    def parse_config(content):
        cfg = {}
        section_block_re = re.compile(
            r"([A-Za-z ]+Hyper Parameters:)\s*\n(.*?)(?=\n\n[A-Za-z ]+Hyper Parameters:|\n\n[A-Z][a-z]{2}\s\d{2}\s[A-Za-z]{3}\s\d{4}|\Z)",
            re.DOTALL,
        )

        for _, block in section_block_re.findall(content):
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or " = " not in line:
                    continue
                k, v = line.split(" = ", 1)
                cfg[k.strip()] = to_value(v)

        return cfg

    def parse_dataset_stats(content):
        stats = {}
        m_users = re.search(r"The number of users:\s*(\d+)", content)
        m_items = re.search(r"The number of items:\s*(\d+)", content)
        m_inters = re.search(r"The number of inters:\s*(\d+)", content)

        if m_users:
            stats["n_users"] = int(m_users.group(1))
        if m_items:
            stats["n_items"] = int(m_items.group(1))
        if m_inters:
            stats["train_interactions"] = int(m_inters.group(1))
        return stats

    def parse_epoch_logs(content):
        logs = []
        train_re = re.compile(
            r"epoch\s+(\d+)\s+training\s+\[time:\s*([0-9.]+)s,\s*train loss:\s*([-0-9.]+)(?:,\s*gpu_memory_peak_MB:\s*([0-9.]+))?\]",
            re.IGNORECASE
        )
        for m in train_re.finditer(content):
            logs.append({
                "epoch": int(m.group(1)),
                "loss": float(m.group(3)),
                "epoch_time_seconds": float(m.group(2)),
                "gpu_memory_peak_MB": float(m.group(4)) if m.group(4) is not None else None
            })
        return logs

    def parse_metric_pairs(text):
        metrics = {
            "precision": {},
            "recall": {},
            "ndcg": {},
            "mrr": {},
            "hit_rate": {},
            "map": {}
        }
        pair_re = re.compile(
            r"['\"]?([a-zA-Z_]+)@(\d+)['\"]?\s*:\s*(?:np\.float64\()?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\)?"
        )
        for name, k, v in pair_re.findall(text):
            name_l = name.lower()
            val = float(v)
            if name_l == "recall":
                metrics["recall"][k] = val
            elif name_l == "ndcg":
                metrics["ndcg"][k] = val
            elif name_l == "mrr":
                metrics["mrr"][k] = val
            elif name_l in ("hit", "hit_rate"):
                metrics["hit_rate"][k] = val
            elif name_l == "precision":
                metrics["precision"][k] = val
            elif name_l == "map":
                metrics["map"][k] = val
        return metrics

    def parse_valid_results(content):
        results = []
        eval_re = re.compile(
            r"epoch\s+(\d+)\s+evaluating\s+\[time:\s*([0-9.]+)s,\s*valid_score:\s*([0-9.]+)\]\s*"
            r"\n.*?valid result:\s*\n([^\n]+)",
            re.IGNORECASE
        )
        for m in eval_re.finditer(content):
            epoch = int(m.group(1))
            eval_time = float(m.group(2))
            metric_line = m.group(4)
            metrics = parse_metric_pairs(metric_line)
            results.append({
                "epoch": epoch,
                "test_time_seconds": eval_time,
                "metrics": metrics
            })
        return results

    def parse_best_epoch(content):
        m = re.search(r"Finished training,\s*best eval result in epoch\s+(\d+)", content)
        return int(m.group(1)) if m else None

    def parse_best_valid_metrics(content):
        m = re.search(r"best valid\s*:\s*(\{.*\})", content)
        if not m:
            return None
        return parse_metric_pairs(m.group(1))

    def parse_system_info(content, config):
        gpu_name_match = re.search(r"GPU NAME:\s*(.+)", content)
        gpu_memory_match = re.search(r"GPU MEMORY TOTAL MB:\s*([0-9.]+)", content)

        device = config.get("device")
        if not device:
            device = "cuda" if str(config.get("use_gpu", "")).lower() == "true" else "cpu"

        return {
            "device": device if device else "cpu",
            "gpu_name": gpu_name_match.group(1).strip() if gpu_name_match else None,
            "gpu_memory_total_MB": float(gpu_memory_match.group(1)) if gpu_memory_match else None,
        }

    def parse_final_test_result(content, best_epoch):
        m = re.search(r"test result:\s*(\{.*\})", content)
        if not m:
            return None
        metrics = parse_metric_pairs(m.group(1))
        return {
            "epoch": best_epoch if best_epoch is not None else 0,
            "metrics": metrics
        }

    config = parse_config(log_content)

    data = {}
    data["config"] = config

    dataset = None
    if isinstance(config.get("data_path"), str):
        dataset = config["data_path"].split("/")[-1].split("\\")[-1]
    if dataset is None:
        m_ds = re.search(r"INFO\s+([A-Za-z0-9_-]+)\s*\nThe number of users:", log_content)
        dataset = m_ds.group(1) if m_ds else None
    data["dataset"] = dataset

    m_model = re.search(r"INFO\s+([A-Za-z0-9_]+)\(", log_content)
    data["model"] = m_model.group(1) if m_model else None

    data["seed"] = int(config.get("seed", 2022))

    topk = config.get("topk", [5, 10, 20])
    if isinstance(topk, list):
        data["topks"] = topk
    else:
        data["topks"] = [5, 10, 20]

    data["dataset_stats"] = parse_dataset_stats(log_content)
    data["epoch_logs"] = parse_epoch_logs(log_content)

    valid_results = parse_valid_results(log_content)
    best_epoch = parse_best_epoch(log_content)
    best_valid_metrics = parse_best_valid_metrics(log_content)
    final_test = parse_final_test_result(log_content, best_epoch)

    test_results = valid_results[:]
    if final_test is not None:
        test_results.append(final_test)
    data["test_results"] = test_results

    data["best_results"] = {
        "epoch": best_epoch,
        "metrics": best_valid_metrics if best_valid_metrics else (final_test["metrics"] if final_test else {}),
        "test_metrics": final_test["metrics"] if final_test else {},
        "best_epoch": best_epoch
    } if best_epoch is not None else {}

    total_train_time = sum(x["epoch_time_seconds"] for x in data["epoch_logs"])
    data["total_train_time_seconds"] = round(total_train_time, 2) if total_train_time > 0 else None

    data["system_info"] = parse_system_info(log_content, config)

    return data


def convert_log_file_to_json(log_file_path):
    json_output_path = log_file_path.with_suffix('.json')
    content = log_file_path.read_text(encoding='utf-8')
    data_dict = parse_recbole_log_to_json(content)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    safe_print(f"Chuyen doi thanh cong: {log_file_path.name} -> {json_output_path.name}")

# --- Cách sử dụng ---
if __name__ == '__main__':
    log_dir = Path(r'models\DirectAU\code\log')

    try:
        if not log_dir.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc log: '{log_dir}'")

        log_files = clean_log_files(log_dir)
        if not log_files:
            safe_print("Khong co file .log de chuyen doi.")
            sys.exit(0)

        safe_print("\nBat dau chuyen doi log sang JSON...")
        converted_count = 0
        for log_file in log_files:
            try:
                convert_log_file_to_json(log_file)
                converted_count += 1
            except Exception as e:
                safe_print(f"Loi khi chuyen doi {log_file.name}: {e}")

        safe_print(f"\nHoan tat: da chuyen doi {converted_count}/{len(log_files)} file .log.")

    except FileNotFoundError:
        safe_print(f"Loi: Khong tim thay file/thu muc log tai '{log_dir}'")
    except Exception as e:
        safe_print(f"Da xay ra loi trong qua trinh xu ly: {e}")
