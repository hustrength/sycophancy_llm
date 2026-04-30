import argparse, os, subprocess, sys, time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from prompts.prompt_templates import PROMPT_TEMPLATES


def parse_list(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated list.")
    return items


def expand_requested(value, valid_items):
    requested = parse_list(value)
    if requested == ["all"]:
        return list(valid_items)
    invalid = [item for item in requested if item not in valid_items]
    if invalid:
        raise ValueError(f"Unknown items: {invalid}. Valid options: {list(valid_items)}")
    return requested


def model_tag(model_name):
    return model_name.replace(":", "_").replace("/", "_")


def resolve_input_file(dataset):
    if dataset == "ss":
        return ROOT / "data" / "raw" / "SS.csv"
    if dataset == "oeq":
        matches = sorted((ROOT / "data" / "raw").glob("OEQ*.csv"))
        if not matches:
            raise FileNotFoundError("Could not find OEQ CSV under data/raw/.")
        return matches[0]
    raise ValueError(f"Unknown dataset '{dataset}'.")


def run_command(command, dry_run=False):
    print("\n$", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


OLLAMA_LOG = ROOT / "logs" / "ollama_serve.log"


def restart_ollama_serve(concurrency, ready_timeout=30):
    print("[ollama] --restart_ollama: stopping existing 'ollama' processes (pkill -x ollama)...")
    print("[ollama] If Ollama.app is running on macOS, quit it from the menu bar first or it may respawn the daemon.")
    subprocess.run(["pkill", "-x", "ollama"], check=False)
    time.sleep(1.5)

    OLLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(OLLAMA_LOG, "a")

    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(concurrency)

    print(f"[ollama] Launching 'ollama serve' with OLLAMA_NUM_PARALLEL={concurrency} (logs: {OLLAMA_LOG})...")
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'ollama' CLI was not found on PATH. Install Ollama from https://ollama.com."
        ) from exc

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
                if resp.status == 200:
                    print(f"[ollama] Server is ready (pid={proc.pid}). It will keep running after this script exits.")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)

    raise RuntimeError(
        f"ollama serve did not become ready within {ready_timeout}s. Check the log at {OLLAMA_LOG}."
    )


def ensure_ollama_parallel(concurrency):
    raw_current = os.environ.get("OLLAMA_NUM_PARALLEL")
    try:
        current_int = int(raw_current) if raw_current is not None else None
    except ValueError:
        current_int = None

    os.environ["OLLAMA_NUM_PARALLEL"] = str(concurrency)

    if current_int is not None and current_int >= concurrency:
        print(f"[ollama] OLLAMA_NUM_PARALLEL={current_int} (>= --concurrency={concurrency}).")
        return

    if raw_current is None:
        print(
            f"[ollama] OLLAMA_NUM_PARALLEL is unset in this shell; setting it to {concurrency} "
            "for child processes spawned by this script.\n"
            "[ollama] NOTE: an already-running 'ollama serve' will NOT pick this up. "
            f"For real parallelism, restart it with:  OLLAMA_NUM_PARALLEL={concurrency} ollama serve"
        )
    else:
        print(
            f"[ollama] OLLAMA_NUM_PARALLEL was '{raw_current}', overriding to {concurrency} "
            "for child processes spawned by this script.\n"
            f"[ollama] Restart 'ollama serve' with OLLAMA_NUM_PARALLEL={concurrency} for it to take effect."
        )


def ensure_ollama_models(models, dry_run=False):
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The 'ollama' CLI was not found on PATH. Install Ollama from https://ollama.com "
            "or use --generation_backend transformers."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Failed to query Ollama (is the Ollama service running? Try `ollama serve`). "
            f"`ollama list` exited with status {exc.returncode}: {stderr}"
        ) from exc

    available = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            available.add(parts[0])

    for model_name in models:
        if model_name in available:
            print(f"[ollama] Model '{model_name}' already available.")
            continue
        print(f"[ollama] Model '{model_name}' not found locally. Pulling...")
        if dry_run:
            print(f"  (dry run) would execute: ollama pull {model_name}")
            continue
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"`ollama pull {model_name}` failed with status {exc.returncode}. "
                "Check the model name at https://ollama.com/library."
            ) from exc


def build_generate_command(args, dataset, variant, model_name):
    output_suffix = "_rationale" if (args.oeq_short_rationale and dataset == "oeq") else ""
    output_file = ROOT / "outputs" / "generations" / f"{dataset}_{model_tag(model_name)}{output_suffix}.csv"
    prompt_col = f"{variant}_prompt"
    if args.oeq_short_rationale and dataset == "oeq":
        response_col = f"{model_tag(model_name)}_{variant}_final_label"
        raw_output_col = f"{model_tag(model_name)}_{variant}_raw_output"
    else:
        response_col = f"{model_tag(model_name)}_{variant}_output"
        raw_output_col = None
    command = [
        sys.executable,
        str(ROOT / "scripts" / "02_generate_responses.py"),
        "--input_file", str(resolve_input_file(dataset)),
        "--dataset", dataset,
        "--variant", variant,
        "--template_output_column", prompt_col,
        "--output_column", response_col,
        "--output_file", str(output_file),
        "--backend", args.generation_backend,
        "--model", model_name,
        "--max_tokens", str(args.max_tokens),
    ]
    if args.temperature is not None:
        command += ["--temperature", str(args.temperature)]
    if args.top_p is not None:
        command += ["--top_p", str(args.top_p)]
    if args.max_rows is not None:
        command += ["--max_rows", str(args.max_rows)]
    if args.row_indices is not None:
        command += ["--row_indices", args.row_indices]
    if raw_output_col is not None:
        command += ["--raw_output_column", raw_output_col, "--oeq_short_rationale", "--fail_on_unparsed_label"]
    if args.concurrency is not None and args.concurrency > 1:
        command += ["--concurrency", str(args.concurrency)]
    return command, output_file, prompt_col, response_col, raw_output_col


def build_judge_command(args, dataset, variant, model_name, gen_file, prompt_col, response_col, raw_output_col=None):
    judge_file = ROOT / "outputs" / "judgments" / f"{dataset}_{model_tag(model_name)}_{variant}_judged.csv"
    judge_response_col = raw_output_col or response_col
    command = [
        sys.executable,
        str(ROOT / "scripts" / "03_sycophancy_evaluate.py"),
        "--input_file", str(gen_file),
        "--prompt_column", prompt_col,
        "--response_column", judge_response_col,
        "--dataset", dataset,
        "--output_file", str(judge_file),
        "--overall",
        "--max_tokens", str(args.judge_max_tokens),
        "--output_column_tag", f"{model_tag(model_name)}_{variant}",
    ]
    if args.judge_backend == "ollama":
        command.append("--ollama")
    elif args.judge_backend == "gemini":
        command.append("--gemini")
    if args.judge_model:
        command += ["--judge_model", args.judge_model]
    return command


def main(args):
    datasets = expand_requested(args.datasets, PROMPT_TEMPLATES.keys())
    models = parse_list(args.models)
    if args.generation_backend == "ollama":
        if args.restart_ollama:
            restart_ollama_serve(max(1, args.concurrency))
        elif args.concurrency > 1:
            ensure_ollama_parallel(args.concurrency)
        ensure_ollama_models(models, dry_run=args.dry_run)
    for dataset in datasets:
        variants = expand_requested(args.variants, PROMPT_TEMPLATES[dataset].keys())
        for model_name in models:
            print(f"\n=== dataset={dataset} model={model_name} ===")
            for variant in variants:
                gen_cmd, gen_file, prompt_col, response_col, raw_output_col = build_generate_command(args, dataset, variant, model_name)
                run_command(gen_cmd, dry_run=args.dry_run)
                if args.run_judge:
                    judge_cmd = build_judge_command(args, dataset, variant, model_name, gen_file, prompt_col, response_col, raw_output_col)
                    run_command(judge_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run prompt sensitivity experiments across datasets, variants, and models.")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets: ss,oeq or all.")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or all.")
    parser.add_argument(
        "--generation_backend",
        choices=["ollama", "transformers"],
        default="ollama",
        help="Generation backend to use for scripts/02_generate_responses.py.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model identifiers. Examples: qwen2.5:7b for Ollama, Qwen/Qwen2.5-14B-Instruct for transformers.",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row cap for smoke tests or sampling.")
    parser.add_argument("--row_indices", default=None, help="Optional comma-separated row indices.")
    parser.add_argument("--max_tokens", type=int, default=5, help="Generation max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional generation top_p.")
    parser.add_argument(
        "--oeq_short_rationale",
        action="store_true",
        help="For dataset=oeq, request a one-sentence justification followed by a final YTA/NTA line and save both raw output and final label.",
    )
    parser.add_argument("--run_judge", action="store_true", help="Also run overall sycophancy judging after generation.")
    parser.add_argument("--judge_backend", choices=["openai", "ollama", "gemini"], default="openai")
    parser.add_argument("--judge_model", default=None, help="Optional override for judge model.")
    parser.add_argument("--judge_max_tokens", type=int, default=2, help="Judge max tokens.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent in-flight requests per generation job (Ollama backend only). Should match OLLAMA_NUM_PARALLEL.",
    )
    parser.add_argument(
        "--restart_ollama",
        action="store_true",
        help="Stop existing 'ollama' processes and restart 'ollama serve' as a background process with OLLAMA_NUM_PARALLEL=--concurrency. The new server keeps running after this script exits.",
    )
    main(parser.parse_args())
