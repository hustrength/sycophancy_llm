import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from prompts.prompt_templates import PROMPT_TEMPLATES

YTA_NTA_DATASETS = {"oeq", "aita_yta"}


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
    if dataset == "aita_yta":
        matches = sorted((ROOT / "data" / "raw").glob("AITA-YTA*.csv"))
        if not matches:
            raise FileNotFoundError("Could not find AITA-YTA CSV under data/raw/.")
        return matches[0]
    raise ValueError(f"Unknown dataset '{dataset}'.")


def run_command(command, dry_run=False):
    print("\n$", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def build_generate_command(args, dataset, variant, model_name):
    output_suffix = "_rationale" if (args.oeq_short_rationale and dataset in YTA_NTA_DATASETS) else ""
    output_file = ROOT / "outputs" / "generations" / f"{dataset}_{model_tag(model_name)}{output_suffix}.csv"
    prompt_col = f"{variant}_prompt"
    if args.oeq_short_rationale and dataset in YTA_NTA_DATASETS:
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
    if args.cuda_device is not None:
        command += ["--cuda_device", str(args.cuda_device)]
    if args.max_rows is not None:
        command += ["--max_rows", str(args.max_rows)]
    if args.row_indices is not None:
        command += ["--row_indices", args.row_indices]
    if raw_output_col is not None:
        command += ["--raw_output_column", raw_output_col, "--oeq_short_rationale", "--fail_on_unparsed_label"]
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
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets: ss,oeq,aita_yta or all.")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or all.")
    parser.add_argument(
        "--generation_backend",
        choices=["ollama", "transformers"],
        default="transformers",
        help="Generation backend to use for scripts/02_generate_responses.py. Defaults to Hugging Face transformers.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model identifiers. Examples: Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct for transformers; qwen2.5:7b for Ollama.",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row cap for smoke tests or sampling.")
    parser.add_argument("--row_indices", default=None, help="Optional comma-separated row indices.")
    parser.add_argument("--max_tokens", type=int, default=5, help="Generation max tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional generation top_p.")
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=None,
        help="For generation_backend=transformers only: CUDA device index to pass through to scripts/02_generate_responses.py.",
    )
    parser.add_argument(
        "--oeq_short_rationale",
        action="store_true",
        help="For YTA/NTA datasets (oeq, aita_yta), request a one-sentence justification followed by a final YTA/NTA line and save both raw output and final label.",
    )
    parser.add_argument("--run_judge", action="store_true", help="Also run overall sycophancy judging after generation.")
    parser.add_argument("--judge_backend", choices=["openai", "ollama", "gemini"], default="openai")
    parser.add_argument("--judge_model", default=None, help="Optional override for judge model.")
    parser.add_argument("--judge_max_tokens", type=int, default=2, help="Judge max tokens.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    main(parser.parse_args())
