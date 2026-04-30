import argparse, json, os, shutil, subprocess, sys, tempfile, time
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


LOG_DIR = ROOT / "logs"


def detect_total_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def detect_numa_topology():
    """Return [{'node': int, 'cpus': [...], 'phys_cpus': [...]}] sorted by node id.
    Empty list if /sys/devices/system/node is unavailable (non-Linux or no NUMA)."""
    base = Path("/sys/devices/system/node")
    if not base.exists():
        return []
    nodes = []
    for node_dir in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        try:
            node_id = int(node_dir.name[4:])
        except ValueError:
            continue
        cpus = sorted(
            int(p.name[3:]) for p in node_dir.glob("cpu[0-9]*") if p.name[3:].isdigit()
        )
        if not cpus:
            continue
        seen_siblings = set()
        phys_cpus = []
        for cpu in cpus:
            sib_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            try:
                sibs = sib_path.read_text().strip()
            except OSError:
                sibs = str(cpu)
            if sibs in seen_siblings:
                continue
            seen_siblings.add(sibs)
            phys_cpus.append(cpu)
        nodes.append({"node": node_id, "cpus": cpus, "phys_cpus": phys_cpus})
    return nodes


def wait_for_ollama_ready(port, log_path, ready_timeout):
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    raise RuntimeError(
        f"ollama serve on port {port} did not become ready within {ready_timeout}s. "
        f"Check the log at {log_path}."
    )


def restart_ollama_serve(concurrency, num_instances=1, base_port=11434, cores_per_instance=None, ready_timeout=60, use_numa=True):
    print("[ollama] --restart_ollama: stopping existing 'ollama' processes (pkill -x ollama)...")
    print("[ollama] If Ollama.app is running on macOS, quit it from the menu bar first or it may respawn the daemon.")
    subprocess.run(["pkill", "-x", "ollama"], check=False)
    time.sleep(1.5)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_cores = detect_total_cores()
    per_inst_parallel = max(1, concurrency // num_instances)
    has_numactl = shutil.which("numactl") is not None
    has_taskset = shutil.which("taskset") is not None
    numa_nodes = detect_numa_topology() if use_numa else []

    numa_mode = None
    if use_numa and has_numactl and numa_nodes:
        if num_instances == 1 and len(numa_nodes) > 1:
            numa_mode = "interleave"
        elif num_instances <= len(numa_nodes):
            numa_mode = "bind"
    if use_numa and not has_numactl and len(numa_nodes) > 1:
        print("[ollama] NOTE: multiple NUMA nodes detected but 'numactl' not on PATH; install it for better locality (apt-get install numactl).")

    if numa_mode == "bind":
        print(f"[ollama] NUMA-aware: binding {num_instances} instance(s) to {num_instances} of {len(numa_nodes)} NUMA node(s) via numactl.")
    elif numa_mode == "interleave":
        print(f"[ollama] NUMA-aware: single instance with --interleave=all across {len(numa_nodes)} NUMA nodes.")
    elif num_instances > 1 and not has_taskset:
        print("[ollama] WARNING: 'taskset' not found on PATH; instances will not be core-pinned and may contend for cores.")

    endpoints = []
    procs = []
    for i in range(num_instances):
        port = base_port + i

        if numa_mode == "bind":
            node = numa_nodes[i]
            phys_count = len(node["phys_cpus"]) or len(node["cpus"])
            inst_threads = cores_per_instance if cores_per_instance is not None else phys_count
            cmd_prefix = ["numactl", f"--cpunodebind={node['node']}", f"--membind={node['node']}"]
            pin_msg = f"NUMA node {node['node']} (phys cores={phys_count})"
        elif numa_mode == "interleave":
            phys_total = sum(len(n["phys_cpus"]) for n in numa_nodes) or total_cores
            inst_threads = cores_per_instance if cores_per_instance is not None else phys_total
            cmd_prefix = ["numactl", "--interleave=all"]
            pin_msg = f"NUMA interleave=all (phys cores={phys_total})"
        else:
            inst_threads = cores_per_instance if cores_per_instance is not None else max(1, total_cores // num_instances)
            core_start = i * inst_threads
            core_end = min(total_cores - 1, core_start + inst_threads - 1)
            core_range = f"{core_start}-{core_end}"
            if has_taskset and num_instances > 1:
                cmd_prefix = ["taskset", "-c", core_range]
                pin_msg = f"cores {core_range}"
            else:
                cmd_prefix = []
                pin_msg = "no core pinning"

        env = os.environ.copy()
        env["OLLAMA_NUM_PARALLEL"] = str(per_inst_parallel)
        env["OLLAMA_NUM_THREAD"] = str(inst_threads)
        env["OMP_NUM_THREADS"] = str(inst_threads)
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        env.setdefault("OLLAMA_KEEP_ALIVE", "24h")

        log_path = LOG_DIR / f"ollama_serve_{port}.log"
        log_file = open(log_path, "a")

        cmd = cmd_prefix + ["ollama", "serve"]

        print(
            f"[ollama] Starting instance {i + 1}/{num_instances} on port {port} ({pin_msg}), "
            f"OLLAMA_NUM_PARALLEL={per_inst_parallel}, OLLAMA_NUM_THREAD={inst_threads} (log: {log_path})..."
        )
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The 'ollama' CLI (or numactl/taskset) was not found on PATH. Install Ollama from https://ollama.com."
            ) from exc

        procs.append((port, proc, log_path))

    for port, proc, log_path in procs:
        wait_for_ollama_ready(port, log_path, ready_timeout)
        print(f"[ollama] Instance on port {port} ready (pid={proc.pid}).")
        endpoints.append(f"http://127.0.0.1:{port}/v1")

    os.environ["OLLAMA_HOST"] = f"127.0.0.1:{base_port}"
    print(f"[ollama] All {num_instances} instance(s) up. Servers will keep running after this script exits.")
    return endpoints


def prewarm_ollama_endpoints(endpoints, models, prewarm_timeout=600):
    if not endpoints or not models:
        return
    print(
        f"[ollama] Pre-warming {len(endpoints)} endpoint(s) x {len(models)} model(s) sequentially "
        "to avoid concurrent model-load memory spikes..."
    )
    for endpoint in endpoints:
        for model in models:
            url = f"{endpoint.rstrip('/')}/chat/completions"
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
                method="POST",
            )
            print(f"[ollama]   loading {model} on {endpoint} ...", end="", flush=True)
            start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=prewarm_timeout) as resp:
                    resp.read()
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                print(" FAILED")
                raise RuntimeError(
                    f"Pre-warm request to {endpoint} for model '{model}' failed: {exc}"
                ) from exc
            print(f" done in {time.time() - start:.1f}s")


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


MMAP_SUFFIX = "-mmap"


def _list_ollama_models():
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
    return available


def create_mmap_variant(base_model, mmap_model, dry_run=False):
    print(f"[ollama] Creating mmap variant '{mmap_model}' from '{base_model}' (PARAMETER use_mmap true)...")
    if dry_run:
        print(f"  (dry run) would execute: ollama create {mmap_model} -f <Modelfile>")
        return
    modelfile_content = f"FROM {base_model}\nPARAMETER use_mmap true\n"
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as tf:
        tf.write(modelfile_content)
        modelfile_path = tf.name
    try:
        subprocess.run(
            ["ollama", "create", mmap_model, "-f", modelfile_path],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"`ollama create {mmap_model}` failed with status {exc.returncode}."
        ) from exc
    finally:
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass


def ensure_ollama_models(models, dry_run=False, auto_mmap=False):
    available = _list_ollama_models()
    final_models = []
    for model_name in models:
        ends_with_mmap = model_name.endswith(MMAP_SUFFIX)
        base = model_name[: -len(MMAP_SUFFIX)] if ends_with_mmap else model_name

        if base in available:
            print(f"[ollama] Model '{base}' already available.")
        else:
            print(f"[ollama] Model '{base}' not found locally. Pulling...")
            if dry_run:
                print(f"  (dry run) would execute: ollama pull {base}")
            else:
                try:
                    subprocess.run(["ollama", "pull", base], check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"`ollama pull {base}` failed with status {exc.returncode}. "
                        "Check the model name at https://ollama.com/library."
                    ) from exc
                available.add(base)

        want_mmap = ends_with_mmap or auto_mmap
        if not want_mmap:
            final_models.append(base)
            continue

        mmap_name = base + MMAP_SUFFIX
        if mmap_name in available:
            print(f"[ollama] mmap variant '{mmap_name}' already available.")
        else:
            create_mmap_variant(base, mmap_name, dry_run=dry_run)
            if not dry_run:
                available.add(mmap_name)
        final_models.append(mmap_name)

    return final_models


def build_generate_command(args, dataset, variant, model_name, ollama_endpoints=None):
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
    if ollama_endpoints:
        command += ["--ollama_endpoints", ",".join(ollama_endpoints)]
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
    ollama_endpoints = None
    if args.generation_backend == "ollama":
        if args.num_ollama_instances > 1 and args.concurrency < args.num_ollama_instances:
            print(
                f"[warn] --concurrency={args.concurrency} < --num_ollama_instances={args.num_ollama_instances}; "
                "some instances will receive no traffic. Increase --concurrency."
            )
        if args.restart_ollama:
            ollama_endpoints = restart_ollama_serve(
                max(1, args.concurrency),
                num_instances=args.num_ollama_instances,
                base_port=args.ollama_base_port,
                cores_per_instance=args.cores_per_instance,
                use_numa=not args.no_numa,
            )
        elif args.num_ollama_instances > 1:
            ollama_endpoints = [
                f"http://127.0.0.1:{args.ollama_base_port + i}/v1"
                for i in range(args.num_ollama_instances)
            ]
            print(f"[ollama] Using existing endpoints (no restart): {ollama_endpoints}")
        elif args.concurrency > 1:
            ensure_ollama_parallel(args.concurrency)
        models = ensure_ollama_models(models, dry_run=args.dry_run, auto_mmap=args.auto_mmap)
        if ollama_endpoints and not args.no_prewarm and not args.dry_run:
            prewarm_ollama_endpoints(ollama_endpoints, models)
    for dataset in datasets:
        variants = expand_requested(args.variants, PROMPT_TEMPLATES[dataset].keys())
        for model_name in models:
            print(f"\n=== dataset={dataset} model={model_name} ===")
            for variant in variants:
                gen_cmd, gen_file, prompt_col, response_col, raw_output_col = build_generate_command(
                    args, dataset, variant, model_name, ollama_endpoints=ollama_endpoints
                )
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
    parser.add_argument(
        "--num_ollama_instances",
        type=int,
        default=1,
        help="Number of Ollama server instances to launch (with --restart_ollama) and round-robin across. Useful on multi-core CPU-only machines to avoid memory-bandwidth contention.",
    )
    parser.add_argument(
        "--cores_per_instance",
        type=int,
        default=None,
        help="Threads per Ollama instance (OLLAMA_NUM_THREAD). Defaults: physical cores per NUMA node when NUMA-bound, otherwise total_cores // num_ollama_instances.",
    )
    parser.add_argument(
        "--no_numa",
        action="store_true",
        help="Disable numactl-based NUMA pinning. By default, when numactl is available and num_ollama_instances <= NUMA node count, each instance is pinned to one NUMA node (--cpunodebind+--membind); a single instance uses --interleave=all across all nodes.",
    )
    parser.add_argument(
        "--ollama_base_port",
        type=int,
        default=11434,
        help="Base port for Ollama instances. Instance i listens on base_port + i.",
    )
    parser.add_argument(
        "--no_prewarm",
        action="store_true",
        help="Skip sequential model pre-warm of Ollama endpoints. Pre-warm is on by default when multiple endpoints are used; it sends one tiny request per (endpoint, model) sequentially to force model load and avoid concurrent load memory spikes.",
    )
    parser.add_argument(
        "--auto_mmap",
        action="store_true",
        help="After pulling a model, also create an '<model>-mmap' derivative (FROM <base> + PARAMETER use_mmap true) and use it downstream so multiple Ollama instances can share weight pages via mmap. Off by default. Note: a model name explicitly ending in '-mmap' always triggers creation regardless of this flag.",
    )
    main(parser.parse_args())
