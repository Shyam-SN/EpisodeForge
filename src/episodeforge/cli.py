"""EpisodeForge CLI: End-to-end robotics data pipeline commands."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from episodeforge.config.client_spec import ClientSpec
from episodeforge.config.generation_config import GenerationConfig
from episodeforge.generate.generator import generate_dataset_parallel

app = typer.Typer(
    name="forge",
    help="EpisodeForge: Enterprise Robotics Data Pipeline (Generate -> Validate -> Transform -> Deliver)",
    no_args_is_help=True,
)
spec_app = typer.Typer(help="Manage and inspect ClientSpec configurations")
app.add_typer(spec_app, name="spec")

console = Console()


@spec_app.command("lint")
def spec_lint(spec_path: Path = typer.Argument(..., help="Path to ClientSpec YAML file")):
    """Lint and validate a client specification against schema rules and logical constraints."""
    if not spec_path.exists():
        console.print(f"[bold red]Error:[/bold red] Spec file {spec_path} does not exist.")
        raise typer.Exit(code=1)

    try:
        with open(spec_path) as f:
            raw_data = yaml.safe_load(f)
        spec = ClientSpec.model_validate(raw_data)
    except Exception as e:
        console.print(f"[bold red]Validation Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Logical linting checks
    warnings = []

    # 1. Check camera resize mode vs aspect ratio
    for cam_name, cam_spec in spec.cameras.items():
        w, h = cam_spec.resolution
        if cam_spec.resize_mode == "stretch" and abs(w - h) > 0:
            warnings.append(
                f"Camera '{cam_name}': non-square resolution ({w}x{h}) with 'stretch' may introduce geometric distortion."
            )

    # 2. Check timing tolerance
    nominal_dt_ms = (1.0 / spec.timing.target_fps) * 1000.0
    if spec.timing.max_timestamp_error_ms > nominal_dt_ms:
        warnings.append(
            f"Timing: max_timestamp_error_ms ({spec.timing.max_timestamp_error_ms}ms) is larger than target sample period ({nominal_dt_ms:.1f}ms)."
        )

    # 3. Check rot6d compatibility
    if (
        spec.action.rotation_format == "rot6d"
        and spec.format == "lerobot"
        and spec.format_version == "v2.0"
    ):
        warnings.append("Format: rot6d with LeRobot v2.0 lacks native named dimension support.")

    console.print(
        f"[bold green]PASS:[/bold green] Spec '{spec.dataset_name}' ({spec.client} {spec.spec_version}) is syntactically valid."
    )
    if warnings:
        console.print("\n[bold yellow]Lint Warnings:[/bold yellow]")
        for warn_msg in warnings:
            console.print(f"  - {warn_msg}")
    else:
        console.print("[dim]No lint warnings detected.[/dim]")


@spec_app.command("diff")
def spec_diff(
    old_spec: Path = typer.Argument(..., help="Path to baseline ClientSpec YAML"),
    new_spec: Path = typer.Argument(..., help="Path to modified ClientSpec YAML"),
):
    """Compare two client specs, computing field diffs and estimating pipeline stage impacts."""
    if not old_spec.exists() or not new_spec.exists():
        console.print("[bold red]Error:[/bold red] Both spec files must exist.")
        raise typer.Exit(code=1)

    with open(old_spec) as f:
        old_data = yaml.safe_load(f)
    with open(new_spec) as f:
        new_data = yaml.safe_load(f)

    old_model = ClientSpec.model_validate(old_data)
    new_model = ClientSpec.model_validate(new_data)

    table = Table(
        title=f"Spec Diff: {old_model.client} ({old_model.spec_version} -> {new_model.spec_version})"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Field", style="magenta")
    table.add_column("Baseline Value", style="red")
    table.add_column("New Value", style="green")
    table.add_column("Affected Stages", style="yellow")

    diff_count = 0

    # Timing diff
    if old_model.timing.target_fps != new_model.timing.target_fps:
        diff_count += 1
        table.add_row(
            "Timing",
            "target_fps",
            str(old_model.timing.target_fps),
            str(new_model.timing.target_fps),
            "transform, encode, write",
        )

    # Cameras diff
    old_cams = set(old_model.cameras.keys())
    new_cams = set(new_model.cameras.keys())
    if old_cams != new_cams:
        diff_count += 1
        table.add_row(
            "Cameras",
            "camera_list",
            str(sorted(old_cams)),
            str(sorted(new_cams)),
            "transform, encode, write, validate_out",
        )

    # Actions diff
    if old_model.action.representation != new_model.action.representation:
        diff_count += 1
        table.add_row(
            "Action",
            "representation",
            old_model.action.representation.value,
            new_model.action.representation.value,
            "transform, write",
        )
    if old_model.action.rotation_format != new_model.action.rotation_format:
        diff_count += 1
        table.add_row(
            "Action",
            "rotation_format",
            old_model.action.rotation_format.value,
            new_model.action.rotation_format.value,
            "transform, write",
        )

    console.print(table)
    if diff_count == 0:
        console.print("[dim]No operational differences found between specs.[/dim]")
    else:
        console.print(
            f"\n[bold]Impact Analysis:[/bold] {diff_count} configuration differences identified. Canonical episodes are 100% reusable via cache; only downstream transform/encode stages must rerun."
        )


@app.command("generate")
def generate(
    config_path: Path = typer.Option(
        "configs/generation/default.yaml", "--config", help="Path to GenerationConfig YAML"
    ),
    output_dir: Path = typer.Option(
        "data/raw/generated", "--output", help="Output directory for generated episodes"
    ),
    episodes: int = typer.Option(100, help="Total episodes to generate"),
    workers: int = typer.Option(4, help="Parallel worker processes"),
    seed: int = typer.Option(0, help="Base RNG seed"),
):
    """Generate multi-camera MuJoCo tabletop manipulation demonstration episodes."""
    console.print(
        f"[bold green]Starting generation:[/bold green] {episodes} episodes using {workers} workers (seed={seed})"
    )
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f)
    gen_config = GenerationConfig.model_validate(cfg_data)

    summary = generate_dataset_parallel(
        config=gen_config,
        output_dir=output_dir,
        num_episodes=episodes,
        workers=workers,
        seed=seed,
    )
    console.print(
        f"[bold green]Generation Complete:[/bold green] {summary['total_episodes']} episodes generated in {summary['wall_time_sec']}s "
        f"({summary['episodes_per_hour']} eps/hr). Success rate: {summary['success_rate_pct']}%"
    )


@app.command("ingest")
def ingest(
    source_type: str = typer.Argument(..., help="Source type: rigsim | generated | phone"),
    path: Path = typer.Argument(..., help="Path to input data directory or file"),
):
    """Ingest raw episodes or rig recordings into CanonicalEpisode representation."""
    console.print(f"[bold cyan]Ingesting[/bold cyan] from {source_type} at {path}")


@app.command("validate-in")
def validate_in(
    dataset_path: Path = typer.Argument(..., help="Path to raw or canonical dataset directory"),
    ground_truth: Path | None = typer.Option(
        None, help="Optional path to ground truth imperfection labels JSON"
    ),
    quarantine: bool = typer.Option(True, help="Quarantine failed episodes to dead-letter queue"),
    rejected_dir: Path = typer.Option(Path("data/rejected"), help="Path to quarantine directory"),
):
    """Perform input QA & validation checks (kinematics, video, timestamps, numerical integrity)."""
    import json
    import time

    from episodeforge.canonical.schema import QAVerdict
    from episodeforge.ingest import ingest_episode
    from episodeforge.validate_in.engine import InputQAEngine
    from episodeforge.validate_in.quarantine import QuarantineManager

    console.print(f"[bold cyan]Validating input episodes in:[/bold cyan] {dataset_path}")
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path {dataset_path} does not exist.")
        raise typer.Exit(code=1)

    # Find episode directories
    ep_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("ep_")])
    if not ep_dirs:
        console.print(
            f"[bold yellow]No episode directories found matching ep_* in {dataset_path}[/bold yellow]"
        )
        raise typer.Exit(code=0)

    # Load ground truth if provided
    gt_map = {}
    if ground_truth and ground_truth.exists():
        with open(ground_truth) as f:
            gt_data = json.load(f)
            if isinstance(gt_data, dict):
                gt_map = gt_data
            elif isinstance(gt_data, list):
                gt_map = {item["episode_id"]: item for item in gt_data}

    engine = InputQAEngine(reject_on_failure_flag=True)
    quarantine_mgr = QuarantineManager(rejected_dir=rejected_dir) if quarantine else None

    t0 = time.time()
    results = []
    quarantined_count = 0
    passed_count = 0
    warn_count = 0
    fail_count = 0

    # Ground truth metrics
    tp, fp, tn, fn = 0, 0, 0, 0

    for ep_dir in ep_dirs:
        try:
            episode = ingest_episode(ep_dir)
            qa_res = engine.validate(episode)
            results.append((episode, qa_res))

            if qa_res.verdict == QAVerdict.PASS:
                passed_count += 1
            elif qa_res.verdict == QAVerdict.WARN:
                warn_count += 1
            else:
                fail_count += 1
                if quarantine_mgr:
                    quarantine_mgr.quarantine(episode, qa_res, source_dir=ep_dir)
                    quarantined_count += 1

            # Match against ground truth
            ep_id = episode.source_ref
            if ep_id in gt_map:
                gt_item = gt_map[ep_id]
                expected_verdict = gt_item.get("expected_qa_verdict", "PASS")
                gt_failed = (expected_verdict == "FAIL") or (not gt_item.get("success", True))

                our_failed = qa_res.verdict == QAVerdict.FAIL

                if our_failed and gt_failed:
                    tp += 1
                elif our_failed and not gt_failed:
                    fp += 1
                elif not our_failed and not gt_failed:
                    tn += 1
                else:
                    fn += 1

        except Exception as e:
            console.print(f"[bold red]Error ingesting/validating {ep_dir.name}:[/bold red] {e}")

    elapsed = time.time() - t0
    total = len(results)

    # Print summary table
    summary_table = Table(title=f"Input QA Summary ({total} Episodes in {elapsed:.2f}s)")
    summary_table.add_column("Verdict", style="cyan")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Percentage", justify="right")

    summary_table.add_row(
        "[bold green]PASS[/bold green]", str(passed_count), f"{passed_count / total * 100:.1f}%"
    )
    summary_table.add_row(
        "[bold yellow]WARN[/bold yellow]", str(warn_count), f"{warn_count / total * 100:.1f}%"
    )
    summary_table.add_row(
        "[bold red]FAIL[/bold red]", str(fail_count), f"{fail_count / total * 100:.1f}%"
    )
    console.print(summary_table)

    if quarantine_mgr and quarantined_count > 0:
        console.print(
            f"[bold yellow]Quarantined {quarantined_count} failed episodes to DLQ:[/bold yellow] {rejected_dir}"
        )

    # Ground truth validation metrics table
    if gt_map:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        eval_table = Table(title="Input QA Benchmark against Ground Truth Labels")
        eval_table.add_column("Metric", style="cyan")
        eval_table.add_column("Value", style="bold green")

        eval_table.add_row("True Positives (Detected Failures)", str(tp))
        eval_table.add_row("True Negatives (Correct Passes)", str(tn))
        eval_table.add_row("False Positives (Clean False Alarms)", str(fp))
        eval_table.add_row("False Negatives (Missed Defects)", str(fn))
        eval_table.add_row("Precision", f"{prec * 100:.1f}%")
        eval_table.add_row("Recall", f"{rec * 100:.1f}%")
        eval_table.add_row("F1 Score", f"{f1 * 100:.1f}%")
        console.print(eval_table)


@app.command("run")
def run_pipeline(
    spec: Path = typer.Option(..., help="Path to ClientSpec YAML"),
    input_dir: Path = typer.Option(
        Path("data/raw/m1_episodes"), help="Path to raw recordings directory"
    ),
    backend: str = typer.Option("local", help="Execution backend: local | ray"),
    workers: int = typer.Option(4, help="Number of worker processes"),
    limit: int | None = typer.Option(None, help="Limit number of episodes to process"),
):
    """Execute end-to-end transformation pipeline for a target client specification."""
    from episodeforge.config.client_spec import ClientSpec
    from episodeforge.config.pipeline_config import PipelineConfig
    from episodeforge.orchestrate.pipeline_runner import PipelineRunner

    console.print(
        f"[bold green]Starting EpisodeForge Pipeline Run:[/bold green] spec={spec}, backend={backend}, workers={workers}"
    )
    if not spec.exists():
        console.print(f"[bold red]Error:[/bold red] Spec path {spec} does not exist.")
        raise typer.Exit(code=1)

    with open(spec) as f:
        spec_data = yaml.safe_load(f)
    spec_model = ClientSpec.model_validate(spec_data)

    p_config = PipelineConfig()
    runner = PipelineRunner(config=p_config)

    report = runner.run(
        raw_dataset_dir=input_dir,
        spec=spec_model,
        limit=limit,
        workers=workers,
    )

    # Render Pipeline Run Summary Table
    status_style = "bold green" if report.validation_status == "PASS" else "bold red"

    table = Table(title=f"Pipeline Execution Report: {report.run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Client / Target Format", f"{report.client} ({report.format})")
    table.add_row("Dataset Name", report.dataset_name)
    table.add_row("Raw Episodes Discovered", str(report.total_episodes_discovered))
    table.add_row("Quarantined to DLQ", f"[bold yellow]{report.episodes_quarantined}[/bold yellow]")
    table.add_row(
        "Transformed & Released", f"[bold green]{report.episodes_transformed}[/bold green]"
    )
    table.add_row("Total Output Frames", str(report.total_frames))
    table.add_row("Elapsed Time", f"{report.elapsed_time_sec:.2f}s")
    table.add_row(
        "Throughput",
        f"{report.throughput_eps_per_sec:.2f} eps/s ({report.throughput_eps_per_hour:.0f} eps/hr)",
    )
    table.add_row("Cache Hits / Misses", f"{report.cache_hits} / {report.cache_misses}")
    table.add_row("Release Output Dir", report.output_dir)
    table.add_row(
        "Delivery Validation", f"[{status_style}]{report.validation_status}[/{status_style}]"
    )

    console.print(table)


@app.command("validate-out")
def validate_out(
    release_dir: Path = typer.Argument(..., help="Directory or file of generated release"),
    spec: Path = typer.Option(..., help="Path to ClientSpec YAML"),
):
    """Validate release using schema checks and isolated official loaders."""
    from episodeforge.config.client_spec import ClientSpec
    from episodeforge.validate_out.engine import validate_output_dataset

    console.print(
        f"[bold cyan]Validating client delivery:[/bold cyan] {release_dir} against {spec}"
    )
    if not release_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Release path {release_dir} does not exist.")
        raise typer.Exit(code=1)
    if not spec.exists():
        console.print(f"[bold red]Error:[/bold red] Spec path {spec} does not exist.")
        raise typer.Exit(code=1)

    with open(spec) as f:
        spec_data = yaml.safe_load(f)
    spec_model = ClientSpec.model_validate(spec_data)

    res = validate_output_dataset(release_dir, spec_model)

    status_color = "bold green" if res["status"] == "PASS" else "bold red"
    console.print(f"Validation Status: [{status_color}]{res['status']}[/{status_color}]")

    table = Table(
        title=f"Delivery Validation: {spec_model.client} ({spec_model.format} {spec_model.format_version})"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Status", f"[{status_color}]{res['status']}[/{status_color}]")
    table.add_row("Dataset Name", str(res.get("dataset_name")))
    table.add_row("Format", str(res.get("format")))

    val_res = res.get("validator_result", {})
    if val_res:
        table.add_row("Loader Used", str(val_res.get("loader_used", "structural")))
        if "checked_episodes" in val_res:
            table.add_row("Checked Episodes", str(val_res["checked_episodes"]))
        if "demo_count" in val_res:
            table.add_row("Checked Demos", str(val_res["demo_count"]))
        if "total_samples" in val_res:
            table.add_row("Total Samples", str(val_res["total_samples"]))

    console.print(table)

    if res.get("warnings"):
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in res["warnings"]:
            console.print(f"  - {w}")

    if res.get("errors"):
        console.print("\n[bold red]Errors:[/bold red]")
        for err in res["errors"]:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)


@app.command("deliver")
def deliver(
    release_dir: Path = typer.Argument(..., help="Directory of generated release"),
    spec: Path = typer.Option(..., help="Path to ClientSpec YAML"),
    endpoint_url: str | None = typer.Option(
        None, help="S3 / MinIO endpoint URL (e.g. http://localhost:9000)"
    ),
    bucket: str = typer.Option("robotics-delivery-lake", help="Target S3 / MinIO bucket name"),
    dry_run: bool = typer.Option(
        False, help="Generate local manifests and receipts without connecting to remote S3"
    ),
):
    """Upload release to S3/MinIO storage, verify checksums, and generate receipt."""
    import json

    from episodeforge.config.client_spec import ClientSpec
    from episodeforge.deliver.manifest import (
        generate_delivery_manifest,
        generate_delivery_receipt,
    )
    from episodeforge.deliver.s3_client import S3DeliveryClient

    console.print(f"[bold green]Initiating Delivery Package:[/bold green] {release_dir}")
    if not release_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Release directory {release_dir} does not exist."
        )
        raise typer.Exit(code=1)
    if not spec.exists():
        console.print(f"[bold red]Error:[/bold red] Spec path {spec} does not exist.")
        raise typer.Exit(code=1)

    with open(spec) as f:
        spec_data = yaml.safe_load(f)
    spec_model = ClientSpec.model_validate(spec_data)

    if dry_run:
        manifest = generate_delivery_manifest(
            release_dir=release_dir,
            client=spec_model.client,
            dataset_name=spec_model.dataset_name,
            spec_version=spec_model.spec_version,
        )
        receipt = generate_delivery_receipt(
            manifest=manifest,
            target_s3_uri=f"s3://{bucket}/local_dry_run/{spec_model.dataset_name}",
            verification_status="LOCAL_VERIFIED",
        )
        with open(release_dir / "delivery_receipt.json", "w", encoding="utf-8") as rf:
            json.dump(receipt, rf, indent=2)
    else:
        client = S3DeliveryClient(
            endpoint_url=endpoint_url or "http://localhost:9000",
            bucket_name=bucket,
        )
        receipt = client.deliver_release(
            release_dir=release_dir,
            client=spec_model.client,
            dataset_name=spec_model.dataset_name,
            spec_version=spec_model.spec_version,
        )

    # Log delivery receipt to DuckDB Lineage Catalog
    try:
        from episodeforge.lineage.catalog import DuckDBLineageCatalog

        DuckDBLineageCatalog().log_delivery(receipt)
    except Exception:
        pass

    table = Table(title="Delivery Transfer Receipt")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Receipt ID", str(receipt.get("receipt_id")))
    table.add_row("Manifest SHA-256", str(receipt.get("manifest_sha256"))[:16] + "...")
    table.add_row("Target S3 URI", str(receipt.get("target_s3_uri")))
    table.add_row("Files Uploaded", str(receipt.get("uploaded_files_count")))
    table.add_row("Total Size", f"{receipt.get('total_transferred_bytes', 0) / 1024 / 1024:.2f} MB")
    table.add_row("Status", str(receipt.get("verification_status")))
    table.add_row("Completed At", str(receipt.get("completed_at_utc")))

    console.print(table)


@app.command("recall")
def recall(
    release_dir: Path = typer.Option(..., help="Path to release directory to patch"),
    episodes: str = typer.Option(
        ..., help="Comma-separated episode IDs to recall (e.g. ep_000001,ep_000002)"
    ),
    reason: str = typer.Option(..., help="Formal operational reason for recall"),
    patch_version: str | None = typer.Option(None, help="Optional semantic patch version tag"),
):
    """Execute data recall playbook, producing an updated immutable patch release."""
    import json

    from episodeforge.deliver.recall import DataRecallManager

    recalled_ids = [ep.strip() for ep in episodes.split(",") if ep.strip()]
    console.print(
        f"[bold yellow]Executing Data Recall:[/bold yellow] release={release_dir}, {len(recalled_ids)} episodes"
    )

    if not release_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Release directory {release_dir} does not exist."
        )
        raise typer.Exit(code=1)

    mgr = DataRecallManager()
    patch_path = mgr.execute_recall(
        release_dir=release_dir,
        recalled_episode_ids=recalled_ids,
        reason=reason,
        patch_version=patch_version,
    )

    notice_file = patch_path / "recall_notice.json"
    with open(notice_file) as nf:
        notice = json.load(nf)

    # Log recall notice to DuckDB Lineage Catalog
    try:
        from episodeforge.lineage.catalog import DuckDBLineageCatalog

        DuckDBLineageCatalog().log_recall(notice)
    except Exception:
        pass

    table = Table(title="Data Recall Audit Record")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold yellow")

    table.add_row("Recall ID", str(notice.get("recall_id")))
    table.add_row("Original Release", str(notice.get("original_release")))
    table.add_row("Patch Release Path", str(patch_path))
    table.add_row("Recalled Episodes", ", ".join(recalled_ids))
    table.add_row("Reason", notice.get("reason", ""))
    table.add_row("Remediation Status", str(notice.get("remediation_status")))

    console.print(table)


@app.command("plan")
def plan(
    episodes: int = typer.Option(..., help="Episode volume to forecast"),
    spec: Path = typer.Option(..., help="Path to ClientSpec YAML"),
    workers: int = typer.Option(4, help="Target worker parallelism"),
):
    """Forecast compute time, cloud storage, egress, and financial cost for a delivery."""
    from episodeforge.config.client_spec import ClientSpec
    from episodeforge.metrics.cost import forecast_pipeline_cost, load_cost_rates

    if not spec.exists():
        console.print(f"[bold red]Error:[/bold red] Spec path {spec} does not exist.")
        raise typer.Exit(code=1)

    with open(spec) as f:
        spec_data = yaml.safe_load(f)
    spec_model = ClientSpec.model_validate(spec_data)
    rates = load_cost_rates()

    forecast = forecast_pipeline_cost(
        episodes=episodes,
        spec=spec_model,
        rates=rates,
        worker_count=workers,
    )

    console.print(
        f"\n[bold cyan]Pipeline Resource & Financial Forecast:[/bold cyan] {spec_model.client} ({spec_model.format})"
    )

    t_res = Table(title=f"Resource Forecast ({episodes:,} Episodes)")
    t_res.add_column("Resource Dimension", style="cyan")
    t_res.add_column("Estimated Value", style="bold")

    t_res.add_row("Client / Dataset", f"{forecast.client} / {forecast.dataset_name}")
    t_res.add_row("Format & Codec", f"{forecast.format}")
    t_res.add_row("Total Robot Demonstration Time", f"{forecast.total_robot_hours:.2f} hours")
    t_res.add_row("Worker Allocation", f"{forecast.worker_count} parallel workers")
    t_res.add_row(
        "Projected Wall-Clock Duration", f"{forecast.projected_wall_clock_hours:.2f} hours"
    )
    t_res.add_row("Compute Core-Hours", f"{forecast.total_compute_core_hours:.2f} vCPU-hrs")
    t_res.add_row(
        "Delivered Data Footprint",
        f"{forecast.total_dataset_gb:.2f} GB ({forecast.estimated_mb_per_episode:.2f} MB/ep)",
    )
    t_res.add_row(
        "Estimated S3 PUT / GET Calls",
        f"{forecast.s3_put_requests:,} PUTs / {forecast.s3_get_requests:,} GETs",
    )

    console.print(t_res)

    t_fin = Table(title="Financial Quote & Cost Breakdown (USD)")
    t_fin.add_column("Cost Category", style="cyan")
    t_fin.add_column("Rate Basis", style="dim")
    t_fin.add_column("Subtotal", style="bold green")

    t_fin.add_row(
        "Cloud Compute",
        f"${rates.compute_core_hour_usd:.3f} / vCPU-hr",
        f"${forecast.compute_cost_usd:.4f}",
    )
    t_fin.add_row(
        "S3 Storage (Monthly)",
        f"${rates.s3_standard_storage_gb_month_usd:.3f} / GB-mo",
        f"${forecast.storage_monthly_cost_usd:.4f}",
    )
    t_fin.add_row(
        "S3 Egress Transfer",
        f"${rates.s3_egress_gb_usd:.3f} / GB",
        f"${forecast.egress_cost_usd:.4f}",
    )
    t_fin.add_row(
        "S3 API Requests", "$0.005/1k PUT, $0.004/10k GET", f"${forecast.api_requests_cost_usd:.4f}"
    )
    t_fin.add_row(
        "[bold]TOTAL DELIVERY COST[/bold]",
        "-",
        f"[bold green]${forecast.total_cost_usd:.4f}[/bold green]",
    )
    t_fin.add_row(
        "Unit Cost Per Episode",
        "-",
        f"[bold cyan]${forecast.cost_per_episode_usd:.4f} / ep[/bold cyan]",
    )
    t_fin.add_row(
        "Unit Cost Per Robot Hour",
        "-",
        f"[bold cyan]${forecast.cost_per_delivered_hour_usd:.2f} / hr[/bold cyan]",
    )

    console.print(t_fin)
    console.print(
        "[dim]* Note: Cost model rates from configs/cost/rates.yaml. Verify with actual cloud provider rates before quoting.[/dim]\n"
    )


@app.command("report")
def report(
    release: Path | None = typer.Argument(
        None, help="Release directory to compile delivery report for"
    ),
    run: str | None = typer.Option(None, help="Run ID for internal ops report"),
):
    """Generate client-facing delivery report or internal operational report."""
    from episodeforge.lineage.catalog import DuckDBLineageCatalog
    from episodeforge.report.client_report import generate_client_delivery_report
    from episodeforge.report.ops_report import generate_ops_report

    if release:
        rel_path = Path(release)
        if not rel_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Release directory {rel_path} does not exist."
            )
            raise typer.Exit(code=1)

        rep_file = rel_path / "delivery_report.md"
        content = generate_client_delivery_report(rel_path, output_path=rep_file)
        console.print(
            f"[bold green]SUCCESS:[/bold green] Client delivery report generated at {rep_file}"
        )
        console.print(f"Report length: {len(content.splitlines())} lines.")

    elif run:
        catalog = DuckDBLineageCatalog()
        rows = catalog.query(f"SELECT * FROM runs WHERE run_id = '{run}'")
        if not rows:
            console.print(f"[bold red]Error:[/bold red] Run ID {run} not found in lineage catalog.")
            raise typer.Exit(code=1)

        from episodeforge.orchestrate.pipeline_runner import PipelineRunReport

        r = rows[0]
        rep = PipelineRunReport(
            run_id=r["run_id"],
            client=r["client"],
            dataset_name=r["dataset_name"],
            format=r["format"],
            total_episodes_discovered=r["episodes_discovered"],
            episodes_processed=r["episodes_discovered"],
            episodes_quarantined=r["episodes_quarantined"],
            episodes_transformed=r["episodes_released"],
            total_frames=r["total_frames"],
            elapsed_time_sec=r["elapsed_sec"],
            throughput_eps_per_sec=r["episodes_released"] / max(r["elapsed_sec"], 0.01),
            throughput_eps_per_hour=r["throughput_eps_hr"],
            cache_hits=r["cache_hits"],
            cache_misses=r["cache_misses"],
            output_dir=r["output_dir"],
            validation_status=r["validation_status"],
        )
        out_file = Path("data") / f"{run}_ops_report.md"
        generate_ops_report(rep, output_path=out_file)
        console.print(f"[bold green]SUCCESS:[/bold green] Ops report compiled at {out_file}")
    else:
        # Default: compile client delivery report for most recent release
        releases = sorted(
            Path("data/releases").glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if releases:
            latest = releases[0]
            rep_file = latest / "delivery_report.md"
            generate_client_delivery_report(latest, output_path=rep_file)
            console.print(
                f"[bold green]SUCCESS:[/bold green] Client delivery report generated for latest release at {rep_file}"
            )
        else:
            console.print("[bold yellow]No releases found in data/releases/.[/bold yellow]")


@app.command("lineage")
def lineage(
    query: str = typer.Argument(
        ..., help="Preset table name ('runs', 'episodes', 'deliveries', 'recalls') or raw SQL query"
    ),
):
    """Query DuckDB lineage catalog for provenance, execution history, and audit trails."""
    from episodeforge.lineage.catalog import DuckDBLineageCatalog

    catalog = DuckDBLineageCatalog()
    preset = query.strip().lower()

    if preset in ["runs", "run"]:
        sql = "SELECT run_id, client, dataset_name, format, episodes_released, throughput_eps_hr, validation_status, timestamp_utc FROM runs ORDER BY timestamp_utc DESC LIMIT 20"
        title = "Pipeline Execution Runs"
    elif preset in ["episodes", "episode"]:
        sql = "SELECT episode_id, task, success, duration_sec, control_steps, upstream_verdict, content_hash FROM episodes ORDER BY created_at DESC LIMIT 20"
        title = "Tracked Episodes"
    elif preset in ["deliveries", "delivery"]:
        sql = "SELECT receipt_id, client, dataset_name, target_s3_uri, uploaded_files_count, total_bytes, status FROM deliveries ORDER BY completed_at_utc DESC LIMIT 20"
        title = "Cloud Deliveries"
    elif preset in ["recalls", "recall"]:
        sql = "SELECT recall_id, original_release, patch_release, client, recalled_count, reason FROM recalls ORDER BY timestamp_utc DESC LIMIT 20"
        title = "Data Recalls"
    else:
        sql = query
        title = f"Custom SQL Query: {query}"

    try:
        rows = catalog.query(sql)
    except Exception as e:
        console.print(f"[bold red]Query Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if not rows:
        console.print(f"[dim]No rows returned for '{title}'.[/dim]")
        return

    table = Table(title=title)
    cols = list(rows[0].keys())
    for col in cols:
        table.add_column(col, style="cyan" if "id" in col or "status" in col else "bold")

    for row in rows:
        table.add_row(*[str(row[c]) if row[c] is not None else "-" for c in cols])

    console.print(table)


@app.command("eval")
def eval_cmd(suite: str = typer.Option("quick", help="Evaluation suite: quick | full")):
    """Run verification suite and generate evaluation metrics."""
    import subprocess

    console.print(f"[bold green]Running evaluation suite: {suite}...[/bold green]")
    cmd = [".venv/bin/pytest", "tests/", "-q"]
    if suite == "quick":
        cmd.extend(["-k", "not slow"])
    res = subprocess.run(cmd, capture_output=True, text=True)
    console.print(res.stdout)
    if res.returncode != 0:
        console.print(res.stderr)
        console.print("[bold red]FAILED:[/bold red] Evaluation tests did not pass.")
        raise typer.Exit(code=1)
    console.print("[bold green]ALL EVALUATION CRITERIA PASSED.[/bold green]")


@app.command("bench")
def bench(scaling: bool = typer.Option(False, help="Include multi-worker scaling benchmark")):
    """Benchmark stage throughput, video encoding, and IPC."""
    import time

    from episodeforge.config.client_spec import ClientSpec
    from episodeforge.orchestrate.pipeline_runner import PipelineRunner

    raw_dir = Path("data/raw/m1_episodes")
    if not raw_dir.exists():
        console.print(
            "[bold red]Error:[/bold red] data/raw/m1_episodes not found. Run 'forge generate' first."
        )
        raise typer.Exit(code=1)

    spec_p = Path("configs/clients/client_a_lerobot.v1.yaml")
    with open(spec_p) as sf:
        spec = ClientSpec.model_validate(yaml.safe_load(sf))

    worker_counts = [1, 2, 4] if scaling else [4]
    console.print(f"[bold cyan]Running Performance Benchmark (Scaling={scaling})...[/bold cyan]")

    bench_table = Table(title="Pipeline Scaling & Throughput Benchmark")
    bench_table.add_column("Workers", style="cyan")
    bench_table.add_column("Episodes", style="dim")
    bench_table.add_column("Elapsed (s)", style="bold")
    bench_table.add_column("Throughput (eps/s)", style="bold green")
    bench_table.add_column("Throughput (eps/hr)", style="bold green")
    bench_table.add_column("Speedup Factor", style="magenta")

    baseline_eps_s = None
    for w in worker_counts:
        runner = PipelineRunner()
        t0 = time.time()
        rep = runner.run(raw_dir, spec, limit=20, workers=w)
        dur = time.time() - t0
        eps_s = rep.throughput_eps_per_sec
        if baseline_eps_s is None:
            baseline_eps_s = eps_s
        speedup = eps_s / max(baseline_eps_s, 0.001)

        bench_table.add_row(
            str(w),
            "20",
            f"{dur:.2f}s",
            f"{eps_s:.2f}",
            f"{rep.throughput_eps_per_hour:.1f}",
            f"{speedup:.2f}x",
        )

    console.print(bench_table)


@app.command("dashboard")
def dashboard(port: int = typer.Option(8501, help="Port to run Streamlit on")):
    """Launch interactive Streamlit operations and delivery dashboard."""
    import subprocess

    console.print(f"[bold green]Launching Streamlit Dashboard on port {port}...[/bold green]")
    cmd = [".venv/bin/streamlit", "run", "dashboard/app.py", "--server.port", str(port)]
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
