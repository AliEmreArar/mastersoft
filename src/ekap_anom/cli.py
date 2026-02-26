"""Click CLI: ekap-anom command-line interface."""

from __future__ import annotations

import click

from ekap_anom.config import load_settings
from ekap_anom.logging import setup_logging, get_logger


@click.group()
@click.option("--config", "-c", default=None, help="Path to YAML config override file.")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """EKAP Multi-Layer Anomaly Detection System."""
    settings = load_settings(config_path=config)
    setup_logging(settings.general.log_level, settings.general.log_format)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


# ── Parse ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Log file or directory path.")
@click.option("--out", "-o", "output_path", default="data/parsed", help="Output Parquet directory.")
@click.pass_context
def parse(ctx: click.Context, input_path: str, output_path: str) -> None:
    """Parse IIS W3C logs to date-partitioned Parquet."""
    from ekap_anom.ingest.parser import parse_to_parquet

    settings = ctx.obj["settings"]
    logger = get_logger("cli.parse")
    logger.info("parse_started", input=input_path, output=output_path)

    files = parse_to_parquet(input_path, output_path, chunk_size=settings.ingest.chunk_size)
    logger.info("parse_completed", files_written=len(files))
    click.echo(f"✅ Parsed → {len(files)} Parquet files in {output_path}")


# ── Filter Dynamic ─────────────────────────────────────────────────────────

@cli.command("filter-dynamic")
@click.option("--in", "-i", "input_dir", default="data/parsed", help="Parsed Parquet directory.")
@click.option("--out", "-o", "output_dir", default="data/dynamic_only", help="Output directory.")
@click.pass_context
def filter_dynamic(ctx: click.Context, input_dir: str, output_dir: str) -> None:
    """Filter to keep only dynamic pages (.aspx, .ashx)."""
    from ekap_anom.filters.dynamic_only import filter_parquet_directory

    settings = ctx.obj["settings"]
    files = filter_parquet_directory(input_dir, output_dir, config=settings.filter)
    click.echo(f"✅ Filtered → {len(files)} files in {output_dir}")


# ── Featurize Layer 1 ──────────────────────────────────────────────────────

@cli.command("featurize-layer1")
@click.option("--in", "-i", "input_dir", default="data/dynamic_only", help="Input Parquet directory.")
@click.option("--out", "-o", "output_dir", default="data/features/layer1", help="Output directory.")
@click.pass_context
def featurize_layer1(ctx: click.Context, input_dir: str, output_dir: str) -> None:
    """Compute Layer 1 (5-min window) aggregation features."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.features.layer1_agg import build_layer1_features, save_layer1_features

    settings = ctx.obj["settings"]
    dfs = []
    for pf in sorted(Path(input_dir).rglob("*.parquet")):
        dfs.append(pd.read_parquet(pf))

    if not dfs:
        click.echo("⚠️  No input data found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    features = build_layer1_features(df, settings.layer1, settings.url, settings.segmentation)
    files = save_layer1_features(features, output_dir)
    click.echo(f"✅ Layer 1 features → {len(files)} files in {output_dir}")


# ── Train Layer 1 ──────────────────────────────────────────────────────────

@cli.command("train-layer1")
@click.option("--in", "-i", "input_dir", default="data/features/layer1", help="Layer 1 features.")
@click.pass_context
def train_layer1(ctx: click.Context, input_dir: str) -> None:
    """Train/fit Layer 1 baseline model (statistical, no persisted model needed)."""
    click.echo("ℹ️  Layer 1 uses rolling statistics — no explicit training step required.")
    click.echo("    Scoring is computed on-the-fly during infer-layer1.")


# ── Infer Layer 1 ──────────────────────────────────────────────────────────

@cli.command("infer-layer1")
@click.option("--in", "-i", "input_dir", default="data/features/layer1", help="Layer 1 features.")
@click.option("--out", "-o", "output_dir", default="data/features/layer1", help="Output (in-place).")
@click.pass_context
def infer_layer1(ctx: click.Context, input_dir: str, output_dir: str) -> None:
    """Score Layer 1 windows and flag hot windows."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.models.layer1_baseline import score_layer1
    from ekap_anom.features.layer1_agg import save_layer1_features

    settings = ctx.obj["settings"]
    dfs = []
    for pf in sorted(Path(input_dir).rglob("*.parquet")):
        dfs.append(pd.read_parquet(pf))

    if not dfs:
        click.echo("⚠️  No Layer 1 features found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    scored = score_layer1(df, settings.layer1)
    files = save_layer1_features(scored, output_dir)

    hot = scored["window_is_hot"].sum()
    click.echo(f"✅ Layer 1 scored: {hot}/{len(scored)} hot windows → {output_dir}")


# ── Featurize Layer 2 ──────────────────────────────────────────────────────

@cli.command("featurize-layer2")
@click.option("--in", "-i", "input_dir", default="data/dynamic_only", help="Dynamic Parquet dir.")
@click.option("--layer1-dir", default="data/features/layer1", help="Layer 1 features (for join).")
@click.option("--out", "-o", "output_dir", default="data/features/layer2", help="Output directory.")
@click.pass_context
def featurize_layer2(ctx: click.Context, input_dir: str, layer1_dir: str, output_dir: str) -> None:
    """Compute Layer 2 request-level features (only hot windows + critical endpoints)."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.features.layer1_agg import enrich_for_layer1
    from ekap_anom.features.layer2_req import build_layer2_features, save_layer2_features
    from ekap_anom.features.joins import join_layer1_scores_to_requests, filter_for_layer2

    settings = ctx.obj["settings"]

    # Load raw dynamic data
    raw_dfs = [pd.read_parquet(pf) for pf in sorted(Path(input_dir).rglob("*.parquet"))]
    if not raw_dfs:
        click.echo("⚠️  No dynamic data found.")
        return
    raw = pd.concat(raw_dfs, ignore_index=True)

    # Enrich
    enriched = enrich_for_layer1(raw, settings.url, settings.segmentation)

    # Load Layer 1 scores
    l1_dfs = [pd.read_parquet(pf) for pf in sorted(Path(layer1_dir).rglob("*.parquet"))]
    if l1_dfs:
        l1 = pd.concat(l1_dfs, ignore_index=True)
        enriched = join_layer1_scores_to_requests(
            enriched, l1, settings.layer1.window_size_minutes
        )
        enriched = filter_for_layer2(enriched, settings.critical_endpoints)

    # Build Layer 2 features
    features = build_layer2_features(enriched, settings.hashing)
    files = save_layer2_features(features, output_dir)
    click.echo(f"✅ Layer 2 features → {len(files)} files ({len(features)} requests) in {output_dir}")


# ── Train Layer 2 ──────────────────────────────────────────────────────────

@cli.command("train-layer2")
@click.option("--in", "-i", "input_dir", default="data/features/layer2", help="Layer 2 features.")
@click.option("--mode", default="iforest", type=click.Choice(["iforest", "ae_latent_if"]))
@click.option("--model-out", default="models/layer2_iforest.pkl", help="Model output path.")
@click.pass_context
def train_layer2(ctx: click.Context, input_dir: str, mode: str, model_out: str) -> None:
    """Train Layer 2 model (Isolation Forest or AE+IF)."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.features.layer2_req import get_layer2_feature_columns

    settings = ctx.obj["settings"]
    dfs = [pd.read_parquet(pf) for pf in sorted(Path(input_dir).rglob("*.parquet"))]
    if not dfs:
        click.echo("⚠️  No Layer 2 features found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    feature_cols = get_layer2_feature_columns(settings.hashing)
    available_cols = [c for c in feature_cols if c in df.columns]

    if mode == "iforest":
        from ekap_anom.models.layer2_iforest import SegmentIForest
        model = SegmentIForest(
            if_config=settings.layer2.iforest,
            min_segment_count=settings.segmentation.min_segment_count,
        )
        model.fit(df, available_cols)
        model.save(model_out)
    else:
        from ekap_anom.models.layer2_autoencoder import AELatentIForest
        model = AELatentIForest(  # type: ignore[assignment]
            ae_config=settings.layer2.autoencoder,
            if_config=settings.layer2.iforest,
        )
        model.fit(df, available_cols)
        model.save(model_out)

    click.echo(f"✅ Layer 2 model ({mode}) saved → {model_out}")


# ── Infer Layer 2 ──────────────────────────────────────────────────────────

@cli.command("infer-layer2")
@click.option("--in", "-i", "input_dir", default="data/features/layer2", help="Layer 2 features.")
@click.option("--model", default="models/layer2_iforest.pkl", help="Model path.")
@click.option("--out", "-o", "output_dir", default="data/features/layer2", help="Output dir.")
@click.pass_context
def infer_layer2(ctx: click.Context, input_dir: str, model: str, output_dir: str) -> None:
    """Score requests with Layer 2 model."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.models.layer2_iforest import SegmentIForest
    from ekap_anom.features.layer2_req import save_layer2_features

    loaded_model = SegmentIForest.load(model)
    dfs = [pd.read_parquet(pf) for pf in sorted(Path(input_dir).rglob("*.parquet"))]
    if not dfs:
        click.echo("⚠️  No Layer 2 features found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df["layer2_score"] = loaded_model.predict(df)
    files = save_layer2_features(df, output_dir)
    click.echo(f"✅ Layer 2 scored → {len(files)} files in {output_dir}")


# ── Score (end-to-end) ─────────────────────────────────────────────────────

@cli.command("score")
@click.option("--date", required=True, help="Date to score (YYYY-MM-DD).")
@click.option("--out", "-o", "output_dir", default="data/alerts", help="Alert output directory.")
@click.pass_context
def score(ctx: click.Context, date: str, output_dir: str) -> None:
    """Produce final scores and alerts for a specific date."""
    import pandas as pd
    from pathlib import Path
    from ekap_anom.scoring.combine import combine_scores
    from ekap_anom.scoring.threshold import compute_segment_thresholds, apply_thresholds
    from ekap_anom.scoring.explain import add_reason_codes
    from ekap_anom.scoring.alerts import produce_alerts, save_alerts

    settings = ctx.obj["settings"]

    # Load Layer 2 scored data
    l2_dir = Path("data/features/layer2") / f"date={date}"
    if not l2_dir.exists():
        click.echo(f"⚠️  No Layer 2 data for {date}.")
        return

    dfs = [pd.read_parquet(pf) for pf in l2_dir.glob("*.parquet")]
    if not dfs:
        click.echo(f"⚠️  No parquet files for {date}.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Combine scores
    df = combine_scores(df, settings.scoring.weights)

    # Segment thresholds
    thresholds = compute_segment_thresholds(df, quantile=settings.scoring.segment_quantile)
    df = apply_thresholds(df, thresholds)

    # Explain
    df = add_reason_codes(df)

    # Produce alerts
    alerts = produce_alerts(df, settings.scoring, settings.critical_endpoints)
    files = save_alerts(alerts, output_dir, date=date)

    click.echo(f"✅ {len(alerts)} alerts for {date} → {output_dir}")


# ── Serve ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=8000, type=int, help="Bind port.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    settings = ctx.obj["settings"]
    click.echo(f"🚀 Starting EKAP Anomaly API on {host}:{port}")
    uvicorn.run(
        "ekap_anom.api.app:app",
        host=host,
        port=port,
        workers=settings.api.workers,
        log_level=settings.general.log_level.lower(),
    )


if __name__ == "__main__":
    cli()
