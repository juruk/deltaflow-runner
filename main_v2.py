# main_v2.py  (v2.2)
# CLI Ð·Ð° ÐºÑ€Ð¾Ð»ÐµÑ€ + HTML Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜ Ð·Ð° Ð¼Ð°Ð³Ð°Ñ†Ð¸Ð½Ð¸/Ð¿Ð»Ð°Ñ†Ð¾Ð²Ð¸ Ð²Ð¾ Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ.
# ÐšÐ¾Ð¼Ð°Ð½Ð´Ð¸:
#   python main_v2.py crawl --since 30d --max-items 200
#   python main_v2.py report --since 30d --out reports/dashboard_v2.html
#   python main_v2.py serve-report

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import typer

# ÐÐ´Ð°Ð¿Ñ‚ÐµÑ€Ð¸ Ð¿Ð¾ Ð¸Ð·Ð²Ð¾Ñ€ (Ð¼Ð¸Ð½Ð¸Ð¼Ð°Ð»Ð½Ð¸)
from sources.pazar3_v2 import crawl_pazar3
from sources.reklama5_v2 import crawl_reklama5

app = typer.Typer(help="Crawler + Reports (Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ: Ð¼Ð°Ð³Ð°Ñ†Ð¸Ð½Ð¸ Ð¸ Ð¿Ð»Ð°Ñ†Ð¾Ð²Ð¸)")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "listings.csv"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _parse_since(since: str) -> int:
    """
    ÐŸÑ€Ð¸Ñ„Ð°Ñ‚ÐµÐ½Ð¸ Ñ„Ð¾Ñ€Ð¼Ð°Ñ‚Ð¸: '7d' (Ð´ÐµÐ½Ð¾Ð²Ð¸), '4w' (Ð½ÐµÐ´ÐµÐ»Ð¸).
    Ð’Ñ€Ð°ÑœÐ° Ð±Ñ€Ð¾Ñ˜ Ð½Ð° Ð´ÐµÐ½Ð¾Ð²Ð¸.
    """
    m = re.match(r"^(\d+)([dw])$", since.strip(), flags=re.IGNORECASE)
    if not m:
        # Ð´ÐµÑ„Ð¾Ð»Ñ‚ 30 Ð´ÐµÐ½Ð° Ð°ÐºÐ¾ Ñ„Ð¾Ñ€Ð¼Ð°Ñ‚Ð¾Ñ‚ Ðµ Ð½ÐµÐ²Ð°Ð»Ð¸Ð´ÐµÐ½
        return 30
    val = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "d":
        return val
    if unit == "w":
        return val * 7
    return 30


@app.command()
def crawl(
    since: str = typer.Option("30d", help="ÐŸÐµÑ€Ð¸Ð¾Ð´ Ð·Ð° ÑÐ¾Ð±Ð¸Ñ€Ð°ÑšÐµ, Ð¿Ñ€. 7d Ð¸Ð»Ð¸ 4w"),
    max_items: int = typer.Option(200, help="ÐœÐ°ÐºÑÐ¸Ð¼ÑƒÐ¼ Ð·Ð°Ð¿Ð¸ÑÐ¸ Ð¿Ð¾ Ð¸Ð·Ð²Ð¾Ñ€"),
):
    """
    Ð¡Ð¾Ð±Ð¸Ñ€Ð° Ð¾Ð³Ð»Ð°ÑÐ¸ Ð¾Ð´ Ð¿Ð¾Ð´Ð´Ñ€Ð¶Ð°Ð½Ð¸Ñ‚Ðµ Ð¸Ð·Ð²Ð¾Ñ€Ð¸ Ð¸ Ð³Ð¸ ÑÐ½Ð¸Ð¼Ð° Ð²Ð¾ data/listings.csv.
    """
    _ensure_data_dir()
    days = _parse_since(since)
    start_date = datetime.utcnow() - timedelta(days=days)

    rows = []
    for src_fn in [crawl_pazar3, crawl_reklama5]:
        try:
            got = src_fn(start_date=start_date, limit=max_items)
            typer.echo(f"[INFO] {src_fn.__name__}: +{len(got)} Ð·Ð°Ð¿Ð¸ÑÐ¸")
            rows += got
        except Exception as e:
            typer.echo(f"[WARN] {src_fn.__name__} failed: {e}")

    df = pd.DataFrame(rows)

    if not df.empty:
        # Ð”ÐµÐ´ÑƒÐ¿ Ð¿Ð¾ source+url Ð°ÐºÐ¾ ÐºÐ¾Ð»Ð¾Ð½Ð¸Ñ‚Ðµ Ð¿Ð¾ÑÑ‚Ð¾Ñ˜Ð°Ñ‚
        keep_cols = df.columns.tolist()
        if "source" in keep_cols and "url" in keep_cols:
            df = df.drop_duplicates(subset=["source", "url"])

        # ÐÐ¾Ñ€Ð¼Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸
        if "price_eur" in df.columns:
            df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")
        if "area_m2" in df.columns:
            df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
        if "price_per_m2_eur" not in df.columns and {"price_eur", "area_m2"}.issubset(
            df.columns
        ):
            df["price_per_m2_eur"] = df.apply(
                lambda r: (r["price_eur"] / r["area_m2"])
                if pd.notnull(r.get("price_eur"))
                and pd.notnull(r.get("area_m2"))
                and float(r["area_m2"]) > 0
                else None,
                axis=1,
            )

        df["scraped_at"] = datetime.utcnow().isoformat()

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    typer.echo(f"[OK] Ð—Ð°Ð¿Ð¸ÑˆÐ°Ð½Ð¸ {len(df)} Ð·Ð°Ð¿Ð¸ÑÐ¸ â†’ {CSV_PATH}")


@app.command()
def report(
    since: str = typer.Option("30d", help="ÐŸÐµÑ€Ð¸Ð¾Ð´ Ð·Ð° Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜, Ð¿Ñ€. 30d, 8w"),
    out: Path = typer.Option(Path("reports/dashboard_v2.html"), help="ÐŸÐ°Ñ‚ÐµÐºÐ° Ð´Ð¾ HTML"),
):
    """
    Ð“ÐµÐ½ÐµÑ€Ð¸Ñ€Ð° Ð¸Ð½Ñ‚ÐµÑ€Ð°ÐºÑ‚Ð¸Ð²ÐµÐ½ HTML Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜ Ð¾Ð´ CSV/SQLite.
    """
    # Lazy import Ð·Ð° Ð´Ð° Ð½Ðµ Ð³Ð¾ Ð±Ð»Ð¾ÐºÐ¸Ñ€Ð° crawl Ð°ÐºÐ¾ Ð¸Ð¼Ð° ÑÐ¸Ð½Ñ‚Ð°ÐºÑÐ¸Ñ‡ÐºÐ° Ð³Ñ€ÐµÑˆÐºÐ° Ð²Ð¾ report-Ð¼Ð¾Ð´ÑƒÐ»Ð¾Ñ‚
    from report_builder_v2 import build_report  # type: ignore

    days = _parse_since(since)
    start_date = datetime.utcnow() - timedelta(days=days)
    out.parent.mkdir(parents=True, exist_ok=True)

    build_report(
        db_path=Path("data/data aggregation.db"),
        csv_path=CSV_PATH,
        out_html=out,
        start_date=start_date,
    )
    typer.echo(f"[OK] HTML Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜: {out}")


@app.command(name="serve-report")
def serve_report(port: int = typer.Option(8000, help="Ð›Ð¾ÐºÐ°Ð»ÐµÐ½ Ð¿Ð¾Ñ€Ñ‚ (default 8000)")):
    """
    Ð¡ÐµÑ€Ð²Ð¸Ñ€Ð° /reports Ð»Ð¾ÐºÐ°Ð»Ð½Ð¾ Ð½Ð° http://localhost:<port>
    """
    from report_builder_v2 import serve_reports  # type: ignore

    serve_reports(port)


if __name__ == "__main__":
    app()

