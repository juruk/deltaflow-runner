# main_v2.py  (v2.2)
# CLI за кролер + HTML извештај за магацини/плацови во Скопје.
# Команди:
#   python main_v2.py crawl --since 30d --max-items 200
#   python main_v2.py report --since 30d --out reports/dashboard_v2.html
#   python main_v2.py serve-report

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import typer

# Адаптери по извор (минимални)
from sources.pazar3_v2 import crawl_pazar3
from sources.reklama5_v2 import crawl_reklama5

app = typer.Typer(help="Crawler + Reports (Скопје: магацини и плацови)")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "listings.csv"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _parse_since(since: str) -> int:
    """
    Прифатени формати: '7d' (денови), '4w' (недели).
    Враќа број на денови.
    """
    m = re.match(r"^(\d+)([dw])$", since.strip(), flags=re.IGNORECASE)
    if not m:
        # дефолт 30 дена ако форматот е невалиден
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
    since: str = typer.Option("30d", help="Период за собирање, пр. 7d или 4w"),
    max_items: int = typer.Option(200, help="Максимум записи по извор"),
):
    """
    Собира огласи од поддржаните извори и ги снима во data/listings.csv.
    """
    _ensure_data_dir()
    days = _parse_since(since)
    start_date = datetime.utcnow() - timedelta(days=days)

    rows = []
    for src_fn in [crawl_pazar3, crawl_reklama5]:
        try:
            got = src_fn(start_date=start_date, limit=max_items)
            typer.echo(f"[INFO] {src_fn.__name__}: +{len(got)} записи")
            rows += got
        except Exception as e:
            typer.echo(f"[WARN] {src_fn.__name__} failed: {e}")

    df = pd.DataFrame(rows)

    if not df.empty:
        # Дедуп по source+url ако колоните постојат
        keep_cols = df.columns.tolist()
        if "source" in keep_cols and "url" in keep_cols:
            df = df.drop_duplicates(subset=["source", "url"])

        # Нормализации
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
    typer.echo(f"[OK] Запишани {len(df)} записи → {CSV_PATH}")


@app.command()
def report(
    since: str = typer.Option("30d", help="Период за извештај, пр. 30d, 8w"),
    out: Path = typer.Option(Path("reports/dashboard_v2.html"), help="Патека до HTML"),
):
    """
    Генерира интерактивен HTML извештај од CSV/SQLite.
    """
    # Lazy import за да не го блокира crawl ако има синтаксичка грешка во report-модулот
    from report_builder_v2 import build_report  # type: ignore

    days = _parse_since(since)
    start_date = datetime.utcnow() - timedelta(days=days)
    out.parent.mkdir(parents=True, exist_ok=True)

    build_report(
        db_path=Path("data/realestate.db"),
        csv_path=CSV_PATH,
        out_html=out,
        start_date=start_date,
    )
    typer.echo(f"[OK] HTML извештај: {out}")


@app.command(name="serve-report")
def serve_report(port: int = typer.Option(8000, help="Локален порт (default 8000)")):
    """
    Сервира /reports локално на http://localhost:<port>
    """
    from report_builder_v2 import serve_reports  # type: ignore

    serve_reports(port)


if __name__ == "__main__":
    app()
