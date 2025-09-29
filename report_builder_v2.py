# report_builder_v2.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import math
import pandas as pd
import plotly.express as px
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- Константи/патеки ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = ROOT / "templates"
CSV_PATH = DATA_DIR / "listings.csv"

# ---------- Помошни ----------

def _utc_now_str() -> str:
    # ISO без микро секунди, секогаш UTC за конзистентност
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        # UTF-8 со ignore за ретки грешки од изворите
        return pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        # fallback – дозволи да не падне build
        return pd.read_csv(csv_path, encoding_errors="ignore", on_bad_lines="skip")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Стандарден сет колони што ги очекува извештајот
    for col in [
        "source", "url", "title", "category",
        "price_eur", "area_m2", "municipality",
        "created_at",
    ]:
        if col not in df.columns:
            df[col] = None

    # Нормализирај типови
    for num in ("price_eur", "area_m2"):
        df[num] = pd.to_numeric(df[num], errors="coerce")

    # created_at → datetime (ако е NaN, ќе стане NaT)
    if df["created_at"].dtype == object:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # €/m²
    df["price_per_m2_eur"] = pd.to_numeric(
        df.get("price_per_m2_eur", pd.Series([math.nan] * len(df))), errors="coerce"
    )
    mask_need = df["price_per_m2_eur"].isna() & df["price_eur"].notna() & df["area_m2"].notna() & (df["area_m2"] > 0)
    df.loc[mask_need, "price_per_m2_eur"] = df.loc[mask_need, "price_eur"] / df.loc[mask_need, "area_m2"]

    # Категорија нормализација
    df["category"] = (df["category"].fillna("")
                      .str.strip()
                      .str.lower()
                      .replace({"warehouse": "warehouse", "land": "land"}))
    # Municipality (ако нема) → "unknown"
    df["municipality"] = df["municipality"].fillna("unknown").replace("", "unknown")

    return df


def _filter_since(df: pd.DataFrame, start_dt_utc: Optional[datetime]) -> pd.DataFrame:
    if df.empty or not start_dt_utc:
        return df
    if start_dt_utc.tzinfo is None:
        start_dt_utc = start_dt_utc.replace(tzinfo=timezone.utc)
    if "created_at" in df.columns and pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        return df[df["created_at"] >= start_dt_utc]
    return df  # ако нема дата, не филтрираме


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Едноставен score: понови огласи и разумен €/m² добиваат подобар ранг."""
    if df.empty:
        df["score"] = []
        return df

    # Свежина во денови (помалку подобро)
    now = datetime.now(timezone.utc)
    age_days = (now - df["created_at"]).dt.total_seconds() / 86400
    age_days = age_days.fillna(age_days.max() if len(age_days) else 30)

    # €/m² – нормализирај
    ppm = df["price_per_m2_eur"].astype(float)
    ppm_med = float(pd.Series(ppm.dropna()).median()) if ppm.notna().any() else 0.0
    ppm_dev = (ppm - ppm_med).abs()
    ppm_dev = ppm_dev.fillna(ppm_dev.max() if len(ppm_dev) else 0)

    # Ниска старост и ниска девијација од медијаната → висок скор
    # Превртени и скалирани на [0,1]
    age_norm = 1 / (1 + age_days)  # 0..1
    dev_norm = 1 / (1 + (ppm_dev / (ppm_med + 1e-6)))  # 0..1

    df = df.copy()
    df["score"] = 0.6 * age_norm + 0.4 * dev_norm
    return df


def _plot_html(fig) -> str:
    if fig is None:
        return ""
    try:
        return pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            config={"displayModeBar": False},
        )
    except Exception:
        return ""


# ---------- Генерација на графици ----------

def _build_charts(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    if df.empty:
        return ("", "", "", "")

    # Чисти податоци за графици
    dfp = df.copy()

    # 1) Дистрибуција €/m² по категорија
    try:
        df1 = dfp[dfp["price_per_m2_eur"].notna() & (dfp["price_per_m2_eur"] > 0)]
        if df1.empty:
            plot_dist = ""
        else:
            fig1 = px.histogram(
                df1, x="price_per_m2_eur", color="category",
                nbins=40, opacity=0.85, barmode="overlay",
                labels={"price_per_m2_eur": "€/m²", "category": "Категорија"},
                title=None,
            )
            fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
            plot_dist = _plot_html(fig1)
    except Exception:
        plot_dist = ""

    # 2) Scatter: Површина vs Цена
    try:
        df2 = dfp[dfp["area_m2"].notna() & dfp["price_eur"].notna()]
        if df2.empty:
            plot_scatter = ""
        else:
            fig2 = px.scatter(
                df2, x="area_m2", y="price_eur", color="category",
                hover_data=["title", "municipality", "price_per_m2_eur"],
                labels={"area_m2": "m²", "price_eur": "€"},
                trendline="ols",
                title=None,
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
            plot_scatter = _plot_html(fig2)
    except Exception:
        plot_scatter = ""

    # 3) Топ општини по медијана €/m²
    try:
        df3 = dfp[dfp["price_per_m2_eur"].notna() & (dfp["price_per_m2_eur"] > 0)]
        if df3.empty:
            plot_top_munis = ""
        else:
            g = df3.groupby("municipality", dropna=False)["price_per_m2_eur"].median().sort_values(ascending=False).head(12)
            fig3 = px.bar(
                g.reset_index(), x="municipality", y="price_per_m2_eur",
                labels={"municipality": "Општина", "price_per_m2_eur": "Медијана €/m²"},
                title=None,
            )
            fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            plot_top_munis = _plot_html(fig3)
    except Exception:
        plot_top_munis = ""

    # 4) Обем по старост (created_at → бинови)
    try:
        df4 = dfp[dfp["created_at"].notna()]
        if df4.empty:
            plot_volume = ""
        else:
            # Број по ден (последни 60 дена)
            s = df4.groupby(df4["created_at"].dt.date).size().sort_index()
            s = s.tail(60)
            fig4 = px.area(
                s.reset_index(names=["date"]).rename(columns={0: "count"}),
                x="date", y="count", title=None,
                labels={"date": "Датум", "count": "Број огласи"},
            )
            fig4.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            plot_volume = _plot_html(fig4)
    except Exception:
        plot_volume = ""

    return (plot_dist, plot_scatter, plot_top_munis, plot_volume)


# ---------- Главна функција ----------

def build_report(start_date_utc: Optional[datetime], out_html: Path) -> Path:
    """
    Гради HTML извештај (UTF-8) во `out_html`.
    :param start_date_utc: филтрирај од оваа дата (UTC) ако е зададена
    :param out_html: патека за HTML (на пр. reports/dashboard_v2.html)
    :return: патеката до генерираниот HTML
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    df = _safe_load_csv(CSV_PATH)
    df = _ensure_columns(df)

    if start_date_utc:
        df = _filter_since(df, start_date_utc)

    # Сортирај по score за табела
    if not df.empty:
        df = _score_rows(df)
        df_sorted = df.sort_values("score", ascending=False).copy()
    else:
        df_sorted = df.copy()

    # Избери колони за табела и првите 100
    table_cols = [c for c in ["title", "category", "price_eur", "area_m2", "price_per_m2_eur", "municipality", "source", "url", "created_at", "score"] if c in df_sorted.columns]
    table_rows = []
    if not df_sorted.empty and table_cols:
        # форматирај разумно бројки
        df_show = df_sorted[table_cols].head(100).copy()
        if "price_eur" in df_show:
            df_show["price_eur"] = df_show["price_eur"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
        if "area_m2" in df_show:
            df_show["area_m2"] = df_show["area_m2"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
        if "price_per_m2_eur" in df_show:
            df_show["price_per_m2_eur"] = df_show["price_per_m2_eur"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
        if "created_at" in df_show and pd.api.types.is_datetime64_any_dtype(df_show["created_at"]):
            df_show["created_at"] = df_show["created_at"].dt.strftime("%Y-%m-%d")
        table_rows = df_show.fillna("").to_dict(orient="records")

    # Графици
    plot_dist, plot_scatter, plot_top_munis, plot_volume = _build_charts(df)

    # Рендерирај Jinja2 темплејт
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    tmpl = env.get_template("dashboard_v2.html.j2")
    html = tmpl.render(
        generated_at=_utc_now_str(),
        plot_dist=plot_dist,
        plot_scatter=plot_scatter,
        plot_top_munis=plot_top_munis,
        plot_volume=plot_volume,
        columns=table_cols,
        rows=table_rows,
    )

    # Запиши UTF-8 (клучно против „грд encoding“)
    out_html.write_text(html, encoding="utf-8")
    return out_html


# ---------- Локално сервирање ----------

def serve_reports(port: int = 8000) -> None:
    """
    Сервира ./reports/ на http://localhost:<port>
    """
    import http.server
    import socketserver
    import os

    reports_dir = (ROOT / "reports").resolve()
    reports_dir.mkdir(exist_ok=True)

    # Промени работна директорија само за серверот
    os.chdir(str(reports_dir))

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"[SERVE] http://localhost:{port}  (CTRL+C за стоп)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[OK] Stop сервер")
            httpd.server_close()
