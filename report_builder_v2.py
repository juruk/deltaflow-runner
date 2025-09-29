# report_builder_v2.py  (v2.3 â€” clean Python file)
"""
HTML Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜ Ð·Ð° Ð¼Ð°Ð³Ð°Ñ†Ð¸Ð½Ð¸/Ð¿Ð»Ð°Ñ†Ð¾Ð²Ð¸ (Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ) ÑÐ¾ Ð¸Ð½Ñ‚ÐµÑ€Ð°ÐºÑ‚Ð¸Ð²Ð½Ð¸ Ð³Ñ€Ð°Ñ„Ð¸Ñ†Ð¸ Ð¸ Ñ‚Ð°Ð±ÐµÐ»Ð°.
- Ð’Ñ‡Ð¸Ñ‚ÑƒÐ²Ð° Ð¾Ð´ CSV (Ð¿Ñ€ÐµÑ„ÐµÑ€ÐµÐ½Ñ†Ð¸Ñ€Ð°Ð½Ð¾) Ð¸Ð»Ð¸ SQLite.
- Ð“ÐµÐ½ÐµÑ€Ð¸Ñ€Ð° Plotly Ð³Ñ€Ð°Ñ„Ð¸Ñ†Ð¸ + DataTables Ñ‚Ð°Ð±ÐµÐ»Ð°.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import http.server
import socketserver

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.express as px
import plotly.io as pio


# ---------- Data loading & prep ----------

def _load_df(db_path: Path, csv_path: Path) -> pd.DataFrame:
    """Ð’Ñ‡Ð¸Ñ‚Ð°Ñ˜ Ð¿Ð¾Ð´Ð°Ñ‚Ð¾Ñ†Ð¸: Ð¿Ñ€Ð²Ð¾ Ð¾Ð´ CSV, Ð°ÐºÐ¾ Ð³Ð¾ Ð½ÐµÐ¼Ð°/Ðµ Ð¿Ñ€Ð°Ð·ÐµÐ½ â†’ Ð¾Ð´ SQLite."""
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            if not df.empty:
                return df
        except Exception:
            pass
    if db_path.exists():
        import sqlite3
        with sqlite3.connect(db_path) as con:
            return pd.read_sql_query("SELECT * FROM listings", con)
    # Ð¿Ñ€Ð°Ð·ÐµÐ½ DataFrame ÑÐ¾ Ð¾Ñ‡ÐµÐºÑƒÐ²Ð°Ð½Ð¸ ÐºÐ¾Ð»Ð¾Ð½Ð¸
    cols = [
        "source","url","listing_id","title","transaction","category",
        "municipality","neighborhood","area_m2","plot_area_m2",
        "price_eur","currency_raw","price_per_m2_eur",
        "contact_phone","contact_name","date_listed","scraped_at","images_count",
        "description","score"
    ]
    return pd.DataFrame(columns=cols)


def _compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """ÐŸÑ€Ð¸Ð±Ð»Ð¸Ð¶ÐµÐ½ ÑÐºÐ¾Ñ€: Ñ†ÐµÐ½Ð°/mÂ² (z-score, Ð¿Ð¾Ð½Ð¸ÑÐºÐ¾=Ð¿Ð¾Ð´Ð¾Ð±Ñ€Ð¾), ÑÐ²ÐµÐ¶Ð¸Ð½Ð°, Ð´Ð¾ÑÑ‚Ð°Ð¿Ð½Ð¾ÑÑ‚, ÐºÐ¾Ð¼Ð¿Ð»ÐµÑ‚Ð½Ð¾ÑÑ‚."""
    import numpy as np
    d = df.copy()

    # z-score Ð½Ð° â‚¬/mÂ² Ð¿Ð¾ category+municipality
    if "price_per_m2_eur" in d.columns:
        grp = d.groupby(["category","municipality"], dropna=False)["price_per_m2_eur"]
        z = (d["price_per_m2_eur"] - grp.transform("mean")) / grp.transform("std")
        z = z.replace([np.inf, -np.inf], pd.NA).fillna(0.0)
        zneg = -z  # Ð¿Ð¾Ð½Ð¸ÑÐºÐ° Ñ†ÐµÐ½Ð°/mÂ² -> Ð¿Ð¾Ð´Ð¾Ð±Ñ€Ð¾
    else:
        zneg = 0.0

    # ÑÐ²ÐµÐ¶Ð¸Ð½Ð°
    if "scraped_at" in d.columns:
        d["scraped_at"] = pd.to_datetime(d["scraped_at"], errors="coerce", utc=True)
        now = pd.Timestamp.utcnow()
        age_days = (now - d["scraped_at"]).dt.days.clip(lower=0)
        freshness = (30 - age_days).clip(lower=0, upper=30) / 30.0
    else:
        freshness = 0.5

    # Ð´Ð¾ÑÑ‚Ð°Ð¿Ð½Ð¾ÑÑ‚
    has_phone = d["contact_phone"].notna().astype(float) if "contact_phone" in d.columns else 0.0
    imgs = d["images_count"].fillna(0) if "images_count" in d.columns else 0
    accessibility = (has_phone > 0).astype(float) * 0.5 + (imgs >= 5).astype(float) * 0.5

    # ÐºÐ¾Ð¼Ð¿Ð»ÐµÑ‚Ð½Ð¾ÑÑ‚
    completeness_cols = ["title","price_eur","area_m2","municipality","url"]
    have = [(d[c].notna().astype(float) if c in d.columns else 0.0) for c in completeness_cols]
    completeness = sum(have) / max(1, len(completeness_cols))

    S = 0.55 * (zneg if isinstance(zneg, pd.Series) else 0.0) + 0.25 * freshness + 0.10 * accessibility + 0.10 * completeness
    d["score"] = S
    return d


def _prep_df(df: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
    """ÐÐ¾Ñ€Ð¼Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ð¸ Ð¸ Ñ„Ð¸Ð»Ñ‚Ñ€Ð¸ Ð·Ð° Ð¿ÐµÑ€Ð¸Ð¾Ð´."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
        try:
            df = df[df["scraped_at"] >= pd.Timestamp(start_date, tz="UTC")]
        except Exception:
            pass

    # Ð±Ñ€Ð¾Ñ˜ÐºÐ¸
    for c in ("price_eur", "area_m2", "plot_area_m2", "price_per_m2_eur", "images_count", "score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive â‚¬/m2 Ð°ÐºÐ¾ Ð½ÐµÐ´Ð¾ÑÑ‚Ð¸Ð³Ð°
    if "price_per_m2_eur" not in df.columns and {"price_eur","area_m2"}.issubset(df.columns):
        df["price_per_m2_eur"] = df.apply(
            lambda r: (float(r["price_eur"]) / float(r["area_m2"])) if pd.notnull(r.get("price_eur")) and pd.notnull(r.get("area_m2")) and float(r["area_m2"]) > 0 else None,
            axis=1
        )

    # Ð´ÐµÑ„Ð¾Ð»Ñ‚ Ñ‚ÐµÐºÑÑ‚ Ð¿Ð¾Ð»Ð¸ÑšÐ°
    for c in ("category", "municipality", "title", "source"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "municipality" not in df.columns:
        df["municipality"] = "unknown"

    if "score" not in df.columns or df["score"].isna().all():
        df = _compute_score(df)

    return df


# ---------- Charts & table ----------

def _to_plot_html(fig) -> str:
    """Ð’Ð³Ñ€Ð°Ð´ÐµÐ½ Plotly HTML (Ð±ÐµÐ· full_html)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True})


def _placeholder_html(msg: str) -> str:
    return f'<div style="padding:1rem;border:1px dashed #ccc;border-radius:8px;background:#fafafa;">{msg}</div>'


def _make_figures(df: pd.DataFrame) -> dict[str, str]:
    """ÐšÑ€ÐµÐ¸Ñ€Ð°Ñ˜ Ð³Ñ€Ð°Ñ„Ð¸Ñ†Ð¸; Ð°ÐºÐ¾ Ð½ÐµÐ¼Ð° Ð¿Ð¾Ð´Ð°Ñ‚Ð¾Ñ†Ð¸, Ð²Ñ€Ð°Ñ‚Ð¸ placeholders."""
    figs_html: dict[str, str] = {"dist": "", "scatter": "", "top_munis": "", "volume": ""}

    if df is None or df.empty:
        ph = _placeholder_html("ÐÐµÐ¼Ð° Ð¿Ð¾Ð´Ð°Ñ‚Ð¾Ñ†Ð¸ Ð·Ð° Ð¿Ñ€Ð¸ÐºÐ°Ð¶ÑƒÐ²Ð°ÑšÐµ (CSV/DB Ðµ Ð¿Ñ€Ð°Ð·ÐµÐ½). ÐŸÑƒÑˆÑ‚Ð¸ crawl Ð½Ð°Ñ˜Ð¿Ñ€Ð²Ð¾.")
        for k in figs_html:
            figs_html[k] = ph
        return figs_html

    # 1) Ð¥Ð¸ÑÑ‚Ð¾Ð³Ñ€Ð°Ð¼ â‚¬/mÂ² Ð¿Ð¾ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ñ˜Ð°
    d1 = df[df["price_per_m2_eur"].notnull()] if "price_per_m2_eur" in df.columns else pd.DataFrame()
    if not d1.empty:
        fig1 = px.histogram(d1, x="price_per_m2_eur", color="category", nbins=50, marginal="box",
                            title="Ð Ð°ÑÐ¿Ñ€ÐµÐ´ÐµÐ»Ð±Ð° Ð½Ð° â‚¬/mÂ² Ð¿Ð¾ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ñ˜Ð°")
        figs_html["dist"] = _to_plot_html(fig1)
    else:
        figs_html["dist"] = _placeholder_html("ÐÐµÐ´Ð¾ÑÑ‚Ð°ÑÑƒÐ²Ð° price_per_m2_eur Ð·Ð° Ñ…Ð¸ÑÑ‚Ð¾Ð³Ñ€Ð°Ð¼.")

    # 2) Scatter: Ð¿Ð¾Ð²Ñ€ÑˆÐ¸Ð½Ð° vs Ñ†ÐµÐ½Ð°
    d2 = df[df["area_m2"].notnull() & df["price_eur"].notnull()] if {"area_m2","price_eur"}.issubset(df.columns) else pd.DataFrame()
    if not d2.empty:
        try:
            import statsmodels.api  # noqa: F401
            trend = "ols"
        except Exception:
            trend = None
        fig2 = px.scatter(
            d2, x="area_m2", y="price_eur", color="category",
            hover_data=[c for c in ["title","municipality","url"] if c in d2.columns],
            trendline=trend, title="ÐŸÐ¾Ð²Ñ€ÑˆÐ¸Ð½Ð° vs Ð¦ÐµÐ½Ð°"
        )
        fig2.update_layout(xaxis_title="mÂ²", yaxis_title="â‚¬")
        figs_html["scatter"] = _to_plot_html(fig2)
    else:
        figs_html["scatter"] = _placeholder_html("ÐÐµÐ´Ð¾ÑÑ‚Ð°ÑÑƒÐ²Ð°Ð°Ñ‚ Ð´Ð¾Ð²Ð¾Ð»Ð½Ð¾ Ð·Ð°Ð¿Ð¸ÑÐ¸ ÑÐ¾ area_m2 Ð¸ price_eur Ð·Ð° scatter.")

    # 3) Ð¢Ð¾Ð¿ Ð¾Ð¿ÑˆÑ‚Ð¸Ð½Ð¸ Ð¿Ð¾ Ð¼ÐµÐ´Ð¸Ñ˜Ð°Ð½Ð° â‚¬/mÂ²
    if not d1.empty and "municipality" in d1.columns:
        d3 = d1.groupby("municipality", dropna=False)["price_per_m2_eur"].median().sort_values().tail(10)
        if not d3.empty:
            fig3 = px.bar(d3, x=d3.index, y=d3.values, title="Ð¢Ð¾Ð¿ Ð¾Ð¿ÑˆÑ‚Ð¸Ð½Ð¸ Ð¿Ð¾ Ð¼ÐµÐ´Ð¸Ñ˜Ð°Ð½Ð° â‚¬/mÂ²")
            fig3.update_layout(xaxis_title="ÐžÐ¿ÑˆÑ‚Ð¸Ð½Ð°", yaxis_title="â‚¬/mÂ² (Ð¼ÐµÐ´Ð¸Ñ˜Ð°Ð½Ð°)")
            figs_html["top_munis"] = _to_plot_html(fig3)
        else:
            figs_html["top_munis"] = _placeholder_html("ÐÐµÐ¼Ð° Ð´Ð¾Ð²Ð¾Ð»Ð½Ð¾ Ð¿Ð¾Ð´Ð°Ñ‚Ð¾Ñ†Ð¸ Ð·Ð° Ð¾Ð¿ÑˆÑ‚Ð¸Ð½Ð¸.")
    else:
        figs_html["top_munis"] = _placeholder_html("ÐÐµÐ´Ð¾ÑÑ‚Ð°ÑÑƒÐ²Ð°Ð°Ñ‚ Ð¿Ð¾Ð´Ð°Ñ‚Ð¾Ñ†Ð¸ Ð·Ð° municipality/â‚¬/mÂ².")

    # 4) ÐžÐ±ÐµÐ¼ Ð¿Ð¾ ÑÑ‚Ð°Ñ€Ð¾ÑÑ‚
    if "scraped_at" in df.columns and pd.api.types.is_datetime64_any_dtype(df["scraped_at"]):
        now = pd.Timestamp.utcnow()
        age = (now - df["scraped_at"]).dt.days
        buckets = pd.cut(age, bins=[-1,3,7,30,365], labels=["â‰¤3 Ð´ÐµÐ½Ð°","â‰¤7","â‰¤30",">30"])
        d4 = buckets.value_counts().reindex(["â‰¤3 Ð´ÐµÐ½Ð°","â‰¤7","â‰¤30",">30"]).fillna(0)
        fig4 = px.bar(x=d4.index, y=d4.values, title="ÐžÐ±ÐµÐ¼ Ð½Ð° Ð¾Ð³Ð»Ð°ÑÐ¸ Ð¿Ð¾ ÑÑ‚Ð°Ñ€Ð¾ÑÑ‚")
        fig4.update_layout(xaxis_title="Ð¡Ñ‚Ð°Ñ€Ð¾ÑÑ‚", yaxis_title="Ð‘Ñ€Ð¾Ñ˜ Ð½Ð° Ð¾Ð³Ð»Ð°ÑÐ¸")
        figs_html["volume"] = _to_plot_html(fig4)
    else:
        figs_html["volume"] = _placeholder_html("ÐÐµÐ´Ð¾ÑÑ‚Ð°ÑÑƒÐ²Ð° Ð²Ð°Ð»Ð¸Ð´Ð½Ð¾ Ð¿Ð¾Ð»Ðµ scraped_at (Ð´Ð°Ñ‚ÑƒÐ¼/Ð²Ñ€ÐµÐ¼Ðµ).")

    return figs_html


def _format_currency_eur(x) -> str:
    try:
        n = float(x)
        return f"â‚¬{int(round(n)):,}".replace(",", " ")
    except Exception:
        return ""


def _format_price_per_m2(x) -> str:
    try:
        v = float(x)
        return f"â‚¬{v:,.0f}/mÂ²".replace(",", " ")
    except Exception:
        return ""


def _table_rows(df: pd.DataFrame, limit=100) -> pd.DataFrame:
    """ÐŸÐ¾Ð´Ð³Ð¾Ñ‚Ð²Ð¸ Ñ‚Ð°Ð±ÐµÐ»Ð° (Top ÑÐ¿Ð¾Ñ€ÐµÐ´ score/ÑÐ²ÐµÐ¶Ð¸Ð½Ð°) ÑÐ¾ Ð½Ð°Ñ˜Ð²Ð°Ð¶Ð½Ð¸ ÐºÐ¾Ð»Ð¾Ð½Ð¸."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols_pref = [
        "score","category","municipality","area_m2","plot_area_m2",
        "price_eur","price_per_m2_eur","title","url","scraped_at","source"
    ]
    present = [c for c in cols_pref if c in df.columns]
    if not present:
        return pd.DataFrame()
    d = df[present].copy()
    if "score" in d.columns and d["score"].notnull().any():
        d = d.sort_values("score", ascending=False)
    elif "scraped_at" in d.columns and pd.api.types.is_datetime64_any_dtype(d["scraped_at"]):
        d = d.sort_values("scraped_at", ascending=False)
    d = d.head(limit)
    if "price_eur" in d.columns:
        d["price_eur"] = d["price_eur"].apply(_format_currency_eur)
    if "price_per_m2_eur" in d.columns:
        d["price_per_m2_eur"] = d["price_per_m2_eur"].apply(_format_price_per_m2)
    if "scraped_at" in d.columns:
        d["scraped_at"] = pd.to_datetime(d["scraped_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    return d


# ---------- Public API ----------

def build_report(db_path: Path, csv_path: Path, out_html: Path, start_date: datetime):
    """Ð“ÐµÐ½ÐµÑ€Ð¸Ñ€Ð°Ñ˜ HTML Ð¸Ð·Ð²ÐµÑˆÑ‚Ð°Ñ˜ (dashboard_v2.html)."""
    df = _load_df(db_path, csv_path)
    df = _prep_df(df, start_date)
    figs = _make_figures(df)
    top = _table_rows(df, limit=100)

    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html", "xml"])
    )
    tmpl = env.get_template("dashboard_v2.html.j2")

    html = tmpl.render(
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        plot_dist=figs.get("dist", ""),
        plot_scatter=figs.get("scatter", ""),
        plot_top_munis=figs.get("top_munis", ""),
        plot_volume=figs.get("volume", ""),
        rows=(top.to_dict(orient="records") if not top.empty else []),
        columns=(list(top.columns) if not top.empty else [])
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def serve_reports(port: int = 8000):
    """Ð¡ÐµÑ€Ð²Ð¸Ñ€Ð°Ñ˜ Ñ˜Ð° Ð¿Ð°Ð¿ÐºÐ°Ñ‚Ð° /reports Ð»Ð¾ÐºÐ°Ð»Ð½Ð¾ (http://localhost:<port>)."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    os.chdir(str(reports_dir))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving reports/ at http://localhost:{port}")
        httpd.serve_forever()

