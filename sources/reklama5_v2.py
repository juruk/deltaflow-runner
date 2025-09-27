# sources/reklama5_v2.py
from __future__ import annotations
from datetime import datetime
import httpx
from selectolax.parser import HTMLParser

WAREHOUSES_URL = "https://www.reklama5.mk/Search/Index?cat=159&city=1"  # магацини, Скопје
# За плацови можеш да додадеш посебна URL ако ја селектира категоријата за земјиште

def _parse_listing(card) -> dict:
    a = card.css_first("a[href^='/Ad/Details']")
    if not a: return {}
    url = "https://www.reklama5.mk" + a.attributes.get("href","")
    title = (card.css_first(".adTitle") or a).text(strip=True)
    price_eur = None
    price_el = card.css_first(".price")
    if price_el:
        t = price_el.text(strip=True).replace(".", "").replace(" ", "")
        if "€" in t:
            try: price_eur = float(t.replace("€",""))
            except: pass
        elif "ден" in t.lower():
            try: price_eur = float(t.lower().replace("ден",""))/61.5
            except: pass
    area_m2 = None
    # Пробај да извлечеш квадратура ако ја има
    return {
        "source": "reklama5",
        "url": url,
        "title": title,
        "category": "warehouse",
        "municipality": "Скопје",
        "area_m2": area_m2,
        "price_eur": price_eur,
        "scraped_at": datetime.utcnow().isoformat()
    }

def crawl_reklama5(start_date, limit=200):
    out = []
    with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0 (crawler non-commercial)"}) as client:
        r = client.get(WAREHOUSES_URL)
        r.raise_for_status()
        html = HTMLParser(r.text)
        cards = html.css(".aditem, .ad-box, .aditem-row") or []
        for c in cards[:limit]:
            row = _parse_listing(c)
            if row and row.get("url"): out.append(row)
    return out
