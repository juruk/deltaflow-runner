# sources/reklama5_v2.py
from __future__ import annotations
from datetime import datetime
import httpx
from selectolax.parser import HTMLParser

WAREHOUSES_URL = "https://www.reklama5.mk/Search/Index?cat=159&city=1"  # Ð¼Ð°Ð³Ð°Ñ†Ð¸Ð½Ð¸, Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ
# Ð—Ð° Ð¿Ð»Ð°Ñ†Ð¾Ð²Ð¸ Ð¼Ð¾Ð¶ÐµÑˆ Ð´Ð° Ð´Ð¾Ð´Ð°Ð´ÐµÑˆ Ð¿Ð¾ÑÐµÐ±Ð½Ð° URL Ð°ÐºÐ¾ Ñ˜Ð° ÑÐµÐ»ÐµÐºÑ‚Ð¸Ñ€Ð° ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ñ˜Ð°Ñ‚Ð° Ð·Ð° Ð·ÐµÐ¼Ñ˜Ð¸ÑˆÑ‚Ðµ

def _parse_listing(card) -> dict:
    a = card.css_first("a[href^='/Ad/Details']")
    if not a: return {}
    url = "https://www.reklama5.mk" + a.attributes.get("href","")
    title = (card.css_first(".adTitle") or a).text(strip=True)
    price_eur = None
    price_el = card.css_first(".price")
    if price_el:
        t = price_el.text(strip=True).replace(".", "").replace(" ", "")
        if "â‚¬" in t:
            try: price_eur = float(t.replace("â‚¬",""))
            except: pass
        elif "Ð´ÐµÐ½" in t.lower():
            try: price_eur = float(t.lower().replace("Ð´ÐµÐ½",""))/61.5
            except: pass
    area_m2 = None
    # ÐŸÑ€Ð¾Ð±Ð°Ñ˜ Ð´Ð° Ð¸Ð·Ð²Ð»ÐµÑ‡ÐµÑˆ ÐºÐ²Ð°Ð´Ñ€Ð°Ñ‚ÑƒÑ€Ð° Ð°ÐºÐ¾ Ñ˜Ð° Ð¸Ð¼Ð°
    return {
        "source": "reklama5",
        "url": url,
        "title": title,
        "category": "warehouse",
        "municipality": "Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ",
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

