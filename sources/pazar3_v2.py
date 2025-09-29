# sources/pazar3_v2.py
from __future__ import annotations
from datetime import datetime
import httpx
from selectolax.parser import HTMLParser

WAREHOUSES_URL = "https://www.pazar3.mk/oglasi/rabota-biznis/deloven-prostor/magacin/Region A"
LAND_URL       = "https://www.pazar3.mk/oglasi/zivealista/placovi-nivi-farmi/Region A"

def _parse_listing(card) -> dict:
    title_el = card.css_first("a.title, a.title-link, a[href*='/oglas/']")
    if not title_el: return {}
    url = "https://www.pazar3.mk" + title_el.attributes.get("href","")
    title = title_el.text(strip=True)

    # Ñ†ÐµÐ½Ð° (ÐµÐ²ÐµÐ½Ñ‚ÑƒÐ°Ð»Ð½Ð¾ "Ð”Ð¾Ð³Ð¾Ð²Ð¾Ñ€")
    price_text = (card.css_first(".price") or card.css_first(".ad-price") or card.css_first(".price-tag"))
    price_eur = None
    if price_text:
        t = price_text.text(strip=True)
        # ÐµÐ´Ð½Ð¾ÑÑ‚Ð°Ð²Ð½Ð¸ Ð¸Ð·Ð²Ð»ÐµÐºÑƒÐ²Ð°ÑšÐ°; Ð¿Ð¾ Ð¿Ð¾Ñ‚Ñ€ÐµÐ±Ð° Ð¿Ñ€Ð¾ÑˆÐ¸Ñ€Ð¸ regex
        t = t.replace(".", "").replace(" ", "")
        if "â‚¬" in t:
            try: price_eur = float(t.replace("â‚¬",""))
            except: pass
        elif "Ð´ÐµÐ½" in t.lower():
            # Ð³Ñ€ÑƒÐ±Ð° ÐºÐ¾Ð½Ð²ÐµÑ€Ð·Ð¸Ñ˜Ð° ÑÐ¾ Ñ„Ð¸ÐºÑ 61.5
            try: price_eur = float(t.lower().replace("Ð´ÐµÐ½","")) / 61.5
            except: pass

    # Ð¿Ð¾Ð²Ñ€ÑˆÐ¸Ð½Ð° (Ð°ÐºÐ¾ Ðµ Ð½Ð°Ð²ÐµÐ´ÐµÐ½Ð° Ð²Ð¾ Ð½Ð°ÑÐ»Ð¾Ð²/Ð¾Ð¿Ð¸Ñ)
    area_m2 = None
    subtitle = card.css_first(".subtitle, .ad-subtitle, .desc")
    if subtitle:
        st = subtitle.text(strip=True)
        # Ð½Ð°Ñ˜Ð¿Ñ€Ð¾ÑÑ‚Ð¾ Ð±Ð°Ñ€Ð°ÑšÐµ Ð½Ð° "m2", "mÂ²"
        import re
        m = re.search(r"(\d{2,6})\s*(m2|mÂ²|ÐºÐ²|ÐºÐ²Ð°Ð´Ñ€Ð°Ñ‚)", st, flags=re.I)
        if m: area_m2 = float(m.group(1))

    return {
        "source": "pazar3",
        "url": url,
        "title": title,
        "category": "warehouse" if "magacin" in url else "land",
        "transaction": None,
        "municipality": "Ð¡ÐºÐ¾Ð¿Ñ˜Ðµ",
        "area_m2": area_m2,
        "price_eur": price_eur,
        "scraped_at": datetime.utcnow().isoformat()
    }

def _scrape(url, limit=200):
    out = []
    with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0 (crawler non-commercial)"}) as client:
        r = client.get(url)
        r.raise_for_status()
        html = HTMLParser(r.text)
        cards = html.css(".ad, .classified, .ad-box") or []
        for c in cards[:limit]:
            row = _parse_listing(c)
            if row and row.get("url"): out.append(row)
    return out

def crawl_pazar3(start_date, limit=200):
    rows = []
    rows += _scrape(WAREHOUSES_URL, limit=limit//2)
    rows += _scrape(LAND_URL,       limit=limit//2)
    return rows

