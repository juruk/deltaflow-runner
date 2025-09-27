# sources/pazar3_v2.py
from __future__ import annotations
from datetime import datetime
import httpx
from selectolax.parser import HTMLParser

WAREHOUSES_URL = "https://www.pazar3.mk/oglasi/rabota-biznis/deloven-prostor/magacin/skopje"
LAND_URL       = "https://www.pazar3.mk/oglasi/zivealista/placovi-nivi-farmi/skopje"

def _parse_listing(card) -> dict:
    title_el = card.css_first("a.title, a.title-link, a[href*='/oglas/']")
    if not title_el: return {}
    url = "https://www.pazar3.mk" + title_el.attributes.get("href","")
    title = title_el.text(strip=True)

    # цена (евентуално "Договор")
    price_text = (card.css_first(".price") or card.css_first(".ad-price") or card.css_first(".price-tag"))
    price_eur = None
    if price_text:
        t = price_text.text(strip=True)
        # едноставни извлекувања; по потреба прошири regex
        t = t.replace(".", "").replace(" ", "")
        if "€" in t:
            try: price_eur = float(t.replace("€",""))
            except: pass
        elif "ден" in t.lower():
            # груба конверзија со фикс 61.5
            try: price_eur = float(t.lower().replace("ден","")) / 61.5
            except: pass

    # површина (ако е наведена во наслов/опис)
    area_m2 = None
    subtitle = card.css_first(".subtitle, .ad-subtitle, .desc")
    if subtitle:
        st = subtitle.text(strip=True)
        # најпросто барање на "m2", "m²"
        import re
        m = re.search(r"(\d{2,6})\s*(m2|m²|кв|квадрат)", st, flags=re.I)
        if m: area_m2 = float(m.group(1))

    return {
        "source": "pazar3",
        "url": url,
        "title": title,
        "category": "warehouse" if "magacin" in url else "land",
        "transaction": None,
        "municipality": "Скопје",
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
