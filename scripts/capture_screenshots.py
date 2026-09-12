"""Regenerate the README screenshots by driving the running dashboard.

Usage — start the app first, then run this from the project root:

    venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
    venv\\Scripts\\python.exe scripts/capture_screenshots.py

Needs Playwright, which is a capture-time tool only and deliberately absent from
requirements.txt so nobody has to download a browser just to run the app:

    venv\\Scripts\\python.exe -m pip install playwright
    venv\\Scripts\\python.exe -m playwright install chromium

Five things here are less obvious than they look, each one a bug this script hit:

  * Headless Chromium ships no colour-emoji font, so every page icon renders as a ☒
    tofu box until Noto Color Emoji is pulled in as a webfont.
  * Overriding the font family then breaks Streamlit's Material icons, which are
    LIGATURE fonts — they print their literal names ("_arrow_right") as visible text
    unless exempted.
  * Panels do real work on load (news summarisation is an LLM call), so a fixed sleep
    either wastes time or catches the page blank. Each shot waits on a marker string
    that only exists once the panel has actually rendered.
  * window.scrollTo is a no-op: Streamlit scrolls an inner container, not the window.
  * A button inside a collapsed expander does not exist yet — the expander has to be
    clicked open first.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
BASE = "http://localhost:8501"
WIDTH, HEIGHT = 1500, 900

# Captured at 2x for crispness, then downscaled to this. Still sharper than GitHub's
# ~850px column needs, at roughly a quarter of the bytes.
FINAL_WIDTH = 1500

POLISH = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
* { font-family: "Source Sans Pro", -apple-system, "Segoe UI", sans-serif,
    "Noto Color Emoji", "Segoe UI Emoji" !important; }
[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
span[class*="material"], i[class*="material"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                 "Material Icons" !important; }
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], footer, #MainMenu { display: none !important; }
"""


def settle(page, seconds):
    try:
        page.wait_for_load_state("networkidle", timeout=60000)
    except Exception:  # noqa: BLE001 - a busy page never goes fully idle
        pass
    time.sleep(seconds)


def wait_for_text(page, text, timeout=120000):
    """Block until a marker string renders, so we never shoot a half-drawn page."""
    try:
        page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        print(f"    ! never saw {text!r} — capturing anyway")
        return False


def click_text(page, text, wait, *, role="button"):
    try:
        target = (page.get_by_role(role, name=text, exact=False).first if role
                  else page.get_by_text(text, exact=False).first)
        if target.is_visible(timeout=5000):
            target.click()
            settle(page, wait)
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def capture(page, path, out_name, *, expect=None, wait=8, opens=(), clicks=(),
            scroll=0, focus=None, nudge=0):
    page.goto(f"{BASE}{path}", wait_until="domcontentloaded")
    page.add_style_tag(content=POLISH)
    if expect:
        wait_for_text(page, expect)
    settle(page, wait)

    for label in opens:  # expanders must open before their contents exist
        click_text(page, label, 3, role=None)
    for label, extra in clicks:
        click_text(page, label, extra)

    page.add_style_tag(content=POLISH)  # Streamlit re-renders and drops the tag

    if focus:
        wait_for_text(page, focus, 60000)
        try:
            page.get_by_text(focus, exact=False).first.scroll_into_view_if_needed()
            time.sleep(2)
        except Exception:  # noqa: BLE001
            pass
    if scroll or nudge:
        page.mouse.move(WIDTH // 2, HEIGHT // 2)
        page.mouse.wheel(0, scroll or nudge)
        time.sleep(3)

    page.screenshot(path=str(OUT / out_name))
    print(f"  saved {out_name}")


SHOTS = [
    dict(path="/Crypto", out_name="crypto-news.png", expect="impact:"),
    dict(path="/Crypto", out_name="crypto-projects.png", expect="Top projects",
         scroll=1400),
    dict(path="/US_Stocks", out_name="stocks-financials.png", expect="52-week",
         opens=["quarterly financials"], clicks=[("Load financials", 25)],
         focus="quarterly revenue", nudge=620),
    dict(path="/Portfolio", out_name="portfolio.png", expect="Market value"),
    dict(path="/Portfolio", out_name="portfolio-risk.png", expect="Allocation",
         clicks=[("Check correlation", 25)]),
    dict(path="/Theses", out_name="theses.png", expect="Conditions being watched"),
]


def downscale():
    """Shrink the 2x captures so six screenshots stay near a megabyte."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed — leaving images at full size)")
        return
    total = 0
    for f in sorted(OUT.glob("*.png")):
        im = Image.open(f)
        if im.width > FINAL_WIDTH:
            im = im.resize((FINAL_WIDTH, int(im.height * FINAL_WIDTH / im.width)),
                           Image.LANCZOS)
        im.convert("RGB").save(f, "PNG", optimize=True)
        total += f.stat().st_size
    print(f"  downscaled to {FINAL_WIDTH}px — {total / 1024:.0f} KB total")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"capturing from {BASE} into {OUT.relative_to(ROOT)}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT},
                                device_scale_factor=2)
        for shot in SHOTS:
            capture(page, **shot)
        browser.close()
    downscale()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
