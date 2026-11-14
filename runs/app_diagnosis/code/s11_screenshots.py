"""Screenshots of both app modes (run with PYTHONPATH=/tmp/artfae_ui_browser for playwright)."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8765"
OUT = Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)
STIM = Path(__file__).resolve().parents[1] / "stimuli/S2_navy_black_bands.jpg"
LABEL = {("en", "r1"): "Human-vote distribution (L/14 R1, recommended)", ("en", "r3"): "Minority-sensitive (L/14 R3)",
         ("zh", "r1"): "人類投票分布（L/14 R1，建議）", ("zh", "r3"): "少數情緒敏感（L/14 R3）"}


def settle(page, sel=".af-result", t=240_000):
    page.wait_for_selector(sel, timeout=t)
    page.wait_for_timeout(2500)
    while page.locator('[data-testid="stStatusWidget"]').count():
        page.wait_for_timeout(500)
    page.wait_for_timeout(1500)


def shot(name, lang, mode, source, page_name=None, prior=False):
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 2300})
        page.goto(URL)
        page.wait_for_selector("text=ArtFAE-Net", timeout=120_000)
        page.wait_for_timeout(1500)
        if lang == "zh":
            page.get_by_text("繁體中文", exact=True).click(); page.wait_for_timeout(1500)
        page.get_by_text(LABEL[(lang, mode)], exact=True).click(); page.wait_for_timeout(1500)
        if prior:
            page.get_by_text("Match validation class frequencies", exact=True).click(); page.wait_for_timeout(1000)
        if page_name:
            page.get_by_text(page_name, exact=True).click()
            page.wait_for_selector(".af-table", timeout=60_000); page.wait_for_timeout(2500)
        elif source == "upload":
            page.set_input_files('input[type="file"]', str(STIM)); settle(page)
        else:
            page.locator('button:has-text("Analyse"), button:has-text("分析")').nth(10).click(); settle(page)
        page.screenshot(path=str(OUT / f"{name}.png"))
        b.close()
    print("saved", name, flush=True)


if __name__ == "__main__":
    shot("r1_en_navy_black_upload", "en", "r1", "upload")
    shot("r3_en_navy_black_upload", "en", "r3", "upload")
    shot("r1_zh_example_sadness", "zh", "r1", "example")
    shot("r3_zh_example_sadness", "zh", "r3", "example")
    shot("r1_en_prior_on_navy_black", "en", "r1", "upload", prior=True)
    shot("about_en", "en", "r1", None, page_name="About")
