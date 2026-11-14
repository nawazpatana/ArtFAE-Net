"""Gallery cards in a real browser: the whole card opens the results (mouse, Enter, Space), focus is visible, hover
lifts the card; EN and zh-TW; in gallery and cloud_live mode; and the server log has no deprecation warning.

  pytest tests/test_cards_browser.py -v
Needs playwright (pip install playwright==1.58.0) with its Chromium (python -m playwright install chromium);
skipped otherwise. cloud_live also needs torch (streamlit_cloud/requirements.txt). The server is started with
this interpreter: python -m streamlit run app/streamlit_app.py.
"""
import glob
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import i18n  # noqa: E402

EXE = (glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")) or [None])[0]
LANG_PILL = {"en": "EN", "zh": "繁中"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module", params=["gallery", "cloud_live"])
def server(request, tmp_path_factory):
    if request.param == "cloud_live":
        pytest.importorskip("torch")
    port, log = free_port(), tmp_path_factory.mktemp("srv") / "server.log"
    env = {**os.environ, "APP_MODE": request.param, "HF_HUB_OFFLINE": "1"}
    with open(log, "w") as fh:
        p = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app/streamlit_app.py"),
                              "--server.headless", "true", "--server.port", str(port)], cwd=ROOT, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    yield f"http://127.0.0.1:{port}/", log, request.param
    p.terminate(); p.wait(30)


def settle(page, extra=0.6):
    page.wait_for_selector(".st-key-af_header", timeout=180_000)
    time.sleep(0.4)
    for _ in range(900):
        if not page.query_selector('[data-testid="stStatusWidgetRunningIcon"], [data-testid="stSpinner"]'):
            break
        time.sleep(0.2)
    time.sleep(extra)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=EXE) if EXE else p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Chromium for playwright: {e}")
        yield b
        b.close()


def open_page(browser, url, lang, width=1440):
    page = browser.new_context(viewport={"width": width, "height": 1600}, is_mobile=width < 600).new_page()
    page.goto(url); settle(page)
    page.locator(".st-key-lang_box label", has_text=LANG_PILL[lang]).click(); settle(page)
    return page


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("width", [1440, 390])
def test_whole_card_opens_results_on_click(server, browser, lang, width):
    url, _, _ = server
    page = open_page(browser, url, lang, width)
    for i in (2, 7):
        card = page.locator(f".st-key-card_{i}").bounding_box()
        btn = page.locator(f".st-key-ex_{i} button").bounding_box()
        assert btn["width"] >= card["width"] - 4 and btn["height"] >= card["height"] - 4, (card, btn)  # no small button
        label = page.locator(f".st-key-ex_{i} button").inner_text()
        assert label.startswith(i18n.t(lang, "card_label", caption="")[:2])  # accessible name, localised
    for frac in (0.3, 0.85):  # the image, then the title / artist line
        page.goto(url); settle(page)
        page.locator(".st-key-lang_box label", has_text=LANG_PILL[lang]).click(); settle(page)
        page.locator(".st-key-card_2").scroll_into_view_if_needed(); time.sleep(0.3)
        card = page.locator(".st-key-card_2").bounding_box()
        page.mouse.click(card["x"] + card["width"] / 2, card["y"] + card["height"] * frac); settle(page)
        assert page.query_selector(".af-result"), f"click at {frac:.0%} of the card height did not open the results"
    page.context.close()


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("key", ["Enter", " "])
def test_keyboard_opens_results_with_visible_focus(server, browser, lang, key):
    url, _, _ = server
    page = open_page(browser, url, lang)
    for _ in range(80):  # Tab until a card has the focus
        page.keyboard.press("Tab")
        if page.evaluate("document.activeElement.closest('[class*=\"st-key-ex_\"]') !== null"):
            break
    else:
        pytest.fail("Tab never reached a gallery card")
    style = page.evaluate("(() => { const c = document.activeElement.closest('[class*=\"st-key-card_\"]'); "
                          "const s = getComputedStyle(c); return [s.outlineStyle, parseFloat(s.outlineWidth)]; })()")
    assert style[0] == "solid" and style[1] >= 2, style  # visible focus outline on the card
    page.keyboard.press(key); settle(page)
    assert page.query_selector(".af-result")
    page.context.close()


def test_hover_lifts_card_and_log_is_clean(server, browser):
    url, log, _ = server
    page = open_page(browser, url, "en")
    c = page.locator(".st-key-card_0").bounding_box()
    page.mouse.move(c["x"] + 40, c["y"] + 40); time.sleep(0.6)
    assert page.evaluate("getComputedStyle(document.querySelector('.st-key-card_0')).transform") != "none"
    assert page.evaluate("getComputedStyle(document.querySelector('.st-key-card_0')).cursor") in ("pointer", "auto")
    page.context.close()
    text = Path(log).read_text().lower()
    assert "deprecat" not in text and "will be removed" not in text and "traceback" not in text, text[-2000:]
