"""cloud_live (the Streamlit Cloud version): precomputed examples, live CPU inference for uploads.

  pytest tests/test_cloud_live.py -v
Run in a clean venv with only streamlit_cloud/requirements.txt (+ pytest), with the pinned CLIP checkpoint in the
Hugging Face cache (HF_HUB_OFFLINE=1). Upload images are gallery examples re-encoded in memory, so they are not
byte-identical to an example and always go through live inference; no dataset is needed.
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import i18n  # noqa: E402
from test_gallery import SWEEP  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import try_to_load_from_cache  # noqa: E402
import inference as inf  # noqa: E402

if not isinstance(try_to_load_from_cache(inf.CLIP_MODEL, "pytorch_model.bin", revision=inf.CLIP_REVISION), str):
    pytest.skip("the pinned CLIP checkpoint is not in the Hugging Face cache", allow_module_level=True)

MB = 1024 * 1024
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "HEIC": "image/heic"}


def upload_bytes(fmt: str, example: int = 0) -> bytes:
    from PIL import Image
    img = Image.open(core.load_examples()[example]["path"]).convert("RGB")
    b = io.BytesIO()
    if fmt == "HEIC":
        pillow_heif = pytest.importorskip("pillow_heif")
        pillow_heif.register_heif_opener()
        img.save(b, "HEIF", quality=90)
    elif fmt == "PNG":
        img.save(b, "PNG")
    else:
        img.save(b, "JPEG", quality=90)
    return b.getvalue()


@pytest.fixture
def live_app(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_MODE", "cloud_live")
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def run_script(code: str, cwd=ROOT, env_extra=None, timeout=900, payload=None):
    """Run code in a fresh interpreter; payload (any picklable object) arrives on stdin, never via a file."""
    import pickle
    env = {k: v for k, v in os.environ.items() if k != "APP_MODE"} | {"HF_HUB_OFFLINE": "1"} | (env_extra or {})
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, timeout=timeout,
                         input=pickle.dumps(payload))
    out.stdout, out.stderr = out.stdout.decode(errors="replace"), out.stderr.decode(errors="replace")
    return out


@pytest.mark.parametrize("start", ["repository root", "another folder"])
def test_cloud_entry_point_runs_cloud_live_and_renders_every_example(start, tmp_path):
    """The entry point's default mode; 12 examples x EN/zh-TW x R1/R3 + Compare from precomputed results;
    uploads enabled; no error, no logged failure, no deprecation warning."""
    entry = str(ROOT / "streamlit_cloud" / "streamlit_app.py")
    errors = [i18n.t(lang, "error_image") for lang in i18n.LANGS]
    out = run_script(SWEEP.format(ERR=errors, entry=entry, uploads=True),
                     cwd=ROOT if start == "repository root" else tmp_path)
    assert out.returncode == 0 and out.stdout.strip().endswith("swept ok"), out.stderr[-4000:]
    log = out.stderr.lower()
    assert "failed" not in log and "deprecat" not in log and "will be removed" not in log, out.stderr[-3000:]


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_upload_runs_live_inference(live_app, lang, fmt):
    at = live_app
    at.radio(key="lang").set_value(lang).run()
    up = at.file_uploader(key="upload")
    assert len(at.get("file_uploader")) == 1 and up.label == i18n.t(lang, "upload_change")  # one control, at the top
    if fmt == "HEIC":
        assert ".heic" in up.allowed_type and ".heif" in up.allowed_type
    from html import escape
    notes = " ".join(m.value for m in at.markdown)  # beside the uploader, with the scope notice
    assert escape(i18n.t(lang, "upload_privacy")) in notes and escape(i18n.t(lang, "scope_note", emotions="")[:20]) in notes
    for mode in core.MODES:
        at.sidebar.radio(key="model").set_value(mode).run()
        at.file_uploader(key="upload").set_value((f"photo.{fmt.lower()}", upload_bytes(fmt), FORMATS[fmt])).run()
        assert not at.exception and not at.error, (at.exception, [e.value for e in at.error])
        html = " ".join(m.value for m in at.markdown)
        assert "af-card af-result" in html and i18n.t(lang, "total_predicted") in html
        caps = [c.value for c in at.caption]
        assert any(c.startswith(i18n.t(lang, "inference_time", seconds=0, device="CPU")[:6]) and "CPU" in c for c in caps)
        assert not any("Precomputed" in c or "預先計算" in c for c in caps)  # live, not the gallery file
        assert (i18n.t(lang, "also_feel") in html) == (mode == "r3")


def test_examples_stay_precomputed_and_match(live_app):
    at = live_app
    at.button(key="ex_4").click().run()
    assert not at.exception and not at.error
    caps = [c.value for c in at.caption]
    assert any(c.startswith("Precomputed with the full pipeline") for c in caps)


def test_upload_limits(live_app):
    at = live_app
    at.file_uploader(key="upload").set_value(("big.jpg", b"\xff" * (10 * MB + 1), "image/jpeg")).run()
    assert [e.value for e in at.error] == [i18n.t("en", "error_too_large")]
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (8000, 7000), "white").save(b, "PNG")  # 56 MP, a few hundred KB
    assert len(b.getvalue()) < 10 * MB
    at.file_uploader(key="upload").set_value(("huge.png", b.getvalue(), "image/png")).run()
    assert [e.value for e in at.error] == [i18n.t("en", "error_too_large")]
    at.file_uploader(key="upload").set_value(("junk.jpg", b"not an image at all", "image/jpeg")).run()
    assert [e.value for e in at.error] == [i18n.t("en", "error_image")]


def test_large_png_is_downscaled_like_a_jpeg_draft():
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (4100, 2100), "gray").save(b, "PNG")
    assert core.load_upload(b.getvalue()).size == (1025, 525)  # /4: both sides stay >= 512, as JPEG draft does
    j = io.BytesIO(); Image.new("RGB", (4100, 2100), "gray").save(j, "JPEG")
    assert core.load_upload(j.getvalue()).size == (1025, 525)


AUDIT = r"""
import io, os, pickle, sys
WARM, UPLOADS = pickle.loads(sys.stdin.buffer.read())
sys.path.insert(0, {app!r})
from streamlit.testing.v1 import AppTest
writes = []
WATCH = {{"os.rename", "os.replace", "os.remove", "os.unlink", "os.truncate", "shutil.copyfile", "shutil.move",
          "tempfile.mkstemp", "tempfile.mkdtemp", "os.mkdir"}}
def hook(event, args):
    if event == "open":
        path, mode = args[0], args[1]
        if isinstance(mode, str) and any(c in mode for c in "wax+") and not str(path).startswith("/dev/"):
            writes.append(("open", str(path), mode))
        elif isinstance(mode, int) and mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(("open", str(path), mode))
    elif event in WATCH:
        writes.append((event, str(args[0]) if args else ""))
at = AppTest.from_file({entry!r}, default_timeout=900); at.run()
at.file_uploader(key="upload").set_value(("warmup.jpg", WARM, "image/jpeg")).run()  # loads the engine
assert not at.exception and not at.error, (at.exception, at.error)
sys.addaudithook(hook)
for name, data, mime in UPLOADS:
    at.file_uploader(key="upload").set_value((name, data, mime)).run()
    assert not at.exception and not at.error, (name, at.exception, [e.value for e in at.error])
    assert any("af-card af-result" in m.value for m in at.markdown)
print("WRITES", writes)
# VmHWM is this process's own peak; getrusage's ru_maxrss would include the forking pytest process (kept across exec)
print("PEAK_MB", int(next(l for l in open("/proc/self/status") if l.startswith("VmHWM")).split()[1]) / 1024)
"""


def test_uploads_never_touch_disk_or_logs():
    """After the engine is loaded, uploading JPEG, PNG and HEIC opens no file for writing (audit hook on open,
    rename, remove, temp files, ...), and the upload file names never appear in the log output."""
    uploads = [(f"secret-holiday-{f.lower()}.{f.lower()}", upload_bytes(f, 3), m) for f, m in FORMATS.items()]
    code = AUDIT.format(app=str(APP), entry=str(ROOT / "streamlit_cloud" / "streamlit_app.py"))
    out = run_script(code, payload=(upload_bytes("JPEG", 1), uploads))
    assert out.returncode == 0, out.stderr[-4000:]
    writes = next(l for l in out.stdout.splitlines() if l.startswith("WRITES"))
    assert writes == "WRITES []", writes
    assert "secret-holiday" not in out.stderr and "secret-holiday" not in out.stdout.replace(writes, "")


def test_peak_memory_three_uploads_under_2gb():
    """A fresh process: page, then 3 different uploads in a row (engine loaded on the first); peak RSS < 2.0 GB."""
    uploads = [(f"m{i}.jpg", upload_bytes("JPEG", i), "image/jpeg") for i in (5, 6, 7)]
    code = AUDIT.format(app=str(APP), entry=str(ROOT / "streamlit_cloud" / "streamlit_app.py"))
    out = run_script(code, payload=(upload_bytes("JPEG", 8), uploads))
    assert out.returncode == 0, out.stderr[-4000:]
    peak = float(next(l for l in out.stdout.splitlines() if l.startswith("PEAK_MB")).split()[1])
    print(f"peak RSS {peak:.0f} MB")
    assert peak < 2000, f"peak RSS {peak:.0f} MB"


def test_vision_only_engine_equals_full_pipeline_on_cpu():
    """The cloud_live engine (no text tower, precomputed text embeddings) gives exactly the full CPU pipeline's
    output on two examples (app/verify_cloud_live.py checks 20 validation images on the server)."""
    pytest.importorskip("transformers")
    live, full = inf.load_engine(device="cpu", vision_only=True), inf.load_engine(device="cpu")
    for i in (2, 9):
        img = inf.load_image(core.load_examples()[i]["path"])
        for mode, prior in (("r1", False), ("r1", True), ("r3", False)):
            a, b = inf.predict(live, img, mode=mode, prior_correction=prior), inf.predict(full, img, mode=mode,
                                                                                         prior_correction=prior)
            assert np.abs(a["probs"] - b["probs"]).max() < 1e-5 and np.array_equal(a["maps"], b["maps"])
            assert a["description"] == b["description"] and a["top"] == b["top"]


def test_cloud_requirements():
    req = [l.strip() for l in (ROOT / "streamlit_cloud/requirements.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    names = {l.split("==")[0].split(" @ ")[0].strip().lower() for l in req}
    assert {"torch", "transformers", "pillow-heif", "streamlit", "numpy", "pandas", "pillow", "plotly"} <= names
    torch_line = next(l for l in req if l.startswith("torch"))
    assert "download.pytorch.org/whl/cpu/" in torch_line and "%2Bcpu-cp312" in torch_line  # CPU-only wheel
    assert not any("nvidia" in l or "+cu" in l for l in req)


def test_no_deprecated_components_api():
    for f in APP.glob("*.py"):
        assert "components.v1" not in f.read_text(encoding="utf-8").replace("deprecated components.html", ""), f


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_example_and_upload_replace_each_other(live_app, lang):
    """Picking a gallery example clears the uploaded file; uploading replaces the example. A "Showing" line always
    names the image the result belongs to."""
    from html import escape
    import ui
    at = live_app
    at.radio(key="lang").set_value(lang).run()
    at.file_uploader[0].set_value(("my_photo.jpg", upload_bytes("JPEG", 2), "image/jpeg")).run()
    html = " ".join(m.value for m in at.markdown)
    assert f'af-showing">{escape(i18n.t(lang, "showing_upload", name="my_photo.jpg"))}<' in html
    at.button(key="ex_5").click().run()
    assert not at.exception and len(at.get("file_uploader")) == 1
    assert at.file_uploader[0].value is None, "the uploaded file must be cleared when an example is picked"
    ex = core.load_examples()[5]
    cap = ui.caption_text(*ui.artwork_parts(ex["artist"], ex["painting"], i18n.t(lang, "untitled")))
    html = " ".join(m.value for m in at.markdown)
    assert f'af-showing">{escape(i18n.t(lang, "showing_example", caption=cap))}<' in html
    assert "my_photo.jpg" not in html
    at.file_uploader[0].set_value(("second.png", upload_bytes("PNG", 3), "image/png")).run()
    html = " ".join(m.value for m in at.markdown)
    assert f'af-showing">{escape(i18n.t(lang, "showing_upload", name="second.png"))}<' in html
    assert 'class="af-selected"' not in html  # no gallery card carries the "Showing" badge any more

