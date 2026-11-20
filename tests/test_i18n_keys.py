"""Every i18n key the app passes to tr() / i18n.t() exists in BOTH languages (EN and zh-TW), in app/ and in the
artfae_demo/ copy. Static: the keys are collected from the source with ast, so a typo or a key added to one
language only fails here instead of as a KeyError on Streamlit Cloud.

  pytest tests/test_i18n_keys.py -v

Resolved key expressions: string constants; both branches of `a if c else b`; ERRORS[...] (the values of the module's
ERRORS dict); loop variables over tuples of constant keys; and f-strings / "prefix" + var built from the value sets in
TEMPLATES below. Any other key expression fails the test, so a new kind of dynamic key must be added here.
"""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODES = ["r1", "r3"]
TEMPLATES = {  # dynamic key template (the variable part as {}) -> every value it takes in the app
    "page_{}": ["analyse", "compare", "about"],
    "model_{}": MODES, "model_{}_plain": MODES, "model_{}_help": MODES, "row_h_{}": MODES,
    "pct_{}": ["luminance", "saturation", "warm", "cool"],
    "row_{}": ["baseline", "b32", "l14_r1", "primary"],
    "aes_g_{}": ["harmony", "balance", "complexity", "contrast"], "aes_x_{}": ["harmony", "balance", "complexity", "contrast"],
}
APP_DIRS = [d for d in (ROOT / "app", ROOT / "artfae_demo" / "app") if (d / "i18n.py").exists()]


def load_i18n(app_dir: Path):
    spec = importlib.util.spec_from_file_location(f"i18n_{app_dir.parent.name}", app_dir / "i18n.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def template_of(node):
    """f'pct_{k}' or 'model_' + x -> 'pct_{}' / 'model_{}' (one variable part), else None."""
    if isinstance(node, ast.JoinedStr):
        parts = ["{}" if isinstance(v, ast.FormattedValue) else v.value for v in node.values]
        return "".join(parts) if parts.count("{}") == 1 else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, r = node.left, node.right
        if isinstance(l, ast.Constant) and isinstance(l.value, str) and not isinstance(r, ast.Constant):
            return l.value + "{}"
    return None


def loop_strings(tree, name, lineno):
    """String constants of the innermost for-loop (above lineno) that binds `name`, from its iterable."""
    best, best_iter = None, None
    for n in ast.walk(tree):
        if not (hasattr(n, "lineno") and n.lineno <= lineno <= (n.end_lineno or n.lineno)):
            continue
        loops = [(n.target, n.iter)] if isinstance(n, ast.For) else \
            [(g.target, g.iter) for g in n.generators] if isinstance(n, (ast.GeneratorExp, ast.ListComp, ast.SetComp,
                                                                         ast.DictComp)) else []
        for target, it in loops:  # for statements and comprehensions
            if name in {t.id for t in ast.walk(target) if isinstance(t, ast.Name)}:
                if best is None or n.lineno >= best.lineno:
                    best, best_iter = n, it
    if best is None:
        return None
    return {c.value for c in ast.walk(best_iter) if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and "_" in c.value}


def module_dict_values(tree, name):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return {v.value for v in n.value.values if isinstance(v, ast.Constant)}
    return None


def collect_keys(path: Path):
    """-> (keys, unresolved): every key passed to tr(key, ...) or i18n.t(lang, key, ...) in one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys, unresolved = {}, []

    def add(k, where):
        keys.setdefault(k, where)

    def resolve(node, where):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            add(node.value, where)
        elif isinstance(node, ast.IfExp):
            resolve(node.body, where); resolve(node.orelse, where)
        elif (tpl := template_of(node)) is not None:
            if tpl not in TEMPLATES:
                unresolved.append(f"{where}: unknown key template {tpl!r} (add its values to TEMPLATES)")
            else:
                for v in TEMPLATES[tpl]:
                    add(tpl.format(v), where)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            vals = module_dict_values(tree, node.value.id)
            if not vals:
                unresolved.append(f"{where}: {ast.unparse(node)}")
            for v in vals or ():
                add(v, where)
        elif isinstance(node, ast.Name):
            vals = loop_strings(tree, node.id, node.lineno)
            if not vals:
                unresolved.append(f"{where}: {ast.unparse(node)}")
            for v in vals or ():
                add(v, where)
        else:
            unresolved.append(f"{where}: {ast.unparse(node)}")

    lambda_params = {a.arg for n in ast.walk(tree) if isinstance(n, ast.Lambda) for a in n.args.args}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn, where = n.func, f"{path.name}:{n.lineno}"
        if isinstance(fn, ast.Name) and fn.id == "tr" and n.args:
            resolve(n.args[0], where)
        elif (isinstance(fn, ast.Attribute) and fn.attr == "t" and isinstance(fn.value, ast.Name)
              and fn.value.id == "i18n" and len(n.args) > 1):
            arg = n.args[1]
            if isinstance(arg, ast.Name) and arg.id in lambda_params:  # tr = lambda key, **kw: i18n.t(L, key, **kw)
                continue
            resolve(arg, where)
    return keys, unresolved


@pytest.mark.parametrize("app_dir", APP_DIRS, ids=lambda d: str(d.relative_to(ROOT)))
def test_every_key_used_in_the_app_exists_in_both_languages(app_dir):
    i18n = load_i18n(app_dir)
    keys, unresolved = {}, []
    for f in sorted(app_dir.glob("*.py")):
        k, u = collect_keys(f)
        for key, where in k.items():
            keys.setdefault(key, where)
        unresolved += u
    assert not unresolved, "key expressions the test cannot resolve:\n" + "\n".join(unresolved)
    assert len(keys) > 100, len(keys)  # sanity: the collector found the app's keys
    for lang in ("en", "zh"):
        missing = sorted(f"{k} (used at {w})" for k, w in keys.items() if k not in i18n.STRINGS[lang])
        assert not missing, f"missing in {lang}:\n" + "\n".join(missing)
    assert set(i18n.STRINGS["en"]) == set(i18n.STRINGS["zh"])
    for key in keys:  # every template formats with balanced braces in both languages
        for lang in ("en", "zh"):
            s = i18n.STRINGS[lang][key]
            assert s.count("{") == s.count("}"), (lang, key)
            fields = set(re.findall(r"{(\w+)", s))
            other = set(re.findall(r"{(\w+)", i18n.STRINGS["zh" if lang == "en" else "en"][key]))
            assert fields == other, (key, lang, fields, other)  # same placeholders in both languages


def test_the_collector_catches_a_missing_key(tmp_path):
    """The check itself works: a key absent from zh is reported."""
    src = tmp_path / "streamlit_app.py"
    src.write_text('ERRORS = {"x": "error_image"}\ntr("showing_upload", name="a")\ntr("only_in_en")\n'
                   'tr(ERRORS[s])\nfor c in ("col_model",):\n    tr(c)\ntr("a" if x else "b_c")\n', encoding="utf-8")
    keys, unresolved = collect_keys(src)
    assert not unresolved
    assert set(keys) == {"error_image", "showing_upload", "only_in_en", "col_model", "a", "b_c"}
    i18n = load_i18n(ROOT / "app")
    assert "only_in_en" not in i18n.STRINGS["zh"]
