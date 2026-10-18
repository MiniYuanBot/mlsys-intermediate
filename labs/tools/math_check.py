#!/usr/bin/env python3
"""Check LaTeX math and diagram rules in tracked Markdown.

Scan (stdlib only):
  M001 Unicode math character (use a LaTeX command inside $...$)
  M002 \\( \\) \\[ \\] used as math delimiters (use $...$ / $$...$$)
  M003 unbalanced $ or $$ in a paragraph
  M004 inline math opened with whitespace after $ or closed after whitespace
  M005 arrow character inside math (use \\to, \\gets, \\Rightarrow, \\leftrightarrow)
  M006 unescaped | inside math in a table row (breaks the table; write \\vert or \\|)
  M010 CJK text inside an ASCII or Mermaid diagram (diagrams are English-only)
  M020 KaTeX failed to parse a formula (only with --render)
  W001 bare-text math outside $...$ (heuristic warning)
  W002 math written as inline code, e.g. `O(n)` or `T_d` (heuristic warning)
  W020 KaTeX strict-mode warning (only with --render)

Fenced code, inline code, front matter, HTML comments, URLs and link targets are
skipped. Put <!-- math-ok --> on a line to allow Unicode math characters there
(e.g. a proper name). Run from anywhere; paths are reported relative to the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "labs"
SCOPES = ("course/", "labs/", "solutions/")
TOP_FILES = ("README.md",)

GREEK = "".join(chr(c) for c in range(0x391, 0x3AA) if c != 0x3A2) + "".join(
    chr(c) for c in range(0x3B1, 0x3CA)
)
GREEK += "ϑϕϵς"  # theta/phi/epsilon variants, final sigma
SUPERSUB = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀᵢⱼₖₗₘₙₜ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
MATH_CHARS = set("×÷≤≥≠≈∼≃≅≡±∓√∛∑∏∫∞∈∉∋⊂⊆⊃⊇∪∩∀∃∄∂∇ℓ−′″∝∠⌈⌉⌊⌋‖") | set(GREEK) | set(SUPERSUB)
ARROWS = set("→←⇒⇐↔⇔↦⟶⟵⟹⟸⟷↑↓")
LATEX_HINT = {
    "×": r"\times",
    "÷": r"\div",
    "≤": r"\le",
    "≥": r"\ge",
    "≠": r"\ne",
    "≈": r"\approx",
    "∼": r"\sim",
    "≃": r"\simeq",
    "≅": r"\cong",
    "≡": r"\equiv",
    "±": r"\pm",
    "∓": r"\mp",
    "√": r"\sqrt{}",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∞": r"\infty",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∀": r"\forall",
    "∃": r"\exists",
    "∂": r"\partial",
    "∇": r"\nabla",
    "ℓ": r"\ell",
    "−": "-",
    "′": "'",
    "·": r"\cdot",
    "∝": r"\propto",
    "‖": r"\|",
    "ᵀ": r"^\top",
    "→": r"\to",
    "←": r"\gets",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "↔": r"\leftrightarrow",
    "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
}
GREEK_NAMES = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
    "sigma tau upsilon phi chi psi omega"
).split()
for _i, _name in enumerate(GREEK_NAMES):
    LATEX_HINT.setdefault(chr(0x3B1 + _i + (1 if _i >= 17 else 0)), "\\" + _name)

CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
BOX = re.compile(r"[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╭╮╯╰▶◀▲▼]|\+-{2,}|-{2,}\+|-->|<--|==>")
DIAGRAM_LANGS = {"", "text", "txt", "ascii", "plain", "plaintext", "diagram"}
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
QUOTE = re.compile(r"^(\s*>\s?)+")
ALLOW = "<!-- math-ok -->"
CODE_MATH = re.compile(r"(?<!`)`(O\([^`]*\)|[A-Za-z]_\{?[A-Za-z0-9]{1,3}\}?)`(?!`)")
BARE = [
    (re.compile(r"(?<![\w$\\])O\([^()$`\n]*[A-Za-z][^()$`\n]*\)"), "big-O expression"),
    (
        re.compile(r"(?<![\w$\\./-])[A-Za-z]_(?:\{[^}]+\}|[A-Za-z0-9])(?![\w./-])"),
        "subscripted variable",
    ),
    (re.compile(r"(?<![\w$\\./])[A-Za-z0-9)]\^(?:\{[^}]+\}|-?[A-Za-z0-9]+)(?![\w./])"), "power"),
]


@dataclass
class Issue:
    path: str
    line: int
    col: int
    code: str
    message: str

    @property
    def is_error(self):
        return not self.code.startswith("W")

    def __str__(self):
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


@dataclass
class MathSpan:
    path: str
    line: int
    col: int
    expr: str
    display: bool


@dataclass
class Result:
    issues: list = field(default_factory=list)
    spans: list = field(default_factory=list)


def _mask(text, start, end, out):
    for i in range(start, end):
        if out[i] != "\n":
            out[i] = " "


def mask_inline(text):
    """Blank inline code, HTML comments, link targets and bare URLs (columns preserved)."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n and text[i + 1] == "`":
            i += 2
            continue
        if text[i] == "`":
            j = i
            while j < n and text[j] == "`":
                j += 1
            run = text[i:j]
            close = re.compile(r"(?<!`)" + re.escape(run) + r"(?!`)")
            m = close.search(text, j)
            if m:
                _mask(text, i, m.end(), out)
                i = m.end()
                continue
            i = j
            continue
        i += 1
    masked = "".join(out)
    for pattern in (
        r"<!--.*?-->",
        r"\]\((?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?\)",
        r"\]\[[^\]]*\]",
        r"^\s*\[[^\]]+\]:\s*\S+.*$",
        r"<https?://[^>]*>",
        r"\bhttps?://[^\s)\]>|]+",
        r"<[A-Za-z][^<>\n]*?\s[a-z-]+=\"[^\"]*\"[^<>\n]*>",
    ):
        for m in re.finditer(pattern, masked, re.S | re.M):
            keep = 1 if m.group(0).startswith("](") or m.group(0).startswith("][") else 0
            _mask(masked, m.start() + keep, m.end(), out)
        masked = "".join(out)
    return masked


def blocks(lines):
    """Yield ('fence', info, [(lineno, text)]) or ('prose', [(lineno, prefix_len, text)])."""
    i = 0
    n = len(lines)
    if lines and lines[0].strip() == "---":
        for j in range(1, n):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    prose = []
    in_comment = False
    while i < n:
        raw = lines[i]
        q = QUOTE.match(raw)
        plen = q.end() if q else 0
        body = raw[plen:]
        if in_comment:
            end = body.find("-->")
            if end < 0:
                prose.append((i + 1, plen, " " * len(body)))
                i += 1
                continue
            prose.append((i + 1, plen, " " * (end + 3) + body[end + 3 :]))
            in_comment = False
            i += 1
            continue
        start = body.rfind("<!--")
        if start >= 0 and body.find("-->", start) < 0:
            prose.append((i + 1, plen, body[:start] + " " * (len(body) - start)))
            in_comment = True
            i += 1
            continue
        m = FENCE.match(body)
        if m:
            if prose:
                yield ("prose", prose)
                prose = []
            marker = m.group(2)
            info = m.group(3).strip().split()
            lang = info[0].lower().strip("{}.") if info else ""
            content = []
            i += 1
            while i < n:
                q2 = QUOTE.match(lines[i])
                b2 = lines[i][q2.end() if q2 else 0 :]
                m2 = FENCE.match(b2)
                if (
                    m2
                    and m2.group(2)[0] == marker[0]
                    and len(m2.group(2)) >= len(marker)
                    and not m2.group(3).strip()
                ):
                    break
                content.append((i + 1, b2))
                i += 1
            yield ("fence", lang, content)
            i += 1
            continue
        if not body.strip():
            prose.append((i + 1, plen, ""))
        else:
            prose.append((i + 1, plen, body))
        i += 1
    if prose:
        yield ("prose", prose)


def paragraphs(prose):
    """Split prose lines into paragraphs; an open $$ block extends across blank lines."""
    cur = []
    open_display = False
    for item in prose:
        text = item[2]
        if not text.strip() and not open_display:
            if cur:
                yield cur
                cur = []
            continue
        cur.append(item)
        masked = mask_inline(text)
        # Count unescaped $$ tokens to know whether a display block stays open.
        count = len(re.findall(r"(?<!\\)\$\$", masked))
        if count % 2:
            open_display = not open_display
    if cur:
        yield cur


def scan_text(path, text):
    res = Result()
    lines = text.splitlines()
    for kind, *rest in blocks(lines):
        if kind == "fence":
            lang, content = rest
            check_diagram(path, lang, content, res)
        else:
            for para in paragraphs(rest[0]):
                scan_paragraph(path, para, lines, res)
    return res


def check_diagram(path, lang, content, res):
    if lang == "mermaid":
        is_diagram = True
    elif lang in DIAGRAM_LANGS:
        is_diagram = any(BOX.search(t) for _, t in content)
    else:
        is_diagram = False
    if not is_diagram:
        return
    for lineno, t in content:
        m = CJK.search(t)
        if m:
            what = "Mermaid" if lang == "mermaid" else "ASCII"
            res.issues.append(
                Issue(
                    path,
                    lineno,
                    m.start() + 1,
                    "M010",
                    f"CJK text in {what} diagram; labels must be English",
                )
            )


def scan_paragraph(path, para, lines, res):
    # Build joined text and a position map back to (line, col).
    parts = []
    pos = []
    for lineno, plen, body in para:
        masked = mask_inline(body)
        for k, ch in enumerate(masked):
            parts.append(ch)
            pos.append((lineno, plen + k + 1))
        parts.append("\n")
        pos.append((lineno, plen + len(masked) + 1))
    text = "".join(parts)
    table_lines = {ln for ln, _, body in para if body.lstrip().startswith("|")}
    allowed = {ln for ln, _, _ in para if ALLOW in lines[ln - 1]}
    n = len(text)
    in_math = [False] * n
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in "()[]":
                ln, col = pos[i]
                res.issues.append(
                    Issue(path, ln, col, "M002", f"\\{nxt} delimiter; use $...$ or $$...$$")
                )
            i += 2
            continue
        if ch != "$":
            i += 1
            continue
        display = text.startswith("$$", i)
        opener = "$$" if display else "$"
        start = i + len(opener)
        j = start
        close = -1
        while j < n:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == "$":
                if display and text.startswith("$$", j):
                    close = j
                    break
                if not display:
                    if text.startswith("$$", j):
                        break  # a $$ cannot close inline math
                    close = j
                    break
            j += 1
        ln, col = pos[i]
        if close < 0:
            res.issues.append(
                Issue(
                    path, ln, col, "M003", f"unbalanced {opener} (no closing {opener} in paragraph)"
                )
            )
            i = start
            continue
        expr = text[start:close]
        if not display:
            if expr[:1].isspace() or expr[-1:].isspace():
                res.issues.append(
                    Issue(
                        path,
                        ln,
                        col,
                        "M004",
                        "inline math must not start or end with whitespace: $x$ not $ x $",
                    )
                )
            if not expr.strip():
                res.issues.append(Issue(path, ln, col, "M003", "empty $...$"))
        for k in range(i, close + len(opener)):
            in_math[k] = True
        for k in range(start, close):
            c = text[k]
            if c in ARROWS:
                l2, c2 = pos[k]
                res.issues.append(
                    Issue(
                        path,
                        l2,
                        c2,
                        "M005",
                        f"arrow {c!r} inside math; write {LATEX_HINT.get(c, 'a LaTeX arrow')}",
                    )
                )
            elif c in MATH_CHARS or c == "·":
                l2, c2 = pos[k]
                if l2 not in allowed:
                    res.issues.append(Issue(path, l2, c2, "M001", _char_msg(c, True)))
            elif c == "|" and pos[k][0] in table_lines and text[k - 1] != "\\":
                l2, c2 = pos[k]
                res.issues.append(
                    Issue(
                        path,
                        l2,
                        c2,
                        "M006",
                        "unescaped | inside math in a table row; write \\vert, \\lvert/\\rvert or \\|",
                    )
                )
        span_text = expr
        span_lines = {pos[k][0] for k in range(start, close)}
        if span_lines & table_lines:
            span_text = span_text.replace("\\|", "|")
        # Remove blockquote prefixes was already done; newline joins are fine for KaTeX.
        res.spans.append(MathSpan(path, ln, col, span_text.strip(), display))
        i = close + len(opener)
    # Prose checks (outside math).
    for k, c in enumerate(text):
        if in_math[k]:
            continue
        ln, col = pos[k]
        if c in MATH_CHARS and ln not in allowed:
            res.issues.append(Issue(path, ln, col, "M001", _char_msg(c, False)))
        elif c == "·" and ln not in allowed:
            before = text[k - 1] if k else " "
            after = text[k + 1] if k + 1 < n else " "
            if re.match(r"[A-Za-z0-9]", before) and re.match(r"[A-Za-z0-9]", after):
                res.issues.append(Issue(path, ln, col, "M001", _char_msg(c, False)))
    for lineno, plen, body in para:
        for m in CODE_MATH.finditer(body):
            res.issues.append(
                Issue(
                    path,
                    lineno,
                    plen + m.start() + 1,
                    "W002",
                    f"math written as inline code {m.group(0)}; use ${m.group(1)}$ (code spans are for code)",
                )
            )
    prose_only = "".join(" " if in_math[k] else ch for k, ch in enumerate(text))
    for regex, what in BARE:
        for m in regex.finditer(prose_only):
            ln, col = pos[m.start()]
            res.issues.append(
                Issue(path, ln, col, "W001", f"bare-text {what} {m.group(0)!r}; wrap it in $...$")
            )


def _char_msg(c, in_math):
    hint = LATEX_HINT.get(c)
    where = "inside math" if in_math else "in prose"
    tip = (
        f"; write ${hint}$"
        if hint and not in_math
        else (f"; write {hint}" if hint else "; use a LaTeX command")
    )
    return f"Unicode math character {c!r} (U+{ord(c):04X}) {where}{tip}"


# ---------------------------------------------------------------- file discovery


def tracked_markdown():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        names = sorted({n for n in out.split("\0") if n})
    except (OSError, subprocess.CalledProcessError):
        names = sorted(str(p.relative_to(ROOT)).replace(os.sep, "/") for p in ROOT.rglob("*.md"))
    keep = []
    for name in names:
        if (name in TOP_FILES or name.startswith(SCOPES)) and (ROOT / name).is_file():
            if any(
                part.startswith(".venv") or part in ("node_modules", "target")
                for part in name.split("/")
            ):
                continue
            keep.append(name)
    return keep


def select(paths):
    all_files = tracked_markdown()
    if not paths:
        return all_files
    chosen = []
    for p in paths:
        target = Path(p)
        if not target.is_absolute():
            target = (Path.cwd() / target) if (Path.cwd() / target).exists() else ROOT / target
        target = target.resolve()
        if target.is_file():
            # Files outside the repo (e.g. scratch drafts) keep their absolute path.
            name = target.relative_to(ROOT) if target.is_relative_to(ROOT) else target
            chosen.append(str(name).replace(os.sep, "/"))
            continue
        prefix = str(target.relative_to(ROOT)).replace(os.sep, "/").rstrip("/") + "/"
        chosen += [f for f in all_files if f.startswith(prefix) or prefix == "./"]
    return sorted(set(chosen))


# ---------------------------------------------------------------- KaTeX render

NODE_SCRIPT = r"""
const fs = require('fs');
const katex = require(process.argv[2]);
const spans = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const s of spans) {
  const warnings = [];
  try {
    katex.renderToString(s.expr, {
      throwOnError: true, displayMode: s.display,
      strict: (code, msg) => { warnings.push(code + ': ' + msg); return 'ignore'; },
    });
    out.push({ok: true, warnings});
  } catch (e) {
    out.push({ok: false, error: String(e.message || e).split('\n')[0], warnings});
  }
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def katex_version():
    for line in (LABS / "tools/formatter-versions.env").read_text(encoding="utf-8").splitlines():
        if line.startswith("KATEX="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("KATEX pin missing from tools/formatter-versions.env")


def npm_command():
    if os.name != "nt" and shutil.which("npm"):
        return [shutil.which("npm")]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from platform_tools import node_package_command  # noqa: E402

    return node_package_command("npm")


def find_katex(version, install=True):
    """Return the katex package directory for the pinned version, installing to a user cache."""
    candidates = []
    if os.environ.get("MLSYS_KATEX_DIR"):
        candidates.append(Path(os.environ["MLSYS_KATEX_DIR"]))
    candidates.append(LABS / ".venv-format/node_modules/katex")
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache_home / "mlsys" / f"katex-{version}"
    candidates.append(cache / "node_modules/katex")
    for c in candidates:
        pkg = c / "package.json"
        if pkg.is_file() and json.loads(pkg.read_text(encoding="utf-8")).get("version") == version:
            return c
    if not install:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        *npm_command(),
        "install",
        "--prefix",
        str(cache),
        "--no-save",
        "--no-audit",
        "--no-fund",
        "--loglevel=error",
        f"katex@{version}",
    ]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return cache / "node_modules/katex"


def render(spans, issues):
    node = shutil.which("node")
    if not node:
        print(
            "SKIP render: node not found (KaTeX check needs Node.js; run in WSL/Linux or install node)"
        )
        return False
    try:
        version = katex_version()
        pkg = find_katex(version)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"SKIP render: cannot prepare katex ({error})")
        return False
    with tempfile.TemporaryDirectory(prefix="mlsys-katex-") as tmp:
        tmp = Path(tmp)
        (tmp / "render.js").write_text(NODE_SCRIPT, encoding="utf-8")
        (tmp / "in.json").write_text(
            json.dumps([{"expr": s.expr, "display": s.display} for s in spans]), encoding="utf-8"
        )
        subprocess.run(
            [
                node,
                str(tmp / "render.js"),
                str(pkg.resolve()),
                str(tmp / "in.json"),
                str(tmp / "out.json"),
            ],
            check=True,
        )
        results = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    for span, r in zip(spans, results):
        if not r["ok"]:
            issues.append(
                Issue(
                    span.path,
                    span.line,
                    span.col,
                    "M020",
                    f"KaTeX {version}: {r['error'].replace(chr(0x332), '')}",
                )
            )
        for w in r["warnings"]:
            issues.append(Issue(span.path, span.line, span.col, "W020", f"KaTeX strict: {w}"))
    print(f"Rendered {len(spans)} formulas with KaTeX {version}")
    return True


# ---------------------------------------------------------------- CLI


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--paths", nargs="+", help="files or directories to check (default: all tracked docs)"
    )
    p.add_argument(
        "--render", action="store_true", help="also parse every formula with the pinned KaTeX"
    )
    p.add_argument("--warn-only", action="store_true", help="report but always exit 0")
    p.add_argument("--no-warnings", action="store_true", help="hide W-codes")
    p.add_argument("--summary", action="store_true", help="print only the per-file summary")
    a = p.parse_args(argv)
    files = select(a.paths)
    issues = []
    spans = []
    for name in files:
        res = scan_text(name, (ROOT / name).read_text(encoding="utf-8"))
        issues += res.issues
        spans += res.spans
    if a.render:
        render(spans, issues)
    if a.no_warnings:
        issues = [i for i in issues if i.is_error]
    issues.sort(key=lambda i: (i.path, i.line, i.col, i.code))
    if not a.summary:
        for issue in issues:
            print(issue)
    per_file = defaultdict(lambda: [0, 0])
    for issue in issues:
        per_file[issue.path][0 if issue.is_error else 1] += 1
    if per_file:
        print("\nSummary (errors, warnings):")
        width = max(len(f) for f in per_file)
        for f in sorted(per_file):
            e, w = per_file[f]
            print(f"  {f:<{width}}  {e:5d} {w:5d}")
    codes = Counter(i.code for i in issues)
    errors = sum(i.is_error for i in issues)
    print(
        f"Checked {len(files)} files, {len(spans)} formulas: {errors} errors, "
        f"{len(issues) - errors} warnings  "
        + " ".join(f"{c}={n}" for c, n in sorted(codes.items()))
    )
    return 1 if errors and not a.warn_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
