#!/usr/bin/env python3
"""Unit tests for math_check.py (stdlib unittest): python tools/test_math_check.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import math_check  # noqa: E402


def codes(text):
    return [i.code for i in math_check.scan_text("t.md", text).issues]


def errors(text):
    return [c for c in codes(text) if not c.startswith("W")]


class GoodDocs(unittest.TestCase):
    def test_clean_math(self):
        text = (
            "---\ntitle: x × y\n---\n"
            "复杂度是 $O(n^2)$，满足 $a \\le b$。\n\n"
            "$$\nT(k)=T_d k + T_v(k)\n$$\n"
        )
        self.assertEqual(codes(text), [])

    def test_code_is_skipped(self):
        text = "`a × b` and `$x`\n\n```python\nx = a × b  # α\n$\n```\n\n~~~\n≤\n~~~\n"
        self.assertEqual(errors(text), [])

    def test_prose_arrow_and_nav_dot_allowed(self):
        text = "[← 课程索引](../index.md) · [D2 →](02.md)\n\n流程：读入 → 量化 → 写回。\n"
        self.assertEqual(codes(text), [])

    def test_urls_and_link_targets_skipped(self):
        text = "见 [论文](https://x.org/a_b?q=$1&r=α) 和 <https://e.com/×>\n"
        self.assertEqual(codes(text), [])

    def test_escaped_dollar(self):
        self.assertEqual(codes("价格 \\$5，另一个 \\$6。\n"), [])

    def test_display_in_callout(self):
        text = "> [!tip]- 核对要点\n> $$\n> a = b\n> $$\n> 结束 $x$。\n"
        res = math_check.scan_text("t.md", text)
        self.assertEqual([i.code for i in res.issues], [])
        self.assertEqual([s.display for s in res.spans], [True, False])
        self.assertEqual(res.spans[0].expr, "a = b")

    def test_allowlist_comment(self):
        self.assertEqual(errors("Ω Labs 是公司名 <!-- math-ok -->\n"), [])

    def test_html_comment_skipped(self):
        text = "<!--\n规则：不要写 × 或 ≤\n```text\n中文\n```\n-->\n正文 $x$\n"
        self.assertEqual(codes(text), [])

    def test_english_diagram(self):
        text = "```text\n+-------+    +------+\n| load  | -> | quant|\n+-------+    +------+\n```\n"
        self.assertEqual(codes(text), [])

    def test_cjk_in_non_diagram_text_block(self):
        self.assertEqual(codes("```text\n这只是一段输出\n```\n"), [])

    def test_table_pipe_escaped(self):
        self.assertEqual(errors("| a | $\\lvert x\\rvert$ |\n| - | - |\n"), [])


class BadDocs(unittest.TestCase):
    def test_unicode_in_prose(self):
        for ch in "×≤≥≈√∑ℓ−ρπμε²ᵀ":
            with self.subTest(ch=ch):
                self.assertIn("M001", codes(f"误差 a{ch}b 很小\n"))

    def test_unicode_in_math(self):
        self.assertIn("M001", codes("$a × b$\n"))
        self.assertIn("M001", codes("$\\alpha·x$\n"))

    def test_middle_dot_between_tokens(self):
        self.assertIn("M001", codes("面积 a·b\n"))

    def test_paren_bracket_delimiters(self):
        self.assertIn("M002", codes("公式 \\(x^2\\)\n"))
        self.assertIn("M002", codes("\\[\nx\n\\]\n"))

    def test_unbalanced(self):
        self.assertIn("M003", codes("单个 $x 没闭合\n"))
        self.assertIn("M003", codes("$$\nx = 1\n"))
        self.assertIn("M003", codes("$x$ 和 $y\n\n下一段 z$\n"))

    def test_whitespace_inline(self):
        self.assertIn("M004", codes("值 $ x$ 不渲染\n"))

    def test_arrow_in_math(self):
        self.assertIn("M005", codes("$k → \\infty$\n"))

    def test_pipe_in_table_math(self):
        self.assertIn("M006", codes("| a | $|x|$ |\n| - | - |\n"))

    def test_cjk_in_ascii_diagram(self):
        self.assertIn(
            "M010", codes("```text\n+---+   +---+\n| 读 | -> | 写 |\n+---+   +---+\n```\n")
        )

    def test_cjk_in_mermaid(self):
        self.assertIn("M010", codes("```mermaid\nflowchart LR\n  A[读入] --> B\n```\n"))

    def test_bare_math_warnings(self):
        c = codes("复杂度 O(n^2)，变量 x_i\n")
        self.assertIn("W001", c)
        self.assertEqual(errors("复杂度 O(n^2)，变量 x_i\n"), [])

    def test_code_span_math_warning(self):
        self.assertIn("W002", codes("复杂度 `O(n)`\n"))

    def test_positions(self):
        issue = math_check.scan_text("t.md", "line one\n\n> 前缀 a≤b\n").issues[0]
        self.assertEqual((issue.line, issue.col), (3, 7))


class Cli(unittest.TestCase):
    def test_katex_pin_present(self):
        version = math_check.katex_version()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
