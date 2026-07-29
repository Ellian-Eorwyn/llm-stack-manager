"""The UI's inline event handlers must resolve to a function that is loaded.

The manager's UI wires its buttons with inline `onclick=` attributes — 108 of
them — which resolve against the global scope at click time. Nothing checks that
resolution: a handler naming a function that no longer exists looks completely
normal in the markup, passes every Python test, renders without complaint, and
fails only when somebody presses the button.

That is tolerable while every function lives in one `<script>` block in
`index.html`, because nothing can move out of a file it never leaves. It stops
being tolerable the moment that block is split into `web/static/js/*.js`: a
module that is written but never added to the page, or a section moved into a
module that loads after its first caller, produces exactly this failure and
produces it silently.

So the check is deliberately whole-page rather than per-file. Handler attributes
are collected from the template *and* from the JavaScript, because half of them
live inside template literals that build service cards and model rows at
runtime; declarations are collected from every script the page loads. Where a
name is declared does not matter — that it is declared somewhere the browser
will have parsed does.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "templates" / "index.html"
STATIC = ROOT / "web" / "static"

# Jinja renders before the browser ever sees an attribute, so its expressions are
# not JavaScript and its filters are not function calls. `{{ section|replace(...) }}`
# inside an onclick would otherwise be reported as a missing handler named
# `replace`.
JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.S)

HANDLER_ATTR = re.compile(r'\bon[a-z]+\s*=\s*"([^"]*)"')
# A call whose callee is a bare identifier: `svcAction(...)` counts, `this.focus()`
# and `JSON.parse(...)` do not, because those resolve against an object rather
# than the global scope.
BARE_CALL = re.compile(r"(?<![\w.$])([A-Za-z_$][\w$]*)\s*\(")

FUNCTION_DECL = re.compile(r"(?:^|\n)\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
ARROW_DECL = re.compile(
    r"(?:^|\n)\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
)

# Names the browser supplies. A handler calling one of these is not referring to
# anything this repo has to declare.
GLOBALS_PROVIDED_BY_THE_BROWSER = {
    "alert", "confirm", "prompt", "fetch", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "encodeURIComponent", "decodeURIComponent",
    "parseInt", "parseFloat", "isNaN", "Number", "String", "Boolean", "Array",
    "Object", "JSON", "Math", "Date", "Promise", "RegExp", "Set", "Map",
    # Statement keywords that are followed by a parenthesis.
    "if", "for", "while", "switch", "catch", "return", "typeof", "new", "function",
}


def _page_sources() -> dict[str, str]:
    """Every file whose contents the loaded page is made of."""
    sources = {str(TEMPLATE.relative_to(ROOT)): TEMPLATE.read_text()}
    for path in sorted(STATIC.rglob("*.js")):
        sources[str(path.relative_to(ROOT))] = path.read_text()
    return sources


class InlineHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = _page_sources()

    def _declared_names(self) -> set[str]:
        declared: set[str] = set()
        for text in self.sources.values():
            declared.update(FUNCTION_DECL.findall(text))
            declared.update(ARROW_DECL.findall(text))
        return declared

    def _referenced_handlers(self) -> dict[str, set[str]]:
        referenced: dict[str, set[str]] = {}
        for filename, text in self.sources.items():
            for attribute in HANDLER_ATTR.findall(text):
                for name in BARE_CALL.findall(JINJA.sub("", attribute)):
                    referenced.setdefault(name, set()).add(filename)
        return referenced

    def test_every_inline_handler_resolves_to_a_loaded_function(self):
        declared = self._declared_names()
        missing = {
            name: sorted(files)
            for name, files in self._referenced_handlers().items()
            if name not in declared and name not in GLOBALS_PROVIDED_BY_THE_BROWSER
        }
        self.assertEqual(
            missing, {},
            "inline handlers name functions that no script on the page declares; "
            "a moved or unregistered module leaves the button dead",
        )

    def test_the_page_actually_wires_handlers(self):
        """Guard the guard: a regex that stops matching would pass vacuously."""
        referenced = self._referenced_handlers()
        self.assertGreater(len(referenced), 50,
                           "handler extraction found almost nothing; the regex has drifted")
        self.assertIn("svcAction", referenced)
        self.assertIn("showTab", referenced)

    def test_every_script_the_template_loads_exists_on_disk(self):
        """A module written but never added to the page is the other half of the
        same failure, and the handler check cannot see it."""
        template = self.sources[str(TEMPLATE.relative_to(ROOT))]
        referenced = re.findall(r"filename\s*=\s*'([^']+)'", template)
        self.assertTrue(referenced, "the template loads no static assets at all")
        for filename in referenced:
            self.assertTrue((STATIC / filename).is_file(),
                            f"index.html loads static/{filename}, which does not exist")

    def test_every_module_on_disk_is_loaded_by_the_page(self):
        """The reverse: a module nobody loads is dead weight that still passes
        the handler check, because its declarations are counted anyway."""
        template = self.sources[str(TEMPLATE.relative_to(ROOT))]
        loaded = set(re.findall(r"filename\s*=\s*'([^']+)'", template))
        for path in sorted((STATIC / "js").glob("*.js")):
            self.assertIn(f"js/{path.name}", loaded,
                          f"{path.name} exists but no <script> tag loads it")


class ScriptLoadOrderTests(unittest.TestCase):
    """Order matters, because these are classic scripts sharing one global scope.

    They are classic scripts on purpose: the markup wires 108 inline `onclick`
    handlers, and those resolve against the global scope, which `type="module"`
    does not populate. The cost of that choice is that load order is real —
    top-level `let`/`const` bindings are in the temporal dead zone until their
    script has run.
    """

    @classmethod
    def setUpClass(cls):
        cls.template = TEMPLATE.read_text()
        cls.order = [f[len("js/"):] for f in re.findall(r"filename\s*=\s*'(js/[^']+)'", cls.template)]

    def test_shell_loads_before_anything_that_reads_its_state(self):
        """`shell.js` declares the shared mutable state — `cfgCurrent`,
        `activeModel`, `savedConfigs`. A module running before it would hit the
        temporal dead zone rather than an undefined value."""
        self.assertIn("shell.js", self.order)
        position = self.order.index("shell.js")
        for name in ("config.js", "models.js", "status.js", "setup.js"):
            self.assertGreater(self.order.index(name), position,
                               f"{name} loads before the state it reads is declared")

    def test_boot_loads_last(self):
        """`boot()` is called at the bottom of boot.js and reaches into every
        other module, so nothing may load after it."""
        self.assertEqual(self.order[-1], "boot.js")

    def test_util_loads_first(self):
        """`escapeHtml`, `toast` and `fetchJSON` are used by every other module."""
        self.assertEqual(self.order[0], "util.js")

    def test_static_assets_are_versioned(self):
        """Without a cache key a browser keeps the previous deploy's modules and
        runs them against new markup, which fails as anything but a caching bug."""
        tags = re.findall(r"<script src=\"\{\{ url_for\('static', filename='js/[^']+'\) \}\}([^\"]*)\"", self.template)
        self.assertEqual(len(tags), len(self.order))
        for suffix in tags:
            self.assertIn("asset_version", suffix)

    def test_the_bootstrap_supplies_what_the_modules_read_from_it(self):
        """The only two values the scripts still need from the server."""
        for key in ("builtinChatVariants", "modelsDir"):
            self.assertIn(f"{key}:", self.template,
                          f"window.__STACK__ does not define {key}")
        for path in sorted((STATIC / "js").glob("*.js")):
            for used in re.findall(r"window\.__STACK__\.(\w+)", path.read_text()):
                self.assertIn(f"{used}:", self.template,
                              f"{path.name} reads window.__STACK__.{used}, which is never set")

    def test_no_javascript_is_left_inline_except_the_bootstrap(self):
        blocks = re.findall(r"<script>(.*?)</script>", self.template, re.S)
        self.assertEqual(len(blocks), 1, "expected exactly one inline block, the bootstrap")
        self.assertIn("window.__STACK__", blocks[0])
        self.assertLess(len(blocks[0]), 1000,
                        "the inline block is growing again; new code belongs in a module")


if __name__ == "__main__":
    unittest.main()
