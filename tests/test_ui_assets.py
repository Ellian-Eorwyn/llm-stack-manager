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

    def test_fleet_loads_before_anything_that_fetches(self):
        """`fetchJSON` asks fleet.js which machine a request is for, through a
        `typeof` guard so util.js can stay first and depend on nothing. Every
        module that fetches on load must run after the answer is bound, or the
        first poll of a session goes to the wrong host."""
        position = self.order.index("fleet.js")
        self.assertEqual(position, 1, "fleet.js loads immediately after util.js")
        for name in ("shell.js", "status.js", "config.js", "models.js"):
            self.assertGreater(self.order.index(name), position,
                               f"{name} loads before the selected host is bound")

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
        """The only value the scripts still need from the server.

        `builtinChatVariants` went with the variant model: there is one unit per
        slot now, so a slot's label comes from its own config rather than from a
        table mapping three units onto one card.
        """
        for key in ("modelsDir",):
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


class FleetUiTests(unittest.TestCase):
    """The browser half of the fleet.

    Two properties are worth pinning, and neither is visual. A request must not
    silently go to the wrong machine, and a page looking at another machine must
    not look like one looking at this one.
    """

    @classmethod
    def setUpClass(cls):
        cls.template = TEMPLATE.read_text()
        cls.fleet = (STATIC / "js" / "fleet.js").read_text()
        cls.util = (STATIC / "js" / "util.js").read_text()

    def test_the_fetch_wrapper_asks_which_host_a_request_is_for(self):
        # The one mechanism. Seventy-eight call sites in the other modules do
        # not know a fleet exists.
        self.assertIn("fleetPath(url, method)", self.util)
        self.assertIn("typeof fleetPath === 'function'", self.util,
                      "util.js must stay first in the load order, so the call is guarded")

    def test_only_whitelisted_paths_are_proxied(self):
        """A list of what *is* forwarded, mirroring `web/routes/fleet.py`.

        A default of "forward it" on an unauthenticated port is how one page bug
        becomes a request to every machine in the fleet.
        """
        self.assertIn("FLEET_PROXIED", self.fleet)
        self.assertIn("FLEET_PROXIED[`${verb} ${path}`] === true", self.fleet)
        self.assertNotIn("startsWith('/api/')", self.fleet)

    def test_the_whitelist_is_keyed_by_method_as_well_as_path(self):
        """The hub's routes are method-specific and this has to match them.

        `/api/saved-configs` is a GET on the hub and has no POST, so a
        path-only whitelist rewrote Save Current into it and got a 405 whose
        HTML body made `r.json()` throw. Method-blind was not a near-miss; it
        was a different route.
        """
        self.assertIn("'GET /api/saved-configs': true", self.fleet)
        self.assertNotIn("'POST /api/saved-configs': true", self.fleet)
        self.assertIn("'POST /api/config': true", self.fleet)
        self.assertIn("'GET /api/config': true", self.fleet)

    def test_the_saved_config_controls_without_a_hub_route_are_local_only(self):
        """Set Default, Clear Default, Delete and Save Current write profiles on
        the machine this page is served from, and the hub proxies none of them.
        Unmarked, they act on the wrong machine and report success."""
        for handler in ("setDefaultSavedConfig()", "clearDefaultSavedConfig()",
                        "deleteSavedConfig()", "saveCurrentConfig()"):
            with self.subTest(handler):
                button = re.search(rf'<button([^>]*)onclick="{re.escape(handler)}"',
                                   self.template)
                self.assertIsNotNone(button, handler)
                self.assertIn("data-local-only", button.group(1))
        # Apply and the list stay: both have real hub routes.
        for handler in ("loadSavedConfig(false)", "loadSavedConfig(true)"):
            with self.subTest(handler):
                button = re.search(rf'<button([^>]*)onclick="{re.escape(handler)}"',
                                   self.template)
                self.assertNotIn("data-local-only", button.group(1))

    def test_the_two_proxied_paths_that_carry_a_name_are_anchored(self):
        """`/api/service/<n>/<a>` and `/api/saved-configs/<n>/apply` cannot be
        exact strings, so they are patterns -- and a pattern that is not
        anchored at both ends is a prefix match wearing a disguise.

        `/api/saved-configs/<name>/patch` and `/default` are local-only: the
        control API does not implement them, so a loose rule would rewrite them
        onto hub routes that do not exist.
        """
        for rule in re.findall(r"^\s*(/\^.*\$/),\s*$", self.fleet, re.M):
            with self.subTest(rule):
                self.assertTrue(rule.startswith("/^") and rule.endswith("$/"), rule)
        self.assertIn("FLEET_PROXIED_PATTERNS", self.fleet)
        self.assertNotIn("saved-configs/[^/]+$", self.fleet)

    def test_the_write_half_is_gated_on_the_hubs_own_answer(self):
        """`controllable` folds in the version check, which needs a round trip.
        The registry only knows that control was *requested*, so a page that
        read `control` would offer an editable form the hub then refuses."""
        self.assertIn("controllable", self.fleet)
        self.assertIn("fleetControllable", self.fleet)

    def test_a_tab_is_local_only_unless_it_says_otherwise(self):
        """`data-fleet-control` is additive, so the default for a new tab stays
        "acts on this machine" rather than "proxy it and hope"."""
        self.assertIn("data-fleet-control", self.template)
        self.assertIn("hasAttribute('data-fleet-control')", self.fleet)
        marked = re.findall(r'data-tab="(\w+)"[^>]*data-fleet-control', self.template)
        self.assertEqual(marked, ["config"])

    def test_a_dropped_key_is_reported_as_a_failure(self):
        """`allowed_config_keys` unions in the *target's* env file, so a hub one
        version ahead can have a setting silently discarded and still be told
        the save worked."""
        scheduling = (STATIC / "js" / "scheduling.js").read_text()
        self.assertIn("ignored_keys", scheduling)
        # Before the pre-flight override, not after: forcing overrides a
        # prediction, and a dropped key is not one.
        self.assertLess(scheduling.index("d.ignored_keys"),
                        scheduling.index("Save it anyway?"))

    def test_a_remote_secret_is_never_posted_back_as_an_empty_string(self):
        """`masked_config` reports a secret as set and never as itself, so its
        value arrives null. Left in an enabled input, the next section save
        would write an empty string over a live credential."""
        config = (STATIC / "js" / "config.js").read_text()
        self.assertIn("data-remote-skip", config)
        scheduling = (STATIC / "js" / "scheduling.js").read_text()
        self.assertIn("closest('[data-remote-skip]')", scheduling)

    def test_switching_back_to_this_machine_clears_the_other_ones_fields(self):
        config = (STATIC / "js" / "config.js").read_text()
        self.assertIn("clearRemoteConfigForm", config)
        self.assertIn("cfg-remote-extra", config)

    def test_a_reshaped_control_can_be_put_back(self):
        """Replacing a control is destructive and this page is never re-served,
        so a peer that declares a field differently would otherwise leave its
        widget behind for every host selected afterwards, including this one."""
        config = (STATIC / "js" / "config.js").read_text()
        self.assertIn("cfgOriginalControls", config)
        self.assertIn("restoreReshapedFields", config)
        # Called on the way out *and* before reconciling against the next host.
        self.assertGreaterEqual(config.count("restoreReshapedFields()"), 2)

    def test_a_field_the_peer_shapes_differently_is_rebuilt_from_its_own_words(self):
        """Two hosts can share a key set and still disagree about a field's type
        or its options -- which changes `fields_digest` and neither key set can
        see. Rendering the local widget would offer choices the target rejects."""
        config = (STATIC / "js" / "config.js").read_text()
        self.assertIn("sameShapeAsDeclared", config)
        self.assertIn("fields_digest", (STATIC / ".." / "control_api.py").read_text())

    def test_the_selection_does_not_outlive_the_tab(self):
        """A remote selection that survives a restart, or that can be
        bookmarked and shared, is how someone edits the wrong machine believing
        it is theirs."""
        self.assertIn("sessionStorage.setItem", self.fleet)
        # Use, not mention: the comment above it names localStorage to say why
        # it is not the one being used.
        self.assertNotIn("localStorage.", self.fleet)

    def test_a_remote_host_is_visible_in_the_page_itself(self):
        for marker in ("fleet-remote", "fleet-banner", "fleet-picker"):
            with self.subTest(marker):
                self.assertIn(marker, self.template)

    def test_the_picker_is_hidden_until_there_is_a_fleet(self):
        self.assertIn('id="fleet-picker" hidden', self.template)

    def test_the_tabs_that_act_on_this_machine_say_so(self):
        """Setup, the installers, the config form and the log stream all reach
        this host. Marked in the markup rather than listed in JavaScript, so a
        new tab is local-only unless someone decides otherwise."""
        marked = re.findall(r'data-tab="(\w+)" data-local-only', self.template)
        self.assertEqual(sorted(marked),
                         ["config", "logs", "playwright", "searxng", "setup", "transcribe"])

    def test_one_predicate_decides_whether_a_tab_may_open(self):
        """There were two, and they disagreed.

        `applyFleetMode` honoured the `data-fleet-control` exception and enabled
        the Configuration tab for a writable peer; `showTab` looked only at
        `data-local-only` and refused it. The remote config form was reachable
        by neither the operator nor a click -- present, tested, and unopenable.
        """
        shell = (STATIC / "js" / "shell.js").read_text()
        self.assertIn("fleetBlocks(button)", shell)
        self.assertNotIn("dataset.localOnly", shell,
                         "showTab must not re-derive the answer fleet.js already gives")
        self.assertIn("function fleetBlocks(", self.fleet)
        # The exception has to be inside the one predicate, or the two drift again.
        blocks = self.fleet[self.fleet.index("function fleetBlocks("):]
        self.assertIn("data-fleet-control", blocks[:blocks.index("\n}")])

    def test_the_controls_that_act_on_this_machine_are_marked_too(self):
        """Not just the tabs.

        `bulkAction('stop')` and `updateApp` are not proxied, so a live Stop All
        on a page whose header names another machine would stop the local stack
        -- which is exactly the failure the picker exists to prevent. The deploy
        badge is marked for a quieter reason: it reports this checkout's git
        state, which is not the peer's.
        """
        for marker in ('class="quick-bar" data-local-only',
                       'data-local-only="1" onclick="updateApp',
                       'id="deploy-badge" data-state="current" data-local-only',
                       'id="group-telemetry" data-local-only'):
            with self.subTest(marker):
                self.assertIn(marker, self.template)

    def test_hiding_by_attribute_actually_hides(self):
        """The user agent's `[hidden] { display: none }` loses to any rule that
        sets `display` on the element itself, and most panels here set one.

        Without the override this is not a style nit: `.quick-bar` is
        `display: flex`, so `hidden` on it did nothing at all, and Stop All
        stayed clickable on a page pointed at another machine.
        """
        self.assertIn("[hidden] { display: none !important; }", self.template)

    def test_the_poll_does_not_ask_a_peer_for_endpoints_it_cannot_serve(self):
        # Four unproxied endpoints, four 404s every five seconds otherwise.
        status = (STATIC / "js" / "status.js").read_text()
        self.assertIn("if (remote) return;", status)
        self.assertIn("renderRemoteServices(d)", status)
