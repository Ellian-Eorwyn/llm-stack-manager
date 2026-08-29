"""One proxy script, two proxies, and the settings that were never separable.

`start-chat-proxy2.sh` was a hand copy of `start-chat-proxy.sh`, and the copy had
already drifted: only one of them exported the three model aliases, and *neither*
exported `THINK_REASONING_EFFORT` or `CODE_REASONING_EFFORT` -- latent only
because systemd hands the whole env file to the unit, so the loss showed up just
for whoever ran the script by hand.

The thing that could not be expressed at all was per-slot tuning. Both scripts
read the same `THINK_TEMP`, so two slots serving different models shared every
sampling setting, and there was no key anyone could set to separate them.

The tests that matter here are the two directions of that: a host which has only
ever set the shared key sees no change, and a host that sets the per-slot key
gets it.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

from backends import proxies  # noqa: E402


def env_with(**over):
    """A plausible host: the shared persona keys set, nothing per-slot."""
    base = {
        "CHAT_BACKEND_HOST": "127.0.0.1", "CHAT_BACKEND_PORT": "8010",
        "CHAT2_BACKEND_PORT": "8020",
        "THINK_TEMP": "0.7", "NOTHINK_TEMP": "0.7", "CODE_TEMP": "0.6",
        "THINK_REASONING_EFFORT": "high", "CODE_REASONING_EFFORT": "medium",
        "CODE_THINKING": "on", "LISTEN_HOST": "0.0.0.0",
        "MEMORY_GATEWAY_ENABLED": "on", "MEMORY_INJECTION_MODE": "system",
    }
    base.update(over)
    return base


class PersonaFallbackTests(unittest.TestCase):

    def test_a_host_that_only_set_the_shared_key_is_unchanged(self):
        """The compatibility half. Every existing host is this host."""
        for name in ("llm-a-proxy", "llm-b-proxy"):
            with self.subTest(name):
                out = proxies.resolve(proxies.PROXIES[name], env_with())
                self.assertEqual(out["THINK_TEMP"], "0.7")
                self.assertEqual(out["CODE_TEMP"], "0.6")

    def test_a_per_slot_key_wins_for_that_slot_alone(self):
        """The point of the change: the two slots serve different models and
        can finally be tuned differently."""
        env = env_with(LLM_B_THINK_TEMP="0.2")
        a = proxies.resolve(proxies.PROXIES["llm-a-proxy"], env)
        b = proxies.resolve(proxies.PROXIES["llm-b-proxy"], env)
        self.assertEqual(a["THINK_TEMP"], "0.7")
        self.assertEqual(b["THINK_TEMP"], "0.2")

    def test_an_empty_per_slot_key_is_not_an_override(self):
        """`LLM_B_THINK_TEMP=` in an env file means "unset", not "empty
        string" -- every other reader in this stack treats it that way, and a
        float("") would take the proxy down on startup."""
        out = proxies.resolve(proxies.PROXIES["llm-b-proxy"],
                              env_with(LLM_B_THINK_TEMP=""))
        self.assertEqual(out["THINK_TEMP"], "0.7")

    def test_the_two_keys_neither_script_exported_are_exported(self):
        """Read by the proxy, exported by neither launcher. systemd supplied
        them; running the script by hand did not."""
        out = proxies.resolve(proxies.PROXIES["llm-a-proxy"], env_with())
        for key in proxies.PREVIOUSLY_UNEXPORTED:
            with self.subTest(key):
                self.assertIn(key, out)


class ProxyIdentityTests(unittest.TestCase):

    def test_each_proxy_dials_its_own_backend(self):
        env = env_with()
        self.assertEqual(
            proxies.resolve(proxies.PROXIES["llm-a-proxy"], env)["CHAT_BACKEND_PORT"], "8010")
        self.assertEqual(
            proxies.resolve(proxies.PROXIES["llm-b-proxy"], env)["CHAT_BACKEND_PORT"], "8020")

    def test_the_ports_and_aliases_are_the_frozen_contract(self):
        """`docs/pi-forge-scheduling-contract.md` pins these and pi-forge and
        open-webui depend on them. Section 2.1: never ports, aliases or persona
        semantics."""
        a = proxies.resolve(proxies.PROXIES["llm-a-proxy"], {})
        self.assertEqual((a["THINK_PORT"], a["NOTHINK_PORT"], a["CODE_PORT"]),
                         ("8003", "8004", "8008"))
        self.assertEqual((a["THINK_MODEL_NAME"], a["NOTHINK_MODEL_NAME"],
                          a["CODE_MODEL_NAME"]), ("think", "chat", "code"))
        self.assertEqual(a["AGGREGATE_PORT"], "8012")
        b = proxies.resolve(proxies.PROXIES["llm-b-proxy"], {})
        self.assertEqual((b["THINK_PORT"], b["NOTHINK_PORT"], b["CODE_PORT"]),
                         ("8103", "8104", "8108"))
        self.assertEqual((b["THINK_MODEL_NAME"], b["NOTHINK_MODEL_NAME"],
                          b["CODE_MODEL_NAME"]), ("think2", "chat2", "code2"))
        self.assertEqual(b["AGGREGATE_PORT"], "8112")

    def test_only_one_proxy_runs_the_memory_gateway(self):
        """Not a hierarchy: the gateway binds a listener of its own, and two of
        them collide on the port."""
        running = [name for name, proxy in proxies.PROXIES.items() if proxy.memory_gateway]
        self.assertEqual(running, ["llm-a-proxy"])
        off = proxies.resolve(proxies.PROXIES["llm-b-proxy"], env_with())
        self.assertEqual(off["MEMORY_GATEWAY_ENABLED"], "off")
        self.assertNotIn("MEMORY_INJECTION_MODE", off)

    def test_the_gateway_settings_reach_the_proxy_that_runs_it(self):
        on = proxies.resolve(proxies.PROXIES["llm-a-proxy"], env_with())
        self.assertEqual(on["MEMORY_GATEWAY_ENABLED"], "on")
        self.assertEqual(on["MEMORY_INJECTION_MODE"], "system")


class RegistryShapeTests(unittest.TestCase):

    def test_every_proxy_fronts_a_slot_that_exists(self):
        from backends.slots import SLOTS
        for name, proxy in proxies.PROXIES.items():
            with self.subTest(name):
                self.assertIn(proxy.slot, SLOTS)

    def test_the_launcher_takes_a_proxy_and_nothing_else_decides(self):
        """One script. A second copy is how the first two drifted."""
        script = (ROOT / "scripts" / "start-proxy.sh").read_text()
        self.assertIn("proxy-env.py", script)
        self.assertFalse(list((ROOT / "scripts").glob("start-chat-proxy*.sh")),
                         "the hand-copied launchers are back")

    def test_every_shim_names_a_proxy_in_the_registry(self):
        for shim in (ROOT / "scripts").glob("start-llm-*-proxy.sh"):
            with self.subTest(shim.name):
                match = re.search(r"start-proxy\.sh\"?\s+(\S+)", shim.read_text())
                self.assertIsNotNone(match, f"{shim.name} does not call start-proxy.sh")
                self.assertIn(match.group(1), proxies.PROXIES)

    def test_the_persona_settings_are_stated_once(self):
        """`CODE_THINKING` is the one per-persona exception; the thinking
        persona is defined by having thinking on, so `THINK_THINKING` would be
        nonsense."""
        self.assertEqual(set(proxies.PERSONA_EXTRA_SUFFIXES), {"CODE"})
        self.assertIn("THINKING", proxies.persona_keys("CODE"))
        self.assertNotIn("THINKING", proxies.persona_keys("THINK"))


if __name__ == "__main__":
    unittest.main()
