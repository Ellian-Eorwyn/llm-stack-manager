"""validate.sh and the launchd half of scripts/cross-platform.sh.

Both are run for real, on either platform, against stub `uname`, `dscl`,
`launchctl` and `curl` on PATH -- the same idea as `platform_harness`, applied
to the shell side. On macOS the script used to crash under `set -u` before
checking anything, and once past that it asked `systemctl`, which a Mac does
not have, so every service would have been skipped as "not running".
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: What `launchctl list <label>` really prints: an OpenStep plist, not JSON.
LAUNCHCTL_LIST = textwrap.dedent('''\
    {
    \t"Label" = "com.llmstack.%(name)s";
    \t"LastExitStatus" = 0;
    \t"PID" = 4242;
    \t"ProgramArguments" = (
    \t\t"/stack/scripts/launchd-wrapper-%(name)s.sh";
    \t);
    };
''')


class _FakeMac:
    """A temp dir laid out as a macOS host with some launchd services running."""

    def __init__(self, running: tuple[str, ...], installed: tuple[str, ...]):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.home = self.dir / "home"
        agents = self.home / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        for name in installed:
            (agents / f"com.llmstack.{name}.plist").write_text("<plist/>")
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        self._stub("uname", 'echo Darwin')
        self._stub("dscl", f'echo "NFSHomeDirectory: {self.home}"')
        cases = "\n".join(
            f'    com.llmstack.{name}) cat <<"EOF"\n{LAUNCHCTL_LIST % {"name": name}}EOF\n    ;;'
            for name in running)
        self._stub("launchctl", textwrap.dedent('''\
            [ "$1" = list ] || exit 1
            case "$2" in
            %s
                *) echo "Could not find service \\"$2\\" in domain" >&2; exit 113 ;;
            esac
        ''') % cases)
        self._stub("curl", 'echo \'{"total_slots":4,"object":"list","content":"ok"}\'')
        # BSD `stat -f` means something else to GNU stat, so a Linux runner
        # needs this for the ownership lookups the installer makes.
        self._stub("stat", 'echo "$(id -un)"')

    def _stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def env(self) -> dict:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}{os.pathsep}{env['PATH']}"
        env.pop("LLM_LAUNCHD_DOMAIN", None)
        env.pop("SERVICE_USER", None)
        return env

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class SvcIsActiveOnDarwinTests(unittest.TestCase):
    def setUp(self):
        self.mac = _FakeMac(running=("llm-a",), installed=("llm-a", "llm-b"))
        self.addCleanup(self.mac.cleanup)

    def _active(self, name: str) -> bool:
        script = f'source "{ROOT}/scripts/cross-platform.sh"; svc_is_active {name}'
        return subprocess.run(["bash", "-uc", script], env=self.mac.env(),
                              capture_output=True).returncode == 0

    def test_a_running_launchd_service_is_active(self):
        # `launchctl list` output was parsed as JSON, which always failed, so
        # this read as inactive on every Mac.
        self.assertTrue(self._active("llm-a"))

    def test_an_installed_but_unloaded_service_is_not(self):
        self.assertFalse(self._active("llm-b"))

    def test_a_service_with_no_plist_is_not(self):
        self.assertFalse(self._active("transcript-backend"))


class ValidateOnDarwinTests(unittest.TestCase):
    """validate.sh finds its config and helpers next to itself, so it is copied
    into a scratch tree with a minimal config."""

    def setUp(self):
        self.mac = _FakeMac(running=("llm-manager", "llm-a", "llm-a-proxy"),
                            installed=("llm-manager", "llm-a", "llm-a-proxy"))
        self.addCleanup(self.mac.cleanup)
        self.tree = self.mac.dir / "stack"
        (self.tree / "scripts").mkdir(parents=True)
        (self.tree / "config").mkdir()
        shutil.copy(ROOT / "validate.sh", self.tree / "validate.sh")
        shutil.copy(ROOT / "scripts" / "cross-platform.sh",
                    self.tree / "scripts" / "cross-platform.sh")
        (self.tree / "config" / "llm-stack.env").write_text(textwrap.dedent('''\
            THINK_PORT=8003
            NOTHINK_PORT=8004
            CODE_PORT=8008
            EMBED_PORT=8005
            RERANK_PORT=8006
            TASK_PORT=8007
            CHAT_BACKEND_PORT=8010
            TRANSCRIPT_ENABLED=off
        '''))

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.tree / "validate.sh")],
                              env=self.mac.env(), capture_output=True, text=True,
                              timeout=60)

    def test_runs_to_the_end_under_set_u(self):
        result = self._run()
        self.assertNotIn("unbound variable", result.stderr)
        self.assertIn("Results:", result.stdout)

    def test_launchd_services_are_checked_not_skipped(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[PASS] GET :8010/props returns slot geometry", result.stdout)
        for port in (8003, 8004, 8008):
            self.assertIn(f"[PASS] GET :{port}/v1/models returns JSON", result.stdout)
        # Not installed and not running: skipped, not failed.
        self.assertIn("[SKIP] embed /v1/models", result.stdout)
        self.assertIn("[SKIP] Transcription sidecar", result.stdout)
        self.assertIn(" 0 failed", result.stdout)

    def test_a_down_backend_is_skipped_when_expected_off(self):
        (self.tree / "config" / "service-expectations.json").write_text(
            '{"llm-a": {"expected": "off"}}')
        result = self._run()
        self.assertIn("[SKIP] Primary backend", result.stdout)



class InstallLaunchdServiceTests(unittest.TestCase):
    """Start on a Mac installs a missing LaunchAgent through this script, so a
    component left out at setup can still be started from the manager."""

    def setUp(self):
        self.mac = _FakeMac(running=(), installed=())
        self.addCleanup(self.mac.cleanup)
        self.tree = self.mac.dir / "stack"
        (self.tree / "scripts").mkdir(parents=True)
        (self.tree / "config").mkdir()
        for name in ("cross-platform.sh", "install-launchd-service.sh"):
            shutil.copy(ROOT / "scripts" / name, self.tree / "scripts" / name)
        for name in ("start-task.sh", "start-embed.sh", "start-embed-mlx.sh"):
            (self.tree / "scripts" / name).write_text("#!/usr/bin/env bash\n")
        self._config("")

    def _config(self, extra: str) -> None:
        (self.tree / "config" / "llm-stack.env").write_text(
            f"STACK_DIR=/somewhere/else\n{extra}")

    def _install(self, name: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.tree / "scripts" / "install-launchd-service.sh"), name],
            env=self.mac.env(), capture_output=True, text=True, timeout=60)

    def _plist(self, name: str) -> dict:
        import plistlib
        path = self.mac.home / "Library" / "LaunchAgents" / f"com.llmstack.{name}.plist"
        return plistlib.loads(path.read_bytes())

    def test_writes_the_plist_install_sh_would(self):
        result = self._install("task")
        self.assertEqual(result.returncode, 0, result.stderr)
        plist = self._plist("task")
        self.assertEqual(plist["Label"], "com.llmstack.task")
        wrapper = pathlib.Path(plist["ProgramArguments"][0])
        # The wrapper runs this tree's launcher, not the one the config names.
        self.assertEqual(wrapper.parent, self.tree / "scripts")
        self.assertIn('exec "${STACK_DIR}/scripts/start-task.sh"', wrapper.read_text())
        self.assertEqual(plist["StandardErrorPath"], str(self.tree / "logs" / "task.stderr.log"))

    def test_follows_the_embedding_engine(self):
        self._config("EMBED_ENGINE=mlx\n")
        self.assertEqual(self._install("embed").returncode, 0)
        wrapper = pathlib.Path(self._plist("embed")["ProgramArguments"][0])
        self.assertIn("start-embed-mlx.sh", wrapper.read_text())

    def test_transcription_follows_its_engine(self):
        # Parakeet on MLX is the Mac's transcription server; the sidecar is
        # faster-whisper. The same choice install.sh makes.
        for engine, launcher in (("parakeet-mlx", "start-parakeet-mlx.sh"),
                                 ("sidecar", "start-transcribe.sh")):
            with self.subTest(engine):
                self._config(f"TRANSCRIPT_ENGINE={engine}\n")
                (self.tree / "scripts" / launcher).write_text("#!/usr/bin/env bash\n")
                self.assertEqual(self._install("transcript-backend").returncode, 0)
                wrapper = pathlib.Path(self._plist("transcript-backend")["ProgramArguments"][0])
                self.assertIn(launcher, wrapper.read_text())

    def test_refuses_what_needs_the_full_installer(self):
        for name in ("glmocr-sdk", "llama-router", "nonsense"):
            with self.subTest(name):
                result = self._install(name)
                self.assertEqual(result.returncode, 2)
                self.assertFalse((self.mac.home / "Library" / "LaunchAgents"
                                  / f"com.llmstack.{name}.plist").exists())

    def test_its_launchers_match_install_sh(self):
        """Two tables of the same fact drift; this holds them together."""
        import re
        installer = (ROOT / "install.sh").read_text()
        expected = dict(re.findall(
            r'install_mac_service\s+"([\w-]+)"\s+"[^"]*"\s+"(start-[\w.-]+\.sh)"', installer))
        ours = (ROOT / "scripts" / "install-launchd-service.sh").read_text()
        mapped = dict(re.findall(r'^\s+([\w-]+)\)\s+script="(start-[\w.-]+\.sh)"', ours, re.M))
        mapped.pop("mlx", None)   # the embedding engine's own case, below
        mapped["llama-router"] = "start-model-router.sh"
        self.assertTrue(mapped)
        for name, script in mapped.items():
            with self.subTest(name):
                self.assertEqual(expected.get(name), script)
        # The engine pairs, which install.sh resolves through a helper.
        self.assertIn('resolve_engine_script transcription "${TRANSCRIPT_ENGINE}" '
                      '"start-transcribe.sh" "start-parakeet-mlx.sh"', installer)
        self.assertIn('resolve_engine_script embed "${EMBED_ENGINE}" "start-embed.sh" "start-embed-mlx.sh"',
                      installer)


class UpdateOnDarwinTests(unittest.TestCase):
    """update.sh on a Mac, user domain, not root.

    Everything after the pull used to sit behind a root check, so an update
    pulled the new code and restarted nothing: the manager kept serving the old
    version. And where it did restart on a Mac, it restarted every service,
    running or not -- which on launchd starts the stopped ones.
    """

    def setUp(self):
        self.mac = _FakeMac(running=(), installed=("llm-manager", "llm-a-proxy", "embed"))
        self.addCleanup(self.mac.cleanup)
        self.calls = self.mac.dir / "launchctl.calls"
        running = ("llm-manager", "llm-a-proxy")
        cases = "\n".join(
            f'        com.llmstack.{name}) printf \'{{\\n\\t"PID" = 4242;\\n}};\\n\' ;;'
            for name in running)
        self.mac._stub("launchctl", textwrap.dedent('''\
            echo "$*" >> "%s"
            if [ "$1" = list ]; then
                case "$2" in
            %s
                    *) exit 113 ;;
                esac
            fi
        ''') % (self.calls, cases))

        origin = self.mac.dir / "origin.git"
        self.tree = self.mac.dir / "stack"
        (self.tree / "scripts").mkdir(parents=True)
        (self.tree / "config").mkdir()
        shutil.copy(ROOT / "update.sh", self.tree / "update.sh")
        for name in ("cross-platform.sh", "stack-services.sh", "install-launchd-service.sh"):
            shutil.copy(ROOT / "scripts" / name, self.tree / "scripts" / name)
        for name in ("start-llm-manager.sh", "start-llm-a-proxy.sh", "start-embed.sh"):
            (self.tree / "scripts" / name).write_text("#!/usr/bin/env bash\n")
        (self.tree / ".gitignore").write_text(
            "/config/llm-stack.env\n/logs/\n/scripts/launchd-wrapper-*.sh\n")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"]
        subprocess.run(git + ["init", "-q", "--bare", str(origin)], check=True)
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"],
                     ["remote", "add", "origin", str(origin)], ["push", "-q", "origin", "main"],
                     ["branch", "-q", "--set-upstream-to=origin/main"]):
            subprocess.run(git + args, cwd=self.tree, check=True, capture_output=True)
        (self.tree / "config" / "llm-stack.env").write_text("")

    def _update(self, *args) -> subprocess.CompletedProcess:
        env = self.mac.env()
        env["LLM_STACK_SKIP_DEP_UPDATE"] = "1"
        return subprocess.run(["bash", str(self.tree / "update.sh"), *args],
                              cwd=self.tree, env=env, capture_output=True, text=True, timeout=120)

    def _kickstarted(self) -> list[str]:
        lines = self.calls.read_text().splitlines() if self.calls.exists() else []
        return [line.split("/")[-1].replace("com.llmstack.", "")
                for line in lines if line.startswith("kickstart")]

    def test_restarts_what_is_running_and_the_manager_last(self):
        for args in ((), ("--manager-only",)):
            with self.subTest(args=args):
                self.calls.unlink(missing_ok=True)
                result = self._update(*args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                kicked = self._kickstarted()
                self.assertIn("llm-a-proxy", kicked)
                # The Update button runs this as the manager's child: anything
                # after the manager's restart might not run.
                self.assertEqual(kicked[-1], "llm-manager")
                # Stopped stays stopped; launchd would start it on a bootstrap.
                self.assertNotIn("embed", kicked)
                self.assertNotIn("bootout", self.calls.read_text())

    def test_refreshes_installed_agents_and_installs_nothing_new(self):
        self.assertEqual(self._update().returncode, 0)
        agents = self.mac.home / "Library" / "LaunchAgents"
        self.assertEqual(sorted(p.name for p in agents.iterdir()),
                         ["com.llmstack.embed.plist", "com.llmstack.llm-a-proxy.plist",
                          "com.llmstack.llm-manager.plist"])
        # Regenerated: the wrappers are written alongside each plist.
        for name in ("llm-manager", "llm-a-proxy", "embed"):
            with self.subTest(name):
                self.assertTrue((self.tree / "scripts" / f"launchd-wrapper-{name}.sh").exists())


class MetalKeepResidentTests(unittest.TestCase):
    """llama.cpp lets a model's Metal buffers go 3 minutes after its last
    request; macOS then compressed and swapped 21.5 GB of the idle 27B model."""

    def _keep_alive(self, platform: str, **env) -> str:
        script = (f'STACK_DIR="{ROOT}"; source "{ROOT}/scripts/lib/backend-preflight.sh"; '
                  'metal_keep_resident; echo "${GGML_METAL_RESIDENCY_KEEP_ALIVE_S:-unset}"')
        environ = {k: v for k, v in os.environ.items()
                   if k not in ("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "METAL_KEEP_MODELS_RESIDENT")}
        environ.update(LLM_STACK_PLATFORM=platform, **env)
        return subprocess.run(["bash", "-uc", script], env=environ,
                              capture_output=True, text=True).stdout.strip()

    def test_a_mac_keeps_models_wired_by_default(self):
        value = int(self._keep_alive("Darwin"))
        self.assertGreater(value, 30 * 24 * 3600)
        # llama.cpp counts it in 5 ms ticks in an int.
        self.assertLess(value * 200, 2**31)

    def test_it_can_be_turned_off_and_an_explicit_value_wins(self):
        self.assertEqual(self._keep_alive("Darwin", METAL_KEEP_MODELS_RESIDENT="off"), "unset")
        self.assertEqual(self._keep_alive("Darwin", GGML_METAL_RESIDENCY_KEEP_ALIVE_S="600"), "600")

    def test_linux_is_untouched(self):
        self.assertEqual(self._keep_alive("Linux"), "unset")

    def test_every_llama_cpp_launcher_applies_it(self):
        for name in ("start-backend.sh", "start-model-router.sh"):
            with self.subTest(name):
                self.assertIn("metal_keep_resident", (ROOT / "scripts" / name).read_text())


class LaunchdStopPersistsTests(unittest.TestCase):
    def setUp(self):
        self.mac = _FakeMac(running=(), installed=("rerank",))
        self.addCleanup(self.mac.cleanup)
        self.calls = self.mac.dir / "calls"
        self.loaded = self.mac.dir / "loaded"
        self.mac._stub("launchctl", textwrap.dedent(f'''\
            echo "$1" >> "{self.calls}"
            case "$1" in
                print) [ -f "{self.loaded}" ] ;;
                bootstrap) touch "{self.loaded}" ;;
                *) exit 0 ;;
            esac
        '''))

    def _run(self, fn: str) -> list[str]:
        self.calls.unlink(missing_ok=True)
        subprocess.run(["bash", "-uc", f'source "{ROOT}/scripts/cross-platform.sh"; {fn} rerank'],
                       env=self.mac.env(), check=True, capture_output=True)
        return self.calls.read_text().split()

    def test_stop_disables_and_start_enables(self):
        self.assertEqual(self._run("svc_stop"), ["bootout", "disable"])
        started = self._run("svc_start")
        self.assertLess(started.index("enable"), started.index("bootstrap"))

    def test_enable_does_not_bootstrap_a_loaded_job(self):
        # install.sh runs under set -e, and bootstrapping a loaded job fails.
        self.assertIn("bootstrap", self._run("svc_enable"))
        self.assertNotIn("bootstrap", self._run("svc_enable"))

if __name__ == "__main__":
    unittest.main()
