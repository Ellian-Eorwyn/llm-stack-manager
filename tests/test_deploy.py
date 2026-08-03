"""Deploy-drift detection, exercised against real git repositories.

The interesting cases here are all about git's behaviour rather than the
manager's, so these tests build actual checkouts in a temp directory and clone
between them. Mocking `git` would test the mock: the specific bug this module
exists to catch — a cached remote ref reporting "up to date" while the tree is
two commits behind — only reproduces against a real one.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def _load(name: str, relative: str):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = pathlib.Path(__file__).resolve().parents[1]
deploy = _load("llm_stack_manager_deploy", "web/deploy.py")

sys.path.insert(0, str(ROOT / "web"))
import app  # noqa: E402


def git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


class DeployDriftTests(unittest.TestCase):
    """A deployed clone that falls behind the tree it was cloned from."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name)

        self.origin = root / "origin"
        self.origin.mkdir()
        git(self.origin, "init", "--initial-branch=main")
        git(self.origin, "config", "user.email", "test@example.invalid")
        git(self.origin, "config", "user.name", "Test")
        (self.origin / "README.md").write_text("first\n")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-m", "first commit")

        self.deployed = root / "deployed"
        subprocess.run(["git", "clone", "--quiet", str(self.origin), str(self.deployed)],
                       check=True, capture_output=True)
        git(self.deployed, "config", "user.email", "test@example.invalid")
        git(self.deployed, "config", "user.name", "Test")

    def _commit_upstream(self, message: str, path: str = "README.md", body: str = ""):
        target = self.origin / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Distinct content per commit; git refuses an empty one.
        target.write_text(body or f"{message}\n")
        git(self.origin, "add", "-A")
        git(self.origin, "commit", "-m", message)

    def test_a_current_checkout_reports_current(self):
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["behind"], 0)
        self.assertEqual(snapshot["ahead"], 0)
        self.assertFalse(snapshot["dirty"])
        self.assertEqual(deploy.summarize(snapshot)["state"], "current")

    def test_the_fetch_is_what_makes_drift_visible(self):
        """The headline regression.

        Observed on this host: the deployed tree was two commits behind while
        its cached origin ref predated both, so the purely local comparison
        answered "0 behind". `local_state` still answers that way — it does not
        touch the network — which is exactly why `refresh_remote` has to.
        """
        self._commit_upstream("second commit")
        self._commit_upstream("third commit")

        stale = git(self.deployed, "rev-list", "--left-right", "--count", "HEAD...origin/main")
        self.assertEqual(stale.split(), ["0", "0"], "the cached ref should still look current")

        snapshot = deploy.refresh_remote(self.deployed)
        self.assertEqual(snapshot["behind"], 2)
        self.assertEqual(deploy.summarize(snapshot)["state"], "behind")
        self.assertIn("2 commits behind", deploy.summarize(snapshot)["message"])

    def test_pending_commits_are_named(self):
        self._commit_upstream("second commit")
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertEqual([c["subject"] for c in snapshot["pending_commits"]], ["second commit"])
        self.assertTrue(snapshot["pending_commits"][0]["sha"])

    def test_backend_launcher_changes_are_called_out(self):
        """A pending change to a launcher means only a model reload picks it up,
        which is the difference between a free update and an expensive one."""
        self._commit_upstream("touch a launcher", path="scripts/lib/common.sh", body="x\n")
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertEqual(snapshot["backend_sensitive_changes"], ["scripts/lib/common.sh"])
        self.assertIn("model backend launcher", deploy.summarize(snapshot)["message"])

    def test_an_ordinary_change_is_not_called_out_as_backend_sensitive(self):
        self._commit_upstream("touch the readme")
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertEqual(snapshot["backend_sensitive_changes"], [])

    def test_a_dirty_tree_is_reported_because_update_sh_refuses_to_run(self):
        (self.deployed / "README.md").write_text("locally edited\n")
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertTrue(snapshot["dirty"])
        self.assertEqual(snapshot["dirty_paths"], ["README.md"])
        self.assertEqual(deploy.summarize(snapshot)["state"], "dirty")

    def test_an_untracked_file_counts_as_dirty(self):
        """`update.sh` gates on `git status --porcelain`, which lists untracked
        files, so the report has to as well or it would promise an update that
        cannot run."""
        (self.deployed / "config").mkdir()
        (self.deployed / "config" / "service-expectations.json").write_text("{}")
        snapshot = deploy.local_state(self.deployed)
        self.assertTrue(snapshot["dirty"])
        self.assertEqual(snapshot["dirty_paths"], ["config/service-expectations.json"])

    def test_dirty_and_behind_reports_both(self):
        """Being behind is what the operator wants to fix; being dirty is why
        the fix will fail. Reporting only one sends them into a failure."""
        self._commit_upstream("second commit")
        (self.deployed / "README.md").write_text("locally edited\n")
        summary = deploy.summarize(deploy.refresh_remote(self.deployed))
        self.assertEqual(summary["state"], "dirty")
        self.assertIn("1 commit behind", summary["message"])
        self.assertIn("refuse to run", summary["message"])

    def test_a_local_commit_reads_as_ahead_not_behind(self):
        (self.deployed / "README.md").write_text("local work\n")
        git(self.deployed, "add", "-A")
        git(self.deployed, "commit", "-m", "local commit")
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertEqual(snapshot["ahead"], 1)
        self.assertEqual(snapshot["behind"], 0)

    def test_head_details_reach_the_report(self):
        snapshot = deploy.local_state(self.deployed)
        self.assertEqual(snapshot["subject"], "first commit")
        self.assertEqual(snapshot["branch"], "main")
        self.assertFalse(snapshot["detached"])
        self.assertEqual(len(snapshot["head"]), 40)
        self.assertTrue(snapshot["head"].startswith(snapshot["head_short"]))
        self.assertTrue(snapshot["committed_at"])

    def test_a_directory_that_is_not_a_checkout_says_so_instead_of_raising(self):
        plain = pathlib.Path(self._tmp.name) / "plain"
        plain.mkdir()
        snapshot = deploy.refresh_remote(plain)
        self.assertFalse(snapshot["ok"])
        self.assertFalse(snapshot["is_git_checkout"])
        self.assertEqual(deploy.summarize(snapshot)["state"], "unknown")

    def test_an_unreachable_remote_keeps_the_local_half_of_the_answer(self):
        """A box that cannot reach GitHub should report that it could not check,
        not lose the version it is running."""
        git(self.deployed, "remote", "set-url", "origin",
            str(pathlib.Path(self._tmp.name) / "does-not-exist"))
        snapshot = deploy.refresh_remote(self.deployed)
        self.assertTrue(snapshot["fetch_error"])
        self.assertEqual(snapshot["subject"], "first commit")


class DriftWatcherTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = pathlib.Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "--initial-branch=main")
        git(self.repo, "config", "user.email", "test@example.invalid")
        git(self.repo, "config", "user.name", "Test")
        (self.repo / "README.md").write_text("first\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", "first commit")

    def test_snapshot_before_any_check_still_reports_the_running_version(self):
        """The header shows a version from the first render, before the
        background thread has had a chance to run."""
        watcher = deploy.DriftWatcher(self.repo)
        snapshot = watcher.snapshot()
        self.assertEqual(snapshot["subject"], "first commit")
        self.assertIsNone(snapshot["remote_checked_at"])

    def test_snapshot_serves_the_cache_once_a_check_has_run(self):
        watcher = deploy.DriftWatcher(self.repo)
        watcher.check()
        self.assertIsNotNone(watcher.snapshot()["remote_checked_at"])

    def test_a_zero_interval_starts_no_thread(self):
        watcher = deploy.DriftWatcher(self.repo)
        watcher.start(lambda: {"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "0"})
        self.assertIsNone(watcher._thread)

    def test_check_interval_parsing(self):
        self.assertEqual(deploy.check_interval({}), deploy.DEFAULT_CHECK_INTERVAL_SECONDS)
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": ""}),
                         deploy.DEFAULT_CHECK_INTERVAL_SECONDS)
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "nonsense"}),
                         deploy.DEFAULT_CHECK_INTERVAL_SECONDS)
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "0"}), 0.0)
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "-5"}), 0.0)
        # Anything under a minute is pointless polling of a tree that changes
        # only when somebody runs update.sh.
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "5"}), 60.0)
        self.assertEqual(deploy.check_interval({"LLM_MANAGER_DEPLOY_CHECK_INTERVAL": "1800"}), 1800.0)


class BackendSensitivePathTests(unittest.TestCase):
    """One list, two readers.

    `update.sh` reports stale launchers after it updates; `deploy.py` reports
    them before. If the lists drift the two disagree about what an update costs,
    and the expensive case is the one that would stop being mentioned.
    """

    def test_update_sh_and_deploy_py_agree(self):
        update_sh = (ROOT / "update.sh").read_text()
        block = re.search(r"BACKEND_SENSITIVE_PATHS=\((.*?)\n\)", update_sh, re.S)
        self.assertIsNotNone(block, "update.sh lost its BACKEND_SENSITIVE_PATHS array")
        from_shell = re.findall(r'"([^"]+)"', block.group(1))
        self.assertEqual(sorted(from_shell), sorted(deploy.BACKEND_SENSITIVE_PATHS))

    def test_the_env_example_documents_the_interval(self):
        example = (ROOT / "config" / "llm-stack.env.example").read_text()
        self.assertIn("LLM_MANAGER_DEPLOY_CHECK_INTERVAL", example)


class LlamaCppPatchTests(unittest.TestCase):
    """A patched llama.cpp checkout must not disable its own updater.

    `deps/llama.cpp` is a pinned clone, not a fork, so every local change to it
    is re-applied from `patches/` on each install. That leaves the worktree
    permanently dirty — and the Update llama.cpp button refuses to run against a
    dirty checkout. Adding the first patch therefore disabled the button, which
    is exactly the kind of breakage that only shows up the day someone needs it.

    Both halves are load-bearing: the guard has to read a patched file as
    expected rather than as a local edit, and the update path has to re-apply
    the patches after the pull, because it builds with cmake directly instead of
    going through install-dependencies.py.
    """

    def test_the_declared_patches_all_exist(self):
        patches = app.llamacpp_patches()
        self.assertTrue(patches, "dependencies.json declares no llama.cpp patches")
        for patch in patches:
            self.assertTrue(patch.is_file(), f"declared but missing: {patch}")

    def test_patched_files_are_discovered_from_the_patches_themselves(self):
        """Not from a second hand-maintained list that could drift."""
        self.assertIn("tools/server/server-common.cpp", app.llamacpp_patched_paths())

    def test_a_patched_file_does_not_block_the_update(self):
        lines: list[str] = []
        restored: list[list[str]] = []

        def fake_run(cmd, timeout=None):
            restored.append(cmd)
            return 0, ""

        with mock.patch.object(app.core, "run_command", fake_run), \
             mock.patch.object(app, "has_uncommitted_git_changes", lambda _cmd: (False, "")):
            ok = app.try_restore_ignorable_llamacpp_update_changes(
                ["git"], " M tools/server/server-common.cpp", lines)
        self.assertTrue(ok)
        self.assertIn(["git", "restore", "--", "tools/server/server-common.cpp"], restored)

    def test_a_genuine_local_edit_still_blocks_the_update(self):
        """The guard is relaxed for patched files only, not switched off."""
        lines: list[str] = []
        self.assertFalse(app.try_restore_ignorable_llamacpp_update_changes(
            ["git"], " M tools/server/server-context.cpp", lines))

    def test_an_untracked_file_still_blocks_the_update(self):
        lines: list[str] = []
        self.assertFalse(app.try_restore_ignorable_llamacpp_update_changes(
            ["git"], "?? tools/server/server-common.cpp", lines))

    def test_the_patches_are_reapplied_after_the_pull(self):
        applied: list[list[str]] = []
        with mock.patch.object(app.core, "run_command",
                               lambda cmd, timeout=None: (applied.append(cmd), (0, ""))[1]):
            self.assertTrue(app.apply_llamacpp_patches(["git"], []))
        names = [c[-1] for c in applied if "apply" in c]
        self.assertEqual(len(names), len(app.llamacpp_patches()))

    def test_a_patch_that_no_longer_applies_fails_the_update_loudly(self):
        """Bumping the pin must surface here, not as a compile error later."""
        lines: list[str] = []
        with mock.patch.object(app.core, "run_command", lambda cmd, timeout=None: (1, "does not apply")):
            self.assertFalse(app.apply_llamacpp_patches(["git"], lines))
        self.assertTrue(any("Refresh the patch" in line for line in lines))

    def test_the_patch_applies_to_the_pinned_revision(self):
        """A stale patch is only discovered on the next rebuild otherwise."""
        source = ROOT / "deps" / "llama.cpp"
        if not (source / ".git").is_dir():
            self.skipTest("deps/llama.cpp is not checked out")
        for patch in app.llamacpp_patches():
            result = subprocess.run(
                ["git", "-c", f"safe.directory={source}", "-C", str(source),
                 "apply", "--check", "--reverse", str(patch)],
                capture_output=True, text=True)
            # --reverse succeeds when the patch is already applied, which is the
            # steady state on a deployed box; forward-check the clean case too.
            if result.returncode != 0:
                forward = subprocess.run(
                    ["git", "-c", f"safe.directory={source}", "-C", str(source),
                     "apply", "--check", str(patch)],
                    capture_output=True, text=True)
                self.assertEqual(forward.returncode, 0,
                                 f"{patch.name} applies neither forwards nor in reverse:\n"
                                 f"{forward.stderr}")


if __name__ == "__main__":
    unittest.main()
