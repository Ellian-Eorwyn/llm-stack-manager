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
import tempfile
import unittest


def _load(name: str, relative: str):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = pathlib.Path(__file__).resolve().parents[1]
deploy = _load("llm_stack_manager_deploy", "web/deploy.py")


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


if __name__ == "__main__":
    unittest.main()
