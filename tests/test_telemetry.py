from __future__ import annotations

import importlib.util
import pathlib
import unittest


def _load_telemetry_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "llm_stack_manager_telemetry", root / "web" / "telemetry.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


telemetry = _load_telemetry_module()

# Captured verbatim from `journalctl -u chat-backend-dense.service
# --output=short-iso-precise` on a live two-slot Qwen3.6-27B backend. The odd
# column spacing is llama.cpp's, and the parsers have to tolerate it.
LINES = {
    "select_by_id": "2026-07-28T15:46:53.867535-07:00 LLMs chat-backend-dense[1172175]: 5992.22.867.535 I slot get_availabl: id  1 | task -1 | selected slot by id (1)",
    "select_by_lru": "2026-07-28T15:14:24.815155-07:00 LLMs chat-backend-dense[1172175]: 5959.53.815.155 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 1820710861478",
    "select_by_lcp": "2026-07-28T15:14:43.880109-07:00 LLMs chat-backend-dense[1172175]: 5960.12.880.109 I slot get_availabl: id  1 | task -1 | selected slot by LCP similarity, sim_best = 1.000 (> 0.100 thold), f_keep = 0.434",
    "launch": "2026-07-28T15:46:56.251035-07:00 LLMs chat-backend-dense[1172175]: 5992.25.103.578 I slot launch_slot_: id  1 | task 424044 | processing task, is_child = 0",
    "release": "2026-07-26T20:52:21.693226-07:00 LLMs chat-backend-dense[1172175]: 3417.50.672.413 I slot      release: id  1 | task 200710 | stop processing: n_tokens = 9994, truncated = 0",
    "generation": "2026-07-28T15:14:47.122557-07:00 LLMs chat-backend-dense[1172175]: 5960.16.122.557 I slot print_timing: id  1 | task 423414 | n_decoded =    192, tg =  64.00 t/s, tg_3s =  64.00 t/s",
    "prompt_eval": "2026-07-26T20:52:21.692391-07:00 LLMs chat-backend-dense[1172175]: 3417.50.671.543 I slot print_timing: id  1 | task 200710 | prompt eval time =     264.39 ms /    30 tokens (    8.81 ms per token,   113.47 tokens per second)",
    "eval": "2026-07-26T20:52:21.692391-07:00 LLMs chat-backend-dense[1172175]: 3417.50.671.548 I slot print_timing: id  1 | task 200710 |        eval time =    2286.89 ms /   137 tokens (   16.69 ms per token,    59.91 tokens per second)",
    "total_time": "2026-07-26T20:52:21.692391-07:00 LLMs chat-backend-dense[1172175]: 3417.50.671.549 I slot print_timing: id  1 | task 200710 |       total time =    2551.29 ms /   167 tokens",
    "eval_sentinel": "2026-07-26T08:18:28.000000-07:00 LLMs chat-backend-dense[1172175]: 2663.57.182.462 I slot print_timing: id  1 | task 194198 |        eval time =       0.00 ms /     1 tokens (    0.00 ms per token, 1000000.00 tokens per second)",
    "draft": "2026-07-26T20:52:21.692391-07:00 LLMs chat-backend-dense[1172175]: 3417.50.671.553 I slot print_timing: id  1 | task 200710 | draft acceptance = 0.93137 (   95 accepted /   102 generated), mean len =  4.17",
    "evict": "2026-07-28T15:14:26.285353-07:00 LLMs chat-backend-dense[1172175]: 5959.55.285.353 W srv         alloc:  - making room for prompt cache entry, removing oldest entry (size = 4452.387 MiB)",
    "checkpoint": "2026-07-24T14:19:15.208296-07:00 LLMs chat-backend-dense[1172175]: 144.44.208.296 W slot create_check: id  0 | task 37491 | erasing old context checkpoint (pos_min = 183, pos_max = 183, n_tokens = 184, size = 150.012 MiB)",
    "overflow": "2026-07-28T15:14:28.028048-07:00 LLMs chat-backend-dense[1172175]: 5959.57.028.048 E srv    send_error: task id = 423411, error: request (155751 tokens) exceeds the available context size (131072 tokens), try increasing it",
    "ignored": "2026-07-28T15:14:48.293570-07:00 LLMs chat-backend-dense[1172175]: 5960.17.293.570 I slot print_timing: id  1 | task 423414 |    graphs reused =     154995",
    "not_journal": "some unrelated text without a journal prefix",
}


class JournalLineParsingTests(unittest.TestCase):
    def test_splits_journal_prefix_and_llama_log_prefix(self):
        ts, unit, body = telemetry.split_journal_line(LINES["evict"])
        self.assertEqual(unit, "chat-backend-dense")
        self.assertIsNotNone(ts)
        # Both the journal prefix and llama.cpp's "5959.55.285.353 W " stamp go.
        self.assertTrue(body.startswith("srv"), body)

    def test_non_journal_line_is_rejected(self):
        self.assertIsNone(telemetry.split_journal_line(LINES["not_journal"]))
        self.assertIsNone(telemetry.parse_line(LINES["not_journal"]))

    def test_uninteresting_line_parses_to_nothing(self):
        self.assertIsNone(telemetry.parse_line(LINES["ignored"]))

    def test_slot_selection_methods(self):
        by_id = telemetry.parse_line(LINES["select_by_id"])
        self.assertEqual((by_id["kind"], by_id["slot"], by_id["method"]), ("slot_select", 1, "id"))
        self.assertEqual(telemetry.parse_line(LINES["select_by_lru"])["method"], "lru")
        self.assertEqual(telemetry.parse_line(LINES["select_by_lcp"])["method"], "lcp")

    def test_slot_launch_and_release(self):
        launch = telemetry.parse_line(LINES["launch"])
        self.assertEqual((launch["kind"], launch["slot"], launch["task"]), ("slot_launch", 1, 424044))
        release = telemetry.parse_line(LINES["release"])
        self.assertEqual((release["kind"], release["n_tokens"]), ("slot_release", 9994))

    def test_generation_sample(self):
        event = telemetry.parse_line(LINES["generation"])
        self.assertEqual(event["kind"], "generation")
        self.assertEqual(event["n_decoded"], 192)
        self.assertAlmostEqual(event["tg_tps"], 64.0)

    def test_prompt_eval_and_eval_are_distinguished(self):
        prompt = telemetry.parse_line(LINES["prompt_eval"])
        self.assertEqual(prompt["kind"], "prompt_eval")
        self.assertAlmostEqual(prompt["tps"], 113.47)
        self.assertEqual(prompt["tokens"], 30)
        evaluated = telemetry.parse_line(LINES["eval"])
        self.assertEqual(evaluated["kind"], "eval")
        self.assertAlmostEqual(evaluated["tps"], 59.91)
        self.assertEqual(telemetry.parse_line(LINES["total_time"])["kind"], "total_time")

    def test_divide_by_zero_throughput_is_discarded(self):
        # llama.cpp prints 1000000.00 tok/s when elapsed time rounds to 0.00 ms.
        event = telemetry.parse_line(LINES["eval_sentinel"])
        self.assertEqual(event["kind"], "eval")
        self.assertIsNone(event["tps"])

    def test_draft_acceptance(self):
        event = telemetry.parse_line(LINES["draft"])
        self.assertEqual(event["kind"], "draft")
        self.assertAlmostEqual(event["rate"], 0.93137)
        self.assertEqual((event["accepted"], event["generated"]), (95, 102))
        self.assertAlmostEqual(event["mean_len"], 4.17)

    def test_prompt_cache_eviction_and_checkpoint(self):
        evicted = telemetry.parse_line(LINES["evict"])
        self.assertEqual(evicted["kind"], "cache_evict")
        self.assertAlmostEqual(evicted["mib"], 4452.387)
        checkpoint = telemetry.parse_line(LINES["checkpoint"])
        self.assertEqual(checkpoint["kind"], "checkpoint_erase")
        self.assertAlmostEqual(checkpoint["mib"], 150.012)
        self.assertEqual(checkpoint["n_tokens"], 184)

    def test_context_overflow(self):
        event = telemetry.parse_line(LINES["overflow"])
        self.assertEqual(event["kind"], "context_overflow")
        self.assertEqual((event["requested"], event["available"]), (155751, 131072))


class SummarizeTests(unittest.TestCase):
    @staticmethod
    def _events(pairs):
        return [dict(kind=kind, ts=ts, **fields) for ts, kind, fields in pairs]

    def test_select_to_launch_delay_is_paired_per_slot(self):
        # Two slots interleave; each delay must pair with its own slot.
        events = self._events([
            (100.0, "slot_select", {"slot": 0, "method": "lcp"}),
            (100.5, "slot_select", {"slot": 1, "method": "id"}),
            (102.0, "slot_launch", {"slot": 0, "task": 1}),
            (100.7, "slot_launch", {"slot": 1, "task": 2}),
        ])
        stats = telemetry.summarize(events, 3600, now=200.0)
        scheduling = stats["scheduling"]
        self.assertEqual(scheduling["select_to_launch_seconds"]["count"], 2)
        self.assertEqual(scheduling["select_to_launch_seconds"]["max"], 2.0)
        self.assertEqual(scheduling["over_1s"], 1)
        self.assertEqual(scheduling["select_methods"], {"lcp": 1, "id": 1})

    def test_launch_without_a_preceding_select_yields_no_delay(self):
        events = self._events([(100.0, "slot_launch", {"slot": 0, "task": 1})])
        stats = telemetry.summarize(events, 3600, now=200.0)
        self.assertEqual(stats["scheduling"]["select_to_launch_seconds"]["count"], 0)
        self.assertEqual(stats["cache"]["launches"], 1)

    def test_evictions_per_launch(self):
        events = self._events([
            (100.0, "slot_launch", {"slot": 0, "task": 1}),
            (100.1, "cache_evict", {"mib": 2000.0}),
            (100.2, "cache_evict", {"mib": 4000.0}),
            (101.0, "slot_launch", {"slot": 0, "task": 2}),
        ])
        cache = telemetry.summarize(events, 3600, now=200.0)["cache"]
        self.assertEqual(cache["launches"], 2)
        self.assertEqual(cache["evictions"], 2)
        self.assertEqual(cache["evictions_per_launch"], 1.0)
        self.assertEqual(cache["evicted_mib_total"], 6000.0)
        self.assertEqual(cache["evicted_mib"]["max"], 4000.0)

    def test_evictions_per_launch_is_none_without_launches(self):
        events = self._events([(100.0, "cache_evict", {"mib": 10.0})])
        self.assertIsNone(telemetry.summarize(events, 3600, now=200.0)["cache"]["evictions_per_launch"])

    def test_generation_and_live_samples_stay_separate(self):
        # `eval` is a whole-request average; `generation` lines are mid-flight
        # samples. Pooling them would double-count long requests.
        events = self._events([
            (100.0, "generation", {"n_decoded": 10, "tg_tps": 70.0}),
            (100.5, "generation", {"n_decoded": 20, "tg_tps": 50.0}),
            (101.0, "eval", {"ms": 1.0, "tokens": 20, "tps": 60.0}),
        ])
        throughput = telemetry.summarize(events, 3600, now=200.0)["throughput"]
        self.assertEqual(throughput["generation_tps"]["count"], 1)
        self.assertEqual(throughput["generation_tps"]["p50"], 60.0)
        self.assertEqual(throughput["live_tps"]["count"], 2)
        self.assertEqual(throughput["last_generation_tps"], 60.0)

    def test_events_outside_the_window_are_dropped(self):
        events = self._events([
            (100.0, "cache_evict", {"mib": 1.0}),
            (900.0, "cache_evict", {"mib": 2.0}),
        ])
        stats = telemetry.summarize(events, 300, now=1000.0)
        self.assertEqual(stats["cache"]["evictions"], 1)

    def test_context_overflow_is_reported(self):
        events = self._events([
            (100.0, "context_overflow", {"requested": 155751, "available": 131072, "message": "x"}),
        ])
        context = telemetry.summarize(events, 3600, now=200.0)["context"]
        self.assertEqual(context["overflow_count"], 1)
        self.assertEqual(context["overflows"][0]["requested"], 155751)

    def test_empty_input_produces_null_distributions(self):
        stats = telemetry.summarize([], 3600, now=200.0)
        self.assertEqual(stats["events"], 0)
        self.assertIsNone(stats["throughput"]["generation_tps"]["p50"])
        self.assertIsNone(stats["scheduling"]["over_1s_pct"])


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(telemetry.percentile(values, 0.0), 1.0)
        self.assertEqual(telemetry.percentile(values, 0.5), 3.0)
        self.assertEqual(telemetry.percentile(values, 1.0), 5.0)

    def test_empty(self):
        self.assertIsNone(telemetry.percentile([], 0.5))


class PrometheusTests(unittest.TestCase):
    def test_parses_values_and_skips_comments(self):
        text = (
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 1234\n"
            "llamacpp:kv_cache_usage_ratio 0.42\n"
            "\n"
            "malformed_line_without_value\n"
        )
        metrics = telemetry.parse_prometheus(text)
        self.assertEqual(metrics["llamacpp:prompt_tokens_total"], 1234.0)
        self.assertEqual(metrics["llamacpp:kv_cache_usage_ratio"], 0.42)
        self.assertNotIn("malformed_line_without_value", metrics)


class WindowTests(unittest.TestCase):
    def test_clamped_and_defaulted(self):
        self.assertEqual(telemetry.clamp_window(None), telemetry.DEFAULT_WINDOW_SECONDS)
        self.assertEqual(telemetry.clamp_window("not-a-number"), telemetry.DEFAULT_WINDOW_SECONDS)
        self.assertEqual(telemetry.clamp_window(1), telemetry.MIN_WINDOW_SECONDS)
        self.assertEqual(telemetry.clamp_window(10 ** 9), telemetry.MAX_WINDOW_SECONDS)
        self.assertEqual(telemetry.clamp_window("7200"), 7200)


class TargetResolutionTests(unittest.TestCase):
    def test_active_unit_selected_from_candidates(self):
        env = {"CHAT_BACKEND_PORT": "8010", "CHAT_BACKEND_HOST": "127.0.0.1"}
        targets = telemetry.resolve_targets(env, lambda unit: "active" if unit == "chat-backend-moe" else "inactive")
        primary = next(t for t in targets if t["name"] == "chat-primary")
        self.assertTrue(primary["active"])
        self.assertEqual(primary["unit"], "chat-backend-moe")
        self.assertEqual(primary["base_url"], "http://127.0.0.1:8010")

    def test_inactive_backend_has_no_unit(self):
        targets = telemetry.resolve_targets({}, lambda unit: "inactive")
        self.assertTrue(all(not t["active"] and t["unit"] is None for t in targets))

    def test_wildcard_bind_address_is_probed_over_loopback(self):
        env = {"EMBED_PORT": "8005", "EMBED_BACKEND_HOST": "0.0.0.0"}
        targets = telemetry.resolve_targets(env, lambda unit: "inactive")
        embed = next(t for t in targets if t["name"] == "embed")
        self.assertEqual(embed["base_url"], "http://127.0.0.1:8005")

    def test_ports_fall_back_to_defaults(self):
        targets = telemetry.resolve_targets({}, lambda unit: "inactive")
        embed = next(t for t in targets if t["name"] == "embed")
        self.assertEqual(embed["port"], "8005")


class HostMemoryTests(unittest.TestCase):
    def test_swap_is_reported(self):
        host = telemetry.host_memory({
            "MemTotal": 32 * 1024 * 1024,
            "MemAvailable": 8 * 1024 * 1024,
            "SwapTotal": 8 * 1024 * 1024,
            "SwapFree": 1 * 1024 * 1024,
        })
        self.assertEqual(host["mem_total_mib"], 32768)
        self.assertEqual(host["mem_used_mib"], 24576)
        self.assertEqual(host["mem_used_pct"], 75)
        self.assertEqual(host["swap_used_mib"], 7168)
        self.assertEqual(host["swap_used_pct"], 88)

    def test_missing_swap_does_not_divide_by_zero(self):
        host = telemetry.host_memory({"MemTotal": 1024, "MemAvailable": 512})
        self.assertIsNone(host["swap_used_pct"])


class SwapMonitorTest(unittest.TestCase):
    """Usage is history; only the rate says whether swapping is happening now."""

    def _monitor(self, *samples):
        """A monitor that reads the given (pswpin, pswpout) pairs in turn."""
        monitor = telemetry.SwapMonitor()
        queue = list(samples)
        monitor._read_counters = lambda: queue.pop(0) if queue else None  # noqa: SLF001
        return monitor

    def test_first_sample_claims_no_rate(self):
        monitor = self._monitor((1000, 2000))
        sample = monitor.sample(now=100.0)
        self.assertTrue(sample["available"])
        self.assertIsNone(sample["active"])

    def test_idle_swap_reports_inactive(self):
        monitor = self._monitor((1000, 2000), (1036, 2000))
        monitor.sample(now=100.0)
        sample = monitor.sample(now=105.0)
        self.assertFalse(sample["active"])
        self.assertAlmostEqual(sample["in_pages_per_second"], 7.2, places=1)
        self.assertEqual(sample["out_pages_per_second"], 0.0)

    def test_sustained_paging_reports_active(self):
        monitor = self._monitor((1000, 2000), (1000, 202000))
        monitor.sample(now=100.0)
        sample = monitor.sample(now=105.0)
        self.assertTrue(sample["active"])
        self.assertEqual(sample["out_pages_per_second"], 40000.0)

    def test_counter_reset_does_not_report_negative_rates(self):
        """A reboot or counter wrap must not read as paging in reverse."""
        monitor = self._monitor((5000, 9000), (10, 20))
        monitor.sample(now=100.0)
        sample = monitor.sample(now=105.0)
        self.assertEqual(sample["in_pages_per_second"], 0.0)
        self.assertEqual(sample["out_pages_per_second"], 0.0)

    def test_unreadable_vmstat_degrades(self):
        monitor = telemetry.SwapMonitor()
        monitor._read_counters = lambda: None  # noqa: SLF001
        sample = monitor.sample(now=100.0)
        self.assertFalse(sample["available"])
        self.assertIsNone(sample["active"])


class WarningTests(unittest.TestCase):
    @staticmethod
    def _backend(**stats):
        return {
            "name": "chat-primary", "label": "Primary Backend", "active": True,
            "metrics_available": True, "stats": stats or None,
        }

    def test_cache_thrash_is_flagged(self):
        backend = self._backend(
            cache={"evictions_per_launch": 0.54},
            scheduling={"select_to_launch_seconds": {}},
            context={},
        )
        texts = [w["text"] for w in telemetry.warnings_for([backend], {}, [])]
        self.assertTrue(any("evictions per slot launch" in t for t in texts), texts)

    def test_slot_delay_is_flagged(self):
        backend = self._backend(
            cache={},
            scheduling={"select_to_launch_seconds": {"p90": 2.4}},
            context={},
        )
        texts = [w["text"] for w in telemetry.warnings_for([backend], {}, [])]
        self.assertTrue(any("select-to-launch" in t for t in texts), texts)

    def test_low_vram_and_unknown_swap_state_are_flagged(self):
        """With no paging rate yet, heavy swap use still warns."""
        gpus = [{"index": 0, "mem_total": 24576, "mem_used": 24127}]
        warnings = telemetry.warnings_for([], {"swap_used_pct": 86, "swap_used_mib": 7080}, gpus)
        texts = [w["text"] for w in warnings]
        self.assertTrue(any("449 MiB free" in t for t in texts), texts)
        self.assertTrue(any("swap is 86%" in t for t in texts), texts)

    def test_active_swapping_warns(self):
        host = {"swap_used_pct": 86, "swap_used_mib": 7080,
                "swap_activity": {"active": True, "in_mib_per_second": 12.0,
                                  "out_mib_per_second": 30.5}}
        warnings = telemetry.warnings_for([], host, [])
        self.assertEqual([w["level"] for w in warnings], ["warn"])
        self.assertIn("actively swapping", warnings[0]["text"])

    def test_idle_swap_is_informational(self):
        """Cold pages from an earlier configuration are history, not pressure.

        This box sat at 83% swap with 19 GB of RAM free and nothing paging;
        warning about that trains operators to ignore the warning.
        """
        host = {"swap_used_pct": 83, "swap_used_mib": 6828,
                "swap_activity": {"active": False, "in_mib_per_second": 0.0,
                                  "out_mib_per_second": 0.0}}
        warnings = telemetry.warnings_for([], host, [])
        self.assertEqual([w["level"] for w in warnings], ["info"])
        self.assertIn("idle", warnings[0]["text"])

    def test_exhausted_host_ram_warns_regardless_of_swap(self):
        warnings = telemetry.warnings_for([], {"mem_available_mib": 900, "mem_available_pct": 3}, [])
        self.assertEqual([w["level"] for w in warnings], ["warn"])
        self.assertIn("900 MiB of host RAM is available", warnings[0]["text"])

    def test_missing_metrics_is_informational_only(self):
        backend = self._backend()
        backend["metrics_available"] = False
        warnings = telemetry.warnings_for([backend], {}, [])
        self.assertEqual([w["level"] for w in warnings], ["info"])

    def test_healthy_stack_produces_no_warnings(self):
        backend = self._backend(
            cache={"evictions_per_launch": 0.0},
            scheduling={"select_to_launch_seconds": {"p90": 0.01}},
            context={"overflow_count": 0},
        )
        gpus = [{"index": 0, "mem_total": 24576, "mem_used": 12000}]
        self.assertEqual(telemetry.warnings_for([backend], {"swap_used_pct": 2}, gpus), [])


if __name__ == "__main__":
    unittest.main()
