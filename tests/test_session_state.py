"""Unit tests for session gating state (time-slot scheduling / timeline /
registry)."""

import time

import pytest

from noriengine_session_state import (
    GateRegistry,
    SessionGate,
    TimeSlotRule,
    backlog_norm_from_pace,
    parse_time_slot_lists,
    parse_time_slots,
    resolve_active_chances,
)


class TestTimeSlotRule:
    def test_normal_range(self):
        rule = TimeSlotRule("白天", "08:00", "23:00", 0.5, 1.0)
        assert rule.covers("12:30") is True
        assert rule.covers("07:59") is False
        assert rule.covers("23:00") is False  # end is exclusive

    def test_overnight_range(self):
        rule = TimeSlotRule("深夜", "23:00", "08:00", 0.0, 0.0)
        assert rule.covers("23:00") is True
        assert rule.covers("02:30") is True
        assert rule.covers("07:59") is True
        assert rule.covers("08:00") is False
        assert rule.covers("12:00") is False


class TestParseTimeSlots:
    def test_valid_rules(self):
        rules, problems = parse_time_slots(
            [
                {"name": "白天", "start": "08:00", "end": "23:00", "group": 0.5, "private": 1.0},
                {"start": "23:00", "end": "08:00", "group": 0, "private": 0},
            ]
        )
        assert len(rules) == 2
        assert problems == []
        assert rules[1].name == "23:00-08:00"  # missing name falls back to the range

    def test_invalid_time_format_skipped(self):
        rules, problems = parse_time_slots([{"start": "8点", "end": "23:00"}])
        assert rules == []
        assert len(problems) == 1

    def test_same_start_end_skipped(self):
        rules, problems = parse_time_slots([{"start": "08:00", "end": "08:00"}])
        assert rules == []
        assert len(problems) == 1

    def test_non_dict_entry_skipped(self):
        rules, problems = parse_time_slots(["白天"])
        assert rules == []
        assert len(problems) == 1

    def test_chance_clamped(self):
        rules, _ = parse_time_slots([{"start": "08:00", "end": "23:00", "group": 1.7, "private": -0.2}])
        assert rules[0].group_chance == 1.0
        assert rules[0].private_chance == 0.0

    def test_empty_input(self):
        rules, problems = parse_time_slots([])
        assert rules == []
        assert problems == []
        rules, problems = parse_time_slots(None)
        assert rules == []
        assert problems == []

    def test_non_list_input(self):
        rules, problems = parse_time_slots({"start": "08:00"})
        assert rules == []
        assert len(problems) == 1

    def test_non_padded_time_normalized(self):
        # "8:00" without zero padding: only after normalization to "08:00"
        # does lexicographic order equal chronological order
        rules, problems = parse_time_slots(
            [{"start": "8:00", "end": "23:00", "group": 0.5, "private": 1.0}]
        )
        assert problems == []
        assert rules[0].start == "08:00"
        assert rules[0].covers("09:30") is True


class TestParseTimeSlotLists:
    def test_aligned_rows(self):
        rules, problems = parse_time_slot_lists(
            ["08:00-23:00", "23:00-08:00"],
            ["0.5", 0],
            ["1.0", 0.2],
        )
        assert problems == []
        assert len(rules) == 2
        assert (rules[0].group_chance, rules[0].private_chance) == (0.5, 1.0)
        assert (rules[1].group_chance, rules[1].private_chance) == (0.0, 0.2)
        assert rules[1].covers("02:00") is True  # crossing midnight

    def test_missing_chance_falls_back(self):
        rules, problems = parse_time_slot_lists(["08:00-23:00"], [], None)
        assert problems == []
        assert rules[0].group_chance == 0.5
        assert rules[0].private_chance == 1.0

    def test_invalid_chance_falls_back_with_problem(self):
        rules, problems = parse_time_slot_lists(["08:00-23:00"], ["abc"], [])
        assert rules[0].group_chance == 0.5
        assert len(problems) == 1

    def test_invalid_range_skipped(self):
        rules, problems = parse_time_slot_lists(["8点-23点", "08:00-23:00"], [], [])
        assert len(rules) == 1
        assert len(problems) == 1

    def test_same_start_end_skipped(self):
        rules, problems = parse_time_slot_lists(["08:00-08:00"], [], [])
        assert rules == []
        assert len(problems) == 1

    def test_non_padded_range_normalized(self):
        rules, _ = parse_time_slot_lists(["8:00 - 23:00"], [], [])
        assert rules[0].start == "08:00"
        assert rules[0].end == "23:00"

    def test_empty_input(self):
        rules, problems = parse_time_slot_lists([], [], [])
        assert rules == []
        assert problems == []

    def test_non_list_input(self):
        rules, problems = parse_time_slot_lists("08:00-23:00", [], [])
        assert rules == []
        assert len(problems) == 1


class TestResolveActiveChances:
    SLOTS = [
        TimeSlotRule("白天", "08:00", "23:00", 0.6, 1.0),
        TimeSlotRule("深夜", "23:00", "08:00", 0.0, 0.1),
    ]

    def test_no_slots_uses_base(self):
        group, private = resolve_active_chances(0.5, 1.0, [])
        assert (group, private) == (0.5, 1.0)

    def test_first_match_wins(self):
        slots = [TimeSlotRule("A", "00:00", "12:00", 0.3, 0.4), *self.SLOTS]
        group, private = resolve_active_chances(0.5, 1.0, slots, now_hhmm="09:00")
        assert (group, private) == (0.3, 0.4)

    def test_overnight_match(self):
        group, private = resolve_active_chances(0.5, 1.0, self.SLOTS, now_hhmm="23:30")
        assert (group, private) == (0.0, 0.1)

    def test_no_match_falls_back_to_base(self):
        slots = [TimeSlotRule("白天", "08:00", "09:00", 0.9, 1.0)]
        group, private = resolve_active_chances(0.5, 1.0, slots, now_hhmm="12:00")
        assert (group, private) == (0.5, 1.0)


class TestBacklogNorm:
    # Norm floor of 3 (v1.3.0): burst_count is at least 1 (the message being
    # scored is counted); a threshold of 1 would keep ratio permanently ≥1
    # and degenerate the pressure into a constant 30 — the floor of 3 keeps
    # a density gradient
    def test_floor_applies_at_high_pace(self):
        assert backlog_norm_from_pace(1.0) == 3
        assert backlog_norm_from_pace(2.0) == 3

    def test_pace_below_floor_uses_curve(self):
        # pace 0.5 → ceil(1/0.25)=4, above the floor: the curve applies as-is
        assert backlog_norm_from_pace(0.5) == 4

    def test_pace_zero_safe(self):
        assert backlog_norm_from_pace(0.0) == 3


class TestSessionGate:
    def test_burst_count_within_window(self):
        gate = SessionGate()
        gate.note_incoming(100.0)
        gate.note_incoming(101.0)
        gate.note_incoming(102.5)
        assert gate.burst_count(102.6, window_seconds=5.0) == 3
        assert gate.burst_count(105.9, window_seconds=5.0) == 2  # 101/102.5 within [100.9, 105.9]
        assert gate.burst_count(108.0, window_seconds=5.0) == 1
        assert gate.burst_count(200.0, window_seconds=5.0) == 1  # floor of 1 when all expired

    def test_burst_count_zero_window(self):
        gate = SessionGate()
        gate.note_incoming(100.0)
        gate.note_incoming(100.5)
        assert gate.burst_count(101.0, window_seconds=0) == 1

    def test_burst_count_ignores_bot_replies(self):
        # The bot's own messages are not pending backlog: excluded from the
        # burst count, floor still 1
        gate = SessionGate()
        gate.note_incoming(100.0)
        gate.note_bot_reply(100.5)
        gate.note_bot_reply(101.0)
        assert gate.burst_count(101.5, window_seconds=5.0) == 1
        gate.note_incoming(102.0)
        assert gate.burst_count(102.5, window_seconds=5.0) == 2

    def test_burst_count_empty_timeline(self):
        gate = SessionGate()
        assert gate.burst_count(100.0, window_seconds=5.0) == 1

    def test_note_and_window_stats_by_count(self):
        gate = SessionGate()
        base = time.time()
        for i in range(10):
            gate.note_incoming(base + i)
        gate.note_bot_reply(base + 100)
        bot, total = gate.window_stats(window_size=5, decay_minutes=0)
        # Count window takes the last 5 entries: 10 external + 1 bot →
        # last 5 = 4 external + 1 bot
        assert (bot, total) == (1, 5)

    def test_window_stats_time_decay(self):
        gate = SessionGate()
        base = time.time()
        gate.note_incoming(base - 3600)  # 1 hour ago
        gate.note_incoming(base - 60)  # 1 minute ago
        gate.note_bot_reply(base - 30)
        bot, total = gate.window_stats(window_size=20, decay_minutes=5)
        assert (bot, total) == (1, 2)  # 1-hour-old expired; 1-min-ago external + 30s-ago bot in window
        bot, total = gate.window_stats(window_size=20, decay_minutes=1)
        assert (bot, total) == (1, 2)  # both base-60 (external) / base-30 (bot) within 1 minute
        bot, total = gate.window_stats(window_size=20, decay_minutes=0.01)
        assert (bot, total) == (1, 1)  # only the last bot message within the 0.6s window

    def test_empty_timeline_stats(self):
        gate = SessionGate()
        assert gate.window_stats(20, 10) == (0, 0)

    def test_idle_state_no_external(self):
        gate = SessionGate()
        gate.note_bot_reply(time.time())
        idle, above = gate.idle_state(time.time() + 10)
        assert idle == 0.0
        assert above is False

    def test_idle_state_single_external(self):
        gate = SessionGate()
        gate.note_incoming(1000.0)
        idle, above = gate.idle_state(1070.0)
        assert idle == 70.0
        assert above is False

    def test_idle_state_above_average(self):
        gate = SessionGate()
        gate.note_incoming(1000.0)
        gate.note_incoming(1010.0)  # average interval 10s
        gate.note_incoming(1020.0)
        idle, above = gate.idle_state(1035.0)  # idle 15s > 10s
        assert idle == 15.0
        assert above is True

    def test_idle_state_below_average(self):
        gate = SessionGate()
        gate.note_incoming(1000.0)
        gate.note_incoming(1030.0)
        gate.note_incoming(1060.0)  # average interval 30s
        _, above = gate.idle_state(1070.0)  # idle 10s < 30s
        assert above is False

    def test_seconds_since_last_bot_reply(self):
        # No bot entry in the timeline → None (window cannot anchor)
        gate = SessionGate()
        gate.note_incoming(1000.0)
        assert gate.seconds_since_last_bot_reply(1005.0) is None
        # Anchors at the LATEST bot entry (scans from the right, external
        # entries in between are transparent)
        gate.note_bot_reply(1010.0)
        gate.note_incoming(1012.0)
        gate.note_bot_reply(1015.0)
        assert gate.seconds_since_last_bot_reply(1020.0) == 5.0
        # Clamped to 0 when queried at/before the anchor
        assert gate.seconds_since_last_bot_reply(1015.0) == 0.0

    def test_cluster_stats(self):
        # Trunk: distinct-sender counting + content flag + exclusions +
        # window expiry, over one seeded timeline
        gate = SessionGate()
        assert gate.cluster_stats(1000.0, window_seconds=30) == (0, False)
        gate.note_incoming(975.0, sender="u1", low_value=False)
        gate.note_incoming(980.0, sender="u1", low_value=False)  # same sender dedups
        gate.note_bot_reply(985.0)                               # bot entries excluded
        gate.note_incoming(990.0, sender="u2", low_value=True)  # low-value: no content credit
        gate.note_incoming(995.0)                                # anonymous (e.g. poke): never a sender
        users, has_content = gate.cluster_stats(1000.0, window_seconds=30)
        assert users == 2
        assert has_content is True
        # Branch: an all-low-value cluster never lights the content flag
        gate2 = SessionGate()
        gate2.note_incoming(980.0, sender="u1", low_value=True)
        gate2.note_incoming(985.0, sender="u2", low_value=True)
        assert gate2.cluster_stats(990.0, window_seconds=30) == (2, False)
        # Branch: the window slides past everything
        assert gate.cluster_stats(1030.0, window_seconds=30) == (0, False)


class TestGateRegistry:
    def test_get_creates_and_reuses(self):
        registry = GateRegistry()
        gate = registry.get("napcat:gm:1")
        assert registry.get("napcat:gm:1") is gate

    def test_drop(self):
        registry = GateRegistry()
        registry.get("napcat:gm:1")
        registry.drop("napcat:gm:1")
        gate2 = registry.get("napcat:gm:1")
        assert gate2 is not None
        assert gate2.pending_score == 0.0

    def test_prune_stale_sessions(self):
        registry = GateRegistry()
        old = registry.get("napcat:gm:old")
        old.note_incoming(time.time() - 8 * 24 * 3600)
        fresh = registry.get("napcat:gm:fresh")
        fresh.note_incoming(time.time())
        removed = registry.prune(keep_sids=set())
        assert removed == 1
        assert "napcat:gm:fresh" in registry.all_sids()
        assert "napcat:gm:old" not in registry.all_sids()

    def test_prune_keeps_listed(self):
        registry = GateRegistry()
        old = registry.get("napcat:gm:old")
        old.note_incoming(time.time() - 8 * 24 * 3600)
        removed = registry.prune(keep_sids={"napcat:gm:old"})
        assert removed == 0
        assert "napcat:gm:old" in registry.all_sids()
