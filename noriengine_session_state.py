"""Session gating state: accumulated score, activity timeline and time-slot
scheduling.

Each session keeps one :class:`SessionGate`:

- ``pending_score``: weak-signal accumulated score; reaching the trigger
  threshold schedules a reply;
- ``timeline``: recent message timeline ``(timestamp, from_bot)``, used for
  presence suppression (bot reply share) and idle detection (relative to the
  average message interval).

Time-slot scheduling (:class:`TimeSlotRule`) overrides the base reply
probabilities by the current time and supports cross-midnight ranges
(start > end means [start, 24:00) ∪ [00:00, end)); a slot with probability 0
sleeps completely (strong signals are suppressed as well).
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

_TIME_PATTERN = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

_SLOT_RANGE_PATTERN = re.compile(
    r"^([01]?\d|2[0-3]):([0-5]\d)\s*-\s*([01]?\d|2[0-3]):([0-5]\d)$"
)


def _norm_hhmm(text: str) -> Optional[str]:
    """Validate and normalize HH:MM (zero-padded); returns None on invalid
    input.

    Zero padding is mandatory: slot coverage relies on string lexicographic
    order being equivalent to chronological order — mixing ``"8:00"`` and
    ``"08:00"`` would break the comparison ordering.
    """

    m = _TIME_PATTERN.match(text.strip())
    if not m:
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


# ---------------------------------------------------------------------------
# Time-slot scheduling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSlotRule:
    """One active-slot rule: overrides group/private base reply probabilities
    inside the time range."""

    name: str
    start: str  # HH:MM (inclusive)
    end: str  # HH:MM (exclusive); start > end means crossing midnight
    group_chance: float
    private_chance: float

    def covers(self, now_hhmm: str) -> bool:
        """Whether the HH:MM time string falls inside this slot."""

        if self.start <= self.end:
            return self.start <= now_hhmm < self.end
        # Crossing midnight: [start, 24:00) ∪ [00:00, end)
        return now_hhmm >= self.start or now_hhmm < self.end


def parse_time_slots(raw: object) -> tuple[list[TimeSlotRule], list[str]]:
    """Parse time-slot rules from the WebUI JSON list config.

    Returns ``(rules, reasons of skipped entries)``; one invalid entry is
    logged and skipped without affecting the rest. Empty input yields an
    empty list.

    Entry format::

        {"name": "day", "start": "08:00", "end": "23:00",
         "group": 0.5, "private": 1.0}
    """

    rules: list[TimeSlotRule] = []
    problems: list[str] = []
    if not raw:
        return rules, problems
    if not isinstance(raw, (list, tuple)):
        return rules, ["时段配置必须是 JSON 数组"]

    for idx, item in enumerate(raw):
        label = f"时段#{idx + 1}"
        if not isinstance(item, dict):
            problems.append(f"{label}: 不是对象，已跳过")
            continue

        start = _norm_hhmm(str(item.get("start", "")))
        end = _norm_hhmm(str(item.get("end", "")))
        if start is None or end is None:
            problems.append(f"{label}: start/end 需为 HH:MM 格式，已跳过")
            continue
        if start == end:
            problems.append(f"{label}: start 与 end 相同，已跳过")
            continue

        try:
            group_chance = _clamp_chance(item.get("group", 0.5))
            private_chance = _clamp_chance(item.get("private", 1.0))
        except (TypeError, ValueError):
            problems.append(f"{label}: 概率需为 0~1 数字，已跳过")
            continue

        name = str(item.get("name", "")).strip() or f"{start}-{end}"
        rules.append(
            TimeSlotRule(
                name=name,
                start=start,
                end=end,
                group_chance=group_chance,
                private_chance=private_chance,
            )
        )
    return rules, problems


def parse_time_slot_lists(
    ranges: object,
    group_chances: object,
    private_chances: object,
) -> tuple[list[TimeSlotRule], list[str]]:
    """Parse time-slot rules from the WebUI visual parallel lists (one rule
    per row, aligned by row number).

    Each ``ranges`` row is ``HH:MM-HH:MM`` (start > end means crossing
    midnight); row N of the two chance lists corresponds to slot N. Missing
    or invalid rows fall back to defaults (group 0.5 / private 1.0); extra
    rows are ignored.
    """
    rules: list[TimeSlotRule] = []
    problems: list[str] = []
    if not ranges:
        return rules, problems
    if not isinstance(ranges, (list, tuple)):
        return rules, ["时段范围需为每行一条的字符串列表"]

    def _chance_at(seq: object, idx: int, fallback: float, label: str) -> float:
        if not isinstance(seq, (list, tuple)) or idx >= len(seq):
            return fallback
        try:
            return _clamp_chance(seq[idx])
        except (TypeError, ValueError):
            problems.append(f"{label}: 概率 {seq[idx]!r} 非法，已回退 {fallback}")
            return fallback

    for idx, raw_range in enumerate(ranges):
        label = f"时段#{idx + 1}"
        m = _SLOT_RANGE_PATTERN.match(str(raw_range or "").strip())
        if not m:
            problems.append(f"{label}: {str(raw_range).strip()!r} 需为 HH:MM-HH:MM 格式，已跳过")
            continue
        start = f"{int(m.group(1)):02d}:{m.group(2)}"
        end = f"{int(m.group(3)):02d}:{m.group(4)}"
        if start == end:
            problems.append(f"{label}: 起止时间相同，已跳过")
            continue

        rules.append(
            TimeSlotRule(
                name=f"{start}-{end}",
                start=start,
                end=end,
                group_chance=_chance_at(group_chances, idx, 0.5, label),
                private_chance=_chance_at(private_chances, idx, 1.0, label),
            )
        )
    return rules, problems


def resolve_active_chances(
    base_group: float,
    base_private: float,
    slots: Sequence[TimeSlotRule],
    now_hhmm: Optional[str] = None,
) -> tuple[float, float]:
    """Resolve the active group/private reply probabilities by the current
    time (first matching slot wins)."""

    if not slots:
        return _clamp_chance(base_group), _clamp_chance(base_private)

    now = now_hhmm or datetime.now().strftime("%H:%M")
    for rule in slots:
        if rule.covers(now):
            return rule.group_chance, rule.private_chance
    return _clamp_chance(base_group), _clamp_chance(base_private)


# Backlog normalization floor: burst_count is at least 1 (the message being
# scored is itself counted); a threshold of 1 would keep ratio permanently
# ≥1, degenerating the pressure into a constant 30. A floor of 3 gives
# external messages within 5 seconds a three-step density gradient of
# 6/22/30 (3 messages count as a full batch)
_BACKLOG_NORM_FLOOR = 3


def backlog_norm_from_pace(pace: float) -> int:
    """Derive the backlog normalization threshold from the reply pace
    factor: the higher the pace, the lower the threshold (easier to trigger).

    The threshold is floored at :data:`_BACKLOG_NORM_FLOOR`: an isolated
    single message only gets 6 pressure points, so reaching the default
    threshold requires content signals combined with a real burst of ≥3
    messages within 5 seconds.
    """

    pace_sq = pace * pace if pace > 0 else 0.0
    if pace_sq <= 0:
        return _BACKLOG_NORM_FLOOR
    return max(_BACKLOG_NORM_FLOOR, math.ceil(1.0 / pace_sq))


def _clamp_chance(value: object) -> float:
    """Clamp a probability value to [0.0, 1.0]."""

    chance = float(value)
    if math.isnan(chance) or chance < 0.0:
        return 0.0
    return min(1.0, chance)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class TimelineEntry:
    """One timeline entry: when a message happened and whether it was the
    bot's own message.

    ``sender``: external sender user_id ("" for bot entries and records
    without sender info — such entries never count toward the topic-cluster
    distinct-sender total). ``low_value``: the scoring-time low-content
    judgment recorded with the entry (content-less entries default to
    low-value, mirroring ``is_low_content_batch([])`` semantics); the
    cluster condition requires at least one non-low-value entry.
    """

    ts: float
    from_bot: bool
    sender: str = ""
    low_value: bool = True


class SessionGate:
    """Gating state of a single session.

    ``poke_back_count``: consecutive mechanical poke-back counter (runtime
    state, not persisted). Once the cap is reached, pokes no longer get
    mechanical poke-backs — only score accumulation remains and the reply
    threshold takes over; any bot reply resets it (breaking mutual-poke
    loops between bots).
    """

    def __init__(self, timeline_capacity: int = 512) -> None:
        self.pending_score: float = 0.0
        self.poke_back_count: int = 0
        self.timeline: deque[TimelineEntry] = deque(maxlen=timeline_capacity)

    # -- timeline recording ---------------------------------------------
    def note_incoming(self, ts: float, sender: str = "", low_value: bool = True) -> None:
        """Record the arrival of one external message.

        ``sender``/``low_value`` feed topic-cluster detection; the defaults
        (anonymous, low-value) keep content-less records — e.g. pokes —
        from ever counting as cluster participants.
        """

        self.timeline.append(
            TimelineEntry(ts=ts, from_bot=False, sender=sender, low_value=low_value)
        )

    def note_bot_reply(self, ts: float) -> None:
        """Record one bot message (for presence statistics)."""

        self.timeline.append(TimelineEntry(ts=ts, from_bot=True))

    # -- statistics queries ----------------------------------------------
    def burst_count(self, now_ts: float, window_seconds: float = 5.0) -> int:
        """Count of external messages within the last ``window_seconds``
        (simulating the "batch" backlog).

        Only non-bot messages are counted: the bot's own messages are not
        pending backlog, and counting them would perversely raise the
        pressure score (the more active the bot, the easier to trigger —
        contrary to the intent of presence suppression). Returns at least 1
        (the message currently being scored).
        """

        if window_seconds <= 0 or not self.timeline:
            return 1
        return max(
            1,
            sum(
                1
                for e in self.timeline
                if not e.from_bot and 0 <= now_ts - e.ts <= window_seconds
            ),
        )

    def seconds_since_last_bot_reply(self, now_ts: float) -> Optional[float]:
        """Seconds since the bot's most recent message, or ``None`` when the
        retained timeline holds no bot entry (anchors the follow-up window)."""

        for entry in reversed(self.timeline):
            if entry.from_bot:
                return max(0.0, now_ts - entry.ts)
        return None

    def cluster_stats(self, now_ts: float, window_seconds: float) -> tuple[int, bool]:
        """``(distinct external senders, any non-low-value entry)`` within
        the last ``window_seconds`` — topic-cluster detection.

        The bot's own entries are excluded. Sender-less entries ("" —
        adapters without sender info, or content-less notices) never count
        toward the distinct-sender total: a single anonymous record must
        not impersonate any number of participants.
        """

        senders: set[str] = set()
        has_content = False
        for entry in self.timeline:
            if entry.from_bot or not (0 <= now_ts - entry.ts <= window_seconds):
                continue
            if entry.sender:
                senders.add(entry.sender)
            has_content = has_content or not entry.low_value
        return len(senders), has_content

    def window_stats(
        self,
        window_size: int,
        decay_minutes: float,
        now_ts: Optional[float] = None,
    ) -> tuple[int, int]:
        """``(bot replies, total messages)`` inside the recent window.

        With ``decay_minutes > 0`` only entries within the last N minutes
        are counted (time decay); with ``decay_minutes = 0`` a count-based
        window (the last ``window_size`` entries) is used instead.
        """

        if not self.timeline:
            return 0, 0

        if decay_minutes > 0:
            now = now_ts if now_ts is not None else self.timeline[-1].ts
            cutoff = now - decay_minutes * 60.0
            entries = [e for e in self.timeline if e.ts >= cutoff]
        else:
            entries = list(self.timeline)[-window_size:]

        if not entries:
            return 0, 0
        if decay_minutes > 0 and len(entries) > window_size:
            entries = entries[-window_size:]

        bot_count = sum(1 for e in entries if e.from_bot)
        return bot_count, len(entries)

    def idle_state(self, now_ts: float) -> tuple[float, bool]:
        """Return ``(seconds since the last external message, whether idle
        exceeds the average message interval)``.

        The average interval is the mean of consecutive external-message
        gaps; with fewer than two external messages "idle above average" is
        never reported.
        """

        external_ts = [e.ts for e in self.timeline if not e.from_bot]
        if not external_ts:
            return 0.0, False

        idle = max(0.0, now_ts - external_ts[-1])
        if len(external_ts) < 2:
            return idle, False

        intervals = [
            external_ts[i] - external_ts[i - 1] for i in range(1, len(external_ts))
        ]
        avg = sum(intervals) / len(intervals)
        above_avg = idle > avg if avg > 0 else False
        return idle, above_avg


class GateRegistry:
    """Registry of all session gating states (in-memory, reset on restart)."""

    def __init__(self, timeline_capacity: int = 512) -> None:
        self._gates: dict[str, SessionGate] = {}
        self._timeline_capacity = timeline_capacity

    def get(self, sid: str) -> SessionGate:
        """Get (or lazily create) the gating state of a session."""

        gate = self._gates.get(sid)
        if gate is None:
            gate = SessionGate(timeline_capacity=self._timeline_capacity)
            self._gates[sid] = gate
        return gate

    def drop(self, sid: str) -> None:
        """Drop a session's state (called when the session is invalidated /
        cleaned up)."""

        self._gates.pop(sid, None)

    def prune(self, keep_sids: set[str], max_idle_seconds: float = 7 * 24 * 3600.0) -> int:
        """Clean up long-inactive sessions not on the keep list; returns the
        number of removed sessions."""

        now = time.time()
        stale = []
        for sid, gate in self._gates.items():
            if sid in keep_sids:
                continue
            if not gate.timeline:
                stale.append(sid)
                continue
            if now - gate.timeline[-1].ts > max_idle_seconds:
                stale.append(sid)
        for sid in stale:
            self._gates.pop(sid, None)
        return len(stale)

    def all_sids(self) -> list[str]:
        """All session IDs currently registered."""

        return list(self._gates.keys())
