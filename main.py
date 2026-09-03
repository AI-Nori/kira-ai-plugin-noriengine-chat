"""NoriEngine Chat — score-gated chat plugin.

Ports the message gating mechanism of nori-core quick_mode into the KiraAI
plugin system, fully replacing the built-in default-chat plugin (the two are
mutually exclusive; disable default-chat before enabling this plugin).

Core behavior:
- Every message enters the session buffer (unmentioned messages are kept as
  follow-up context);
- Private chat: whether to trigger a reply is decided directly by the active
  probability;
- Group chat: each message is scored on arrival — backlog pressure is capped
  globally at 30 and never scaled by time-slot probability; ordinary messages
  (no @/mention) only reach the threshold through the extreme combination of
  fully maxed content signals plus a topped-out backlog; strong signals
  (@bot etc.) trigger immediately once the threshold is reached; weak signals
  accumulate per message as "score × time-slot probability" and trigger when
  the accumulated score reaches the threshold;
- Time-slot scheduling: group/private probabilities can be overridden per
  time slot; slots with probability 0 sleep completely;
- Presence suppression: when the bot's recent reply share is too high, the
  trigger score is automatically reduced;
- Poke gating: poke events do not trigger the LLM directly, they accumulate
  a base score (unaffected by time-slot probability); below the threshold a
  mechanical poke-back may fire (global cooldown + per-session consecutive
  cap, only successful poke-backs are counted; any bot reply resets the cap
  counter);
- After a trigger, the whole buffer is flushed through a short merge window
  (debounce) and handed to the native KiraAI LLM pipeline;
- LLM round serialization (busy queue): the host dispatches batch events
  per session concurrently — two concurrent LLM rounds on the same session
  would race on stale memory and interleave replies. While a round is in
  flight (from flush publish until the session-memory-updated event), new
  triggers do not start a concurrent round: they are deferred while keeping
  the accumulated score, and flushed together through a merge window after
  the round ends; the buffer cap is enlarged during busy periods to reduce
  context loss while serialized; a watchdog timeout backstops a lost
  round-end signal.
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime

# The plugin manager loads main.py via spec_from_file_location, which does NOT
# add the plugin directory to sys.path; insert it explicitly so the sibling
# noriengine_* modules (plugin-prefixed top-level names, to avoid clashing with
# same-named modules of other plugins on sys.path) can be imported
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from core.chat.message_elements import At, Text
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.plugin import BasePlugin, Priority, logger, on
from core.provider import LLMRequest

from noriengine_session_state import (
    GateRegistry,
    backlog_norm_from_pace,
    parse_time_slot_lists,
    parse_time_slots,
    resolve_active_chances,
)
from noriengine_trigger_score import SignalWordlists, TriggerSnapshot, evaluate_trigger_score

_PRUNE_INTERVAL_SECONDS = 3600
_SESSION_STALE_SECONDS = 7 * 24 * 3600
# Idle self-exit deadline for the merge loop: a session-level debounce task
# with no new trigger for this long exits by itself and cleans up its
# registration (task/event), so the corresponding SessionGate becomes
# reclaimable by prune — otherwise flush structures would grow unboundedly
# with the number of historical sessions over long runtimes
_DEBOUNCE_IDLE_EXIT_SECONDS = 3600.0


class NoriEngineChatPlugin(BasePlugin):
    """Main body of the score-gated chat plugin."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.gates = GateRegistry()
        self.flush_events: dict[str, asyncio.Event] = {}
        self.flush_tasks: dict[str, asyncio.Task] = {}
        self.prune_task: asyncio.Task | None = None
        # Mechanical poke-back runtime state: global cooldown timestamp +
        # set of in-flight poke-back tasks
        self._last_poke_back_ts: float = 0.0
        self._poke_back_tasks: set[asyncio.Task] = set()
        # Busy queue (LLM round serialization) runtime state: sessions with a
        # round in flight / sessions with a deferred trigger / per-session
        # watchdog tasks / sessions armed for flush (buffer-cap enlargement
        # window)
        self.busy_sessions: set[str] = set()
        self.deferred_sids: set[str] = set()
        self.busy_watchdogs: dict[str, asyncio.Task] = {}
        self.flush_pending: set[str] = set()
        self._round_end_subscribed = False
        self._load_config()

    # ------------------------------------------------------------------
    # Config (hot-reload safe: initialize is re-entrant)
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        pc = self.plugin_cfg

        def _as_int(section: dict, key: str, fallback: int) -> int:
            try:
                return int(section.get(key, fallback))
            except (TypeError, ValueError):
                return fallback

        def _as_float(section: dict, key: str, fallback: float) -> float:
            try:
                return float(section.get(key, fallback))
            except (TypeError, ValueError):
                return fallback

        # -- Basic --
        basic = pc.get("section_basic", {}) or {}
        waking_words = basic.get("waking_words")
        if waking_words is not None and not isinstance(waking_words, (list, tuple)):
            logger.warning(
                f"[NoriEngineChat] waking_words 配置类型非法（{type(waking_words).__name__}），"
                f"需为字符串数组，已回退为空（唤醒词检测失效）"
            )
            waking_words = None
        self.waking_words: tuple[str, ...] = (
            tuple(str(w).strip() for w in waking_words if str(w).strip())
            if isinstance(waking_words, (list, tuple))
            else ()
        )
        self.max_context_messages = max(1, _as_int(basic, "max_context_messages", 5))
        self.group_chat_prompt = str(basic.get("group_chat_prompt", "") or "")

        # -- Trigger scoring --
        gate_cfg = pc.get("section_trigger", {}) or {}
        self.trigger_threshold = max(1, _as_int(gate_cfg, "trigger_threshold", 80))
        self.reply_pace = _as_float(gate_cfg, "reply_pace", 1.0)
        self.group_reply_chance = min(1.0, max(0.0, _as_float(gate_cfg, "group_reply_chance", 0.5)))
        self.private_reply_chance = min(1.0, max(0.0, _as_float(gate_cfg, "private_reply_chance", 1.0)))
        self.merge_wait = min(30.0, max(0.1, _as_float(gate_cfg, "merge_wait_seconds", 2.0)))
        self.burst_window = max(0.0, _as_float(gate_cfg, "burst_window_seconds", 5.0))
        self.backlog_norm = backlog_norm_from_pace(self.reply_pace)
        # Busy queue: triggers are deferred while an LLM round is processing,
        # and released together once the round ends
        self.busy_queue_max = max(1, _as_int(gate_cfg, "busy_queue_max", 10))
        self.busy_hold_timeout = min(
            300.0, max(5.0, _as_float(gate_cfg, "busy_hold_timeout_seconds", 60.0))
        )

        # -- Time-slot scheduling: prefer the visual parallel lists, keep the
        #    legacy JSON-array config as a fallback --
        slot_problems: list[str] = []
        slot_ranges = gate_cfg.get("slot_ranges")
        if slot_ranges:
            self.time_slots, slot_problems = parse_time_slot_lists(
                slot_ranges,
                gate_cfg.get("slot_group_chances"),
                gate_cfg.get("slot_private_chances"),
            )
        else:
            self.time_slots, slot_problems = parse_time_slots(gate_cfg.get("activity_slots"))
        for problem in slot_problems:
            logger.warning(f"[NoriEngineChat] {problem}")

        # -- Presence suppression --
        presence = pc.get("section_presence", {}) or {}
        self.presence_window_size = max(1, _as_int(presence, "presence_window_size", 20))
        self.presence_decay_minutes = max(0.0, _as_float(presence, "presence_decay_minutes", 10.0))

        # -- Poke --
        poke = pc.get("section_poke", {}) or {}
        self.poke_gate_enabled = bool(poke.get("poke_gate_enabled", True))
        self.poke_base_score = max(0, _as_int(poke, "poke_base_score", 30))
        self.poke_back_enabled = bool(poke.get("poke_back_enabled", False))
        self.poke_back_cooldown_seconds = max(0.0, _as_float(poke, "poke_back_cooldown_seconds", 3.0))
        self.poke_back_max = max(0, _as_int(poke, "poke_back_max", 3))

        # -- Signal wordlists --
        self.wordlists = SignalWordlists.from_config(pc.get("section_wordlists", {}) or {})

    async def initialize(self):
        # Mutual-exclusion guard: running alongside the built-in default-chat
        # double-buffers/double-flushes every message (both register
        # im_message at HIGH priority; the host does not deduplicate)
        get_inst = getattr(self.ctx, "get_plugin_inst", None)
        if callable(get_inst):
            try:
                chat_inst = get_inst("default-chat")
            except Exception:
                chat_inst = None
            if chat_inst is not None:
                logger.warning(
                    "[NoriEngineChat] 检测到内置默认聊天(default-chat)仍在运行，"
                    "两者互斥：请在 WebUI 停用默认聊天，否则消息将被双重处理"
                )
        # Config was loaded in __init__; only re-entrant runtime state is
        # handled here. Clean up merge-task handles that finished naturally
        # (hot-reload re-entry defense)
        for sid in [s for s, t in self.flush_tasks.items() if t.done()]:
            self.flush_tasks.pop(sid, None)
            self.flush_events.pop(sid, None)
        for sid in [s for s, t in self.busy_watchdogs.items() if t.done()]:
            self.busy_watchdogs.pop(sid, None)
        # Subscribe to the host session-memory-updated event (published by
        # update_memory, the last step of every batch round) as the precise
        # LLM round-end signal; same subscription style as session_media_manager
        bus = getattr(self.ctx, "event_bus", None)
        if bus is not None and not self._round_end_subscribed:
            bus.subscribe("session_memory_updated", self._on_session_memory_updated)
            self._round_end_subscribed = True
        if self.prune_task is None or self.prune_task.done():
            self.prune_task = asyncio.create_task(self._prune_loop())
        logger.info(
            "[NoriEngineChat] initialized: threshold=%d pace=%.2f group=%.2f private=%.2f "
            "merge_wait=%.1fs slots=%d wake_words=%d poke_gate=%s(+%d) "
            "poke_back=%s(cooldown=%.1fs max=%d) busy_queue=%d hold_timeout=%.0fs",
            self.trigger_threshold, self.reply_pace, self.group_reply_chance,
            self.private_reply_chance, self.merge_wait, len(self.time_slots),
            len(self.waking_words),
            self.poke_gate_enabled, self.poke_base_score,
            self.poke_back_enabled, self.poke_back_cooldown_seconds, self.poke_back_max,
            self.busy_queue_max, self.busy_hold_timeout,
        )

    async def terminate(self):
        # Unsubscribe the round-end event (hot-reload safe: bound methods
        # compare by instance, so other subscribers are unaffected)
        bus = getattr(self.ctx, "event_bus", None)
        if bus is not None and self._round_end_subscribed:
            try:
                bus.unsubscribe("session_memory_updated", self._on_session_memory_updated)
            except (KeyError, ValueError):
                pass
            self._round_end_subscribed = False
        # Cancel first, then gather everything, so no pending task is left
        # behind to trigger "Task was destroyed but it is pending" warnings
        # when the event loop shuts down
        tasks = (
            list(self.flush_tasks.values())
            + list(self._poke_back_tasks)
            + list(self.busy_watchdogs.values())
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.flush_tasks.clear()
        self.flush_events.clear()
        self.flush_pending.clear()
        self._poke_back_tasks.clear()
        self.busy_watchdogs.clear()
        self.busy_sessions.clear()
        self.deferred_sids.clear()
        if self.prune_task is not None:
            self.prune_task.cancel()
            await asyncio.gather(self.prune_task, return_exceptions=True)
            self.prune_task = None

    # ------------------------------------------------------------------
    # Message entry: waking-word detection + buffering + score gating
    # ------------------------------------------------------------------
    @on.im_message(priority=Priority.HIGH)
    async def handle_msg(self, event: KiraMessageEvent):
        # Waking-word detection: text containing any waking word is treated
        # as a mention (same semantics as default-chat)
        for element in event.message.chain:
            if isinstance(element, Text) and any(w in element.text for w in self.waking_words):
                event.message.is_mentioned = True
                break

        # Poke events: do not trigger the LLM directly, accumulate a base
        # score instead (unaffected by time-slot probability)
        poke_raw = self._poke_notice_raw(event)
        if poke_raw is not None:
            await self._handle_poke(event, poke_raw)
            return

        # Other empty-chain notices (e.g. poke echo aimed at a third party):
        # the host never buffers empty-chain messages, so skip scoring here
        # as well — third-party interactions must not inflate backlog or
        # accumulated score
        if event.is_notice and len(event.message.chain) == 0:
            return

        sid = event.session.sid
        is_group = event.is_group_message()
        now = time.time()

        # Buffer everything: unmentioned messages are kept as context too
        # (oldest trimmed beyond the cap); while busy / armed for flush a
        # larger busy-queue cap is used instead, reducing context loss
        # during round serialization
        buffer = self.ctx.get_buffer(str(event.session))
        cap = self._buffer_cap(sid)
        if buffer is not None and buffer.get_length() >= cap:
            buffer.pop(count=buffer.get_length() - cap + 1)
        event.buffer()

        # Activity timeline entry (presence / idle statistics). Idle is
        # sampled BEFORE recording the current message: idle measures the
        # quiet gap since the previous external message, and note_incoming
        # would make the current message the latest entry (idle always 0)
        gate = self.gates.get(sid)
        _, idle_above_avg = gate.idle_state(now)
        gate.note_incoming(now)

        # Already deferred while busy: subsequent messages only enter the
        # buffer and are flushed together after the round ends (no re-scoring)
        if sid in self.deferred_sids:
            return

        # Resolve the active probabilities from time-slot scheduling
        group_chance, private_chance = resolve_active_chances(
            self.group_reply_chance, self.private_reply_chance, self.time_slots,
            now_hhmm=self._now_hhmm(),
        )

        # ---- Private chat: probability gating ----
        if not is_group:
            if private_chance <= 0.0:
                logger.debug(f"[NoriEngineChat] 私聊时段休眠: sid={sid}")
                return
            if random.random() < private_chance:
                logger.info(f"[NoriEngineChat] 私聊触发回复: sid={sid} chance={private_chance}")
                self._trigger_or_defer(sid)
            return

        # ---- Group chat: score gating ----
        if group_chance <= 0.0:
            logger.debug(f"[NoriEngineChat] 群聊时段休眠: sid={sid}")
            return

        snapshot = self._build_snapshot(event, gate, now, idle_above_avg)
        verdict = evaluate_trigger_score(snapshot, self.wordlists)

        if verdict.score >= self.trigger_threshold:
            logger.info(f"[NoriEngineChat] 强信号触发: sid={sid} {verdict.breakdown}")
            self._trigger_or_defer(sid)
            return

        # Messages arriving inside an armed merge window are flushed with the
        # current round AND keep their accumulated credit — intentional
        # "topic heat inertia" (busy-deferred messages skip scoring instead)
        gate.pending_score += verdict.score * group_chance
        if gate.pending_score >= self.trigger_threshold:
            accumulated = gate.pending_score
            logger.info(
                f"[NoriEngineChat] 累积触发: sid={sid} accumulated={accumulated:.1f} "
                f"({verdict.breakdown})"
            )
            self._trigger_or_defer(sid)
        else:
            logger.debug(
                f"[NoriEngineChat] 静默累积: sid={sid} score={verdict.score} "
                f"acc={gate.pending_score:.1f}/{self.trigger_threshold} {verdict.breakdown}"
            )

    def _now_hhmm(self) -> str:
        """Current HH:MM: wall-clock time in the host locale.TZ timezone when
        configured, falling back to server-local time (same convention as the
        built-in plugins)."""

        get_tz = getattr(self.ctx, "get_timezone", None)
        tz = None
        if callable(get_tz):
            try:
                tz = get_tz()
            except Exception:
                tz = None
        if tz is not None:
            return datetime.now(tz).strftime("%H:%M")
        return datetime.now().strftime("%H:%M")

    def _build_snapshot(
        self,
        event: KiraMessageEvent,
        gate,
        now: float,
        idle_above_average: bool,
    ) -> TriggerSnapshot:
        """Build the scoring snapshot from the message event and session state.

        ``idle_above_average`` must be sampled by the caller BEFORE the
        current message is recorded into the timeline (see handle_msg);
        sampling after note_incoming would always see idle = 0.
        """

        has_at_bot = False
        texts: list[str] = []
        for element in event.message.chain:
            if isinstance(element, At):
                # An @ segment aimed at the bot is the strongest direct
                # signal; @all is not upgraded to an @ signal, but the host
                # sets is_mentioned=True for @all, so it reaches the
                # "mention" signal (80 points) and triggers a strong reply —
                # identical to default-chat behavior, this is intended
                if element.pid != "all" and element.pid == str(event.message.self_id):
                    has_at_bot = True
            elif isinstance(element, Text) and element.text:
                texts.append(element.text)

        bot_replies, window_total = gate.window_stats(
            self.presence_window_size, self.presence_decay_minutes, now
        )
        # Backlog is the recent burst count (external messages only, matching
        # the "batch" semantics), not the buffer length
        pending = gate.burst_count(now, self.burst_window)

        return TriggerSnapshot(
            texts=texts,
            has_at=has_at_bot,
            has_mention=bool(event.is_mentioned),
            is_group_chat=True,
            pending_count=pending,
            backlog_norm=self.backlog_norm,
            bot_recent_replies=bot_replies,
            recent_window_total=window_total,
            pace_factor=self.reply_pace,
            idle_above_average=idle_above_average,
        )

    # ------------------------------------------------------------------
    # Poke: score gating + mechanical poke-back
    # ------------------------------------------------------------------
    def _poke_notice_raw(self, event: KiraMessageEvent) -> dict | None:
        """Identify a poke event aimed at the AI, returning the raw notice data.

        Identification is structural (notify/poke + sender is not self +
        target is self), so it is not bound to any specific platform: any
        channel publishing a structurally identical notice event just works.
        Returns None when not matching, and the event then continues through
        the regular message flow. Self-initiated pokes are excluded to avoid
        a self-poke loop when some protocol ends echo outbound events back.
        """

        if not self.poke_gate_enabled or not event.is_notice:
            return None
        raw = event.raw_message
        if not isinstance(raw, dict):
            return None
        if raw.get("notice_type") != "notify" or raw.get("sub_type") != "poke":
            return None
        self_id = str(raw.get("self_id") or "")
        if str(raw.get("user_id") or "") == self_id:
            return None
        if str(raw.get("target_id") or "") != self_id:
            return None
        return raw

    async def _handle_poke(self, event: KiraMessageEvent, raw: dict) -> None:
        """Poke gating: the base score accumulates directly (not multiplied by
        time-slot probability); below the threshold a mechanical poke-back may
        fire."""

        sid = event.session.sid
        now = time.time()

        # Same as ordinary messages: buffer everything + record the activity
        # timeline (dormant slots still keep the context)
        buffer = self.ctx.get_buffer(str(event.session))
        cap = self._buffer_cap(sid)
        if buffer is not None and buffer.get_length() >= cap:
            buffer.pop(count=buffer.get_length() - cap + 1)
        event.buffer()
        gate = self.gates.get(sid)
        gate.note_incoming(now)

        # Already deferred while busy: subsequent pokes only enter the
        # buffer and are flushed together after the round ends
        if sid in self.deferred_sids:
            return

        # Dormant slot: no score, no poke-back (consistent with the dormant
        # suppression of strong signals)
        group_chance, private_chance = resolve_active_chances(
            self.group_reply_chance, self.private_reply_chance, self.time_slots,
            now_hhmm=self._now_hhmm(),
        )
        chance = group_chance if event.is_group_message() else private_chance
        if chance <= 0.0:
            logger.debug(f"[NoriEngineChat] 戳一戳休眠时段静默: sid={sid}")
            return

        gate.pending_score += self.poke_base_score
        if gate.pending_score >= self.trigger_threshold:
            accumulated = gate.pending_score
            logger.info(
                f"[NoriEngineChat] 戳一戳累积触发: sid={sid} accumulated={accumulated:.1f}"
            )
            self._trigger_or_defer(sid)
            return

        logger.debug(
            f"[NoriEngineChat] 戳一戳静默累积: sid={sid} +{self.poke_base_score} "
            f"acc={gate.pending_score:.1f}/{self.trigger_threshold}"
        )
        self._maybe_schedule_poke_back(event, raw, gate, now)

    def _maybe_schedule_poke_back(
        self, event: KiraMessageEvent, raw: dict, gate, now: float
    ) -> None:
        """Below the reply threshold, schedule one mechanical poke-back
        (constrained by the global cooldown and the per-session cap)."""

        if not self.poke_back_enabled:
            return
        # Global cooldown: multiple people / sessions poking at the same
        # time share a single throttle window
        if now - self._last_poke_back_ts < self.poke_back_cooldown_seconds:
            return
        # Per-session consecutive cap: once exhausted, only score accumulation
        # remains and the reply threshold takes over — this breaks infinite
        # mutual poking between bots; any bot reply resets the counter
        if self.poke_back_max > 0 and gate.poke_back_count >= self.poke_back_max:
            logger.info(
                f"[NoriEngineChat] 戳一戳回戳计数已满"
                f"({gate.poke_back_count}/{self.poke_back_max}): sid={event.session.sid}"
            )
            return
        self._last_poke_back_ts = now
        task = asyncio.create_task(self._do_poke_back(event, raw))
        self._poke_back_tasks.add(task)
        task.add_done_callback(self._poke_back_tasks.discard)

    async def _do_poke_back(self, event: KiraMessageEvent, raw: dict) -> None:
        """Poke the sender back directly (group/private parameters adapt);
        the counter only increments on a successful poke-back."""

        sid = event.session.sid
        try:
            adapter = self.ctx.adapter_mgr.get_adapter(event.adapter.name)
            client = adapter.get_client() if adapter else None
            if client is None or not hasattr(client, "send_poke"):
                logger.warning(
                    f"[NoriEngineChat] 机械回戳跳过: 平台不支持或客户端不可用 sid={sid}"
                )
                return
            target = str(raw.get("user_id"))
            group_id = raw.get("group_id")
            await client.send_poke(user_id=target, group_id=group_id)
            self.gates.get(sid).poke_back_count += 1
            logger.info(
                f"[NoriEngineChat] 机械回戳: sid={sid} target={target} "
                f"group={group_id or '-'}"
            )
        except Exception as e:
            # Failed poke-back: no count, no LLM fallback — the poke already
            # deposited score, later pokes accumulate naturally
            logger.warning(f"[NoriEngineChat] 机械回戳失败: sid={sid} {e}")

    # ------------------------------------------------------------------
    # Merge window (debounce) flush / busy queue (LLM round serialization)
    # ------------------------------------------------------------------
    def _buffer_cap(self, sid: str) -> int:
        """Current buffer cap for a session: enlarged to the busy-queue cap
        while busy or armed for flush.

        The enlargement window covers the "round end → merge window → next
        publish" gap, preventing messages accumulated during deferral from
        being evicted by the normal cap inside the next merge window.
        """
        if sid in self.busy_sessions or sid in self.flush_pending:
            return self.busy_queue_max
        return self.max_context_messages

    def _trigger_or_defer(self, sid: str) -> None:
        """Unified trigger entry point: defer while an LLM round is processing
        (accumulated score kept), otherwise arm the flush directly."""
        if sid not in self.busy_sessions:
            self._arm_flush(sid)
            return
        if sid not in self.deferred_sids:
            self.deferred_sids.add(sid)
            logger.info(
                f"[NoriEngineChat] LLM 回合处理中，触发挂起等待回合结束: sid={sid} "
                f"pending={self.gates.get(sid).pending_score:.1f}"
            )

    def _arm_flush(self, sid: str) -> None:
        """Arm one merge-window flush: merge subsequent burst messages within
        merge_wait."""

        # Clear the accumulated score at the unified trigger-decision point
        # (covers the private probability path and the poke accumulation path)
        self.gates.get(sid).pending_score = 0.0
        self.flush_pending.add(sid)
        if sid not in self.flush_events:
            self.flush_events[sid] = asyncio.Event()
        task = self.flush_tasks.get(sid)
        if task is None or task.done():
            self.flush_tasks[sid] = asyncio.create_task(self._debounce_loop(sid))
        self.flush_events[sid].set()

    async def _debounce_loop(self, sid: str) -> None:
        """Session-level merge loop: wait for trigger → merge window → flush.

        Self-exits and cleans up this session's task/event registration after
        ``_DEBOUNCE_IDLE_EXIT_SECONDS`` without a new trigger, preventing
        flush structures from growing unboundedly with historical sessions;
        once exited, the session's SessionGate is no longer protected as
        in-flight by ``_prune_loop`` and can be reclaimed by the 7-day
        inactivity rule.
        """

        evt = self.flush_events[sid]
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        evt.wait(), timeout=_DEBOUNCE_IDLE_EXIT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Scheduling race between the timeout decision and the
                    # set() in _arm_flush: re-check once, and treat a set
                    # event as a new trigger to keep serving
                    if evt.is_set():
                        continue
                    break
                evt.clear()
                await asyncio.sleep(self.merge_wait)
                if evt.is_set():
                    # New trigger within the merge window: extend by one
                    # window and keep merging
                    continue
                buffer = self.ctx.get_buffer(sid)
                if buffer is None or buffer.get_length() == 0:
                    continue
                # Busy starts at publish time: it covers the whole span of
                # publish → batch processing (VLM/STT preprocessing) → LLM →
                # update_memory, closing the gap earlier than ON_LLM_REQUEST
                self._mark_busy(sid)
                # Read the publish result from the MessageProcessor directly:
                # the PluginContext.flush_session_messages wrapper does not
                # forward the underlying bool, and the empty-flush detection
                # below depends on it (default-chat uses the same direct call)
                flushed = await self.ctx.message_processor.flush_session_messages(sid)
                self.flush_pending.discard(sid)
                if not flushed:
                    # Buffer was drained first by another consumer: no batch
                    # was published, so no round-end event will arrive —
                    # release busy immediately instead of waiting for the
                    # watchdog timeout
                    self._release_busy(sid, reason="empty_flush")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(f"[NoriEngineChat] 合并循环异常: sid={sid}")
        finally:
            # Backstop cleanup for the exception/cancellation path: when
            # flush raises, the discard above is skipped, and a leftover
            # entry would keep the session buffer cap permanently enlarged
            # to the busy-queue value
            self.flush_pending.discard(sid)
            # Only clean up when this task still owns the session's
            # registration slot, to avoid deleting a successor task that
            # _arm_flush has already rebuilt under extreme timing
            if self.flush_tasks.get(sid) is asyncio.current_task():
                self.flush_tasks.pop(sid, None)
                self.flush_events.pop(sid, None)

    def _mark_busy(self, sid: str) -> None:
        """Mark the session as inside an LLM round and (re)start the watchdog.

        Set at two points: flush-publish time (covering the multimodal
        preprocessing gap before batch processing starts) and ON_LLM_REQUEST
        (resetting the watchdog when the round actually starts; if the
        previous round-end signal was lost, the arrival of a new round is
        the self-healing evidence).
        """
        if sid not in self.busy_sessions:
            self.busy_sessions.add(sid)
            logger.debug(f"[NoriEngineChat] 会话进入 LLM 回合: sid={sid}")
        self._start_watchdog(sid)

    def _start_watchdog(self, sid: str) -> None:
        old = self.busy_watchdogs.get(sid)
        if old is not None and not old.done():
            old.cancel()
        self.busy_watchdogs[sid] = asyncio.create_task(self._busy_watchdog(sid))

    async def _busy_watchdog(self, sid: str) -> None:
        """Round-end signal backstop: force the round to be considered over
        on timeout, so a lost signal cannot hang the session."""
        try:
            await asyncio.sleep(self.busy_hold_timeout)
        except asyncio.CancelledError:
            return
        if sid in self.busy_sessions:
            logger.warning(
                f"[NoriEngineChat] LLM 回合看门狗超时({self.busy_hold_timeout:.0f}s)，"
                f"强制判定回合结束: sid={sid}"
            )
            self._release_busy(sid, reason="watchdog")

    def _release_busy(self, sid: str, reason: str) -> None:
        """Round end: clear busy/watchdog; release any deferred trigger
        through the normal merge window.

        Fully synchronous (no awaits), so on a single-threaded event loop it
        atomically interleaves with the busy check in handle_msg — a new
        message either sees busy (keeps deferring/buffering) or sees the
        fully released normal state; no intermediate state is observable.
        """
        self.busy_sessions.discard(sid)
        watchdog = self.busy_watchdogs.pop(sid, None)
        if watchdog is not None and not watchdog.done() and watchdog is not asyncio.current_task():
            watchdog.cancel()
        if sid in self.deferred_sids:
            self.deferred_sids.discard(sid)
            logger.info(f"[NoriEngineChat] LLM 回合结束({reason})，放行挂起触发: sid={sid}")
            self._arm_flush(sid)

    async def _on_session_memory_updated(self, event) -> None:
        """Host session_memory_updated event: the precise LLM round-end signal.

        ``update_memory`` is the last step of every host batch round
        (``handle_im_batch_message``) and the event payload carries the
        session name; filtered by sid, it releases busy and re-arms any
        deferred trigger. Paths that publish no event (e.g. save failures)
        are backstopped by the watchdog.
        """
        payload = getattr(event, "payload", None)
        sid = payload.get("session") if isinstance(payload, dict) else None
        if sid and sid in self.busy_sessions:
            self._release_busy(sid, reason="memory_updated")

    # ------------------------------------------------------------------
    # Presence tracking / group-chat prompt injection
    # ------------------------------------------------------------------
    @on.message_sent(priority=Priority.LOW)
    async def track_bot_reply(self, event: KiraMessageBatchEvent, action, result, *_, **__):
        """After a bot message is sent successfully, record it in the timeline
        (for presence suppression statistics) and reset the poke-back counter."""

        if result is not None and not getattr(result, "ok", True):
            return
        gate = self.gates.get(event.session.sid)
        gate.note_bot_reply(time.time())
        # Any AI reply recalibrates the mechanical poke-back budget (breaks
        # mutual-poke loops between bots)
        gate.poke_back_count = 0

    @on.llm_request(priority=Priority.MEDIUM)
    async def inject_group_prompt(self, event: KiraMessageBatchEvent, req: LLMRequest, *_, **__):
        """Inject an extra prompt for group-chat batches (same behavior as
        default-chat) and refresh the round's busy state."""

        # ON_LLM_REQUEST means the round has actually started: reset the
        # watchdog timer; the arrival of a new round necessarily means the
        # previous round has ended, which also self-heals a lost round-end
        # signal
        self._mark_busy(event.session.sid)
        if not event.is_group_message() or not self.group_chat_prompt:
            return
        for prompt in req.system_prompt:
            if prompt.name == "chat_env":
                prompt.content += self.group_chat_prompt
                break

    # ------------------------------------------------------------------
    # Background maintenance
    # ------------------------------------------------------------------
    async def _prune_loop(self) -> None:
        """Periodically clean up long-inactive session state."""

        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            try:
                active = {
                    sid for sid, task in self.flush_tasks.items() if not task.done()
                }
                removed = self.gates.prune(keep_sids=active, max_idle_seconds=_SESSION_STALE_SECONDS)
                if removed:
                    logger.info(f"[NoriEngineChat] 清理 {removed} 个不活跃会话状态")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[NoriEngineChat] 会话清理异常")
