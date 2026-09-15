"""Logic smoke tests for the plugin body (main.py): KiraAI runtime
dependencies are mocked.

Coverage: waking-word detection, @bot strong-signal pass-through, weak-signal
accumulated triggering, private-chat probability, dormant-slot suppression,
merge-window flush and presence timeline recording.
"""

import asyncio
import logging
import sys
import time
import types

import pytest


# ---------------------------------------------------------------------------
# Mocked KiraAI runtime modules
# ---------------------------------------------------------------------------
class _FakePriority:
    SYS_HIGH = 100
    HIGH = 50
    MEDIUM = 0
    LOW = -50
    SYS_LOW = -100


class _FakeOn:
    """Stub for the `@on.xxx(...)` decorators: returns the function as-is."""

    def __getattr__(self, name):
        def decorator_factory(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
        return decorator_factory


class _FakeBasePlugin:
    def __init__(self, ctx, cfg: dict):
        self.ctx = ctx
        self.plugin_cfg = cfg


class _FakeText:
    def __init__(self, text: str):
        self.text = text


class _FakeAt:
    def __init__(self, pid):
        self.pid = str(pid)


class _FakeImage:
    def __init__(self, url=""):
        self.url = url


class _FakeRecord:
    def __init__(self, url=""):
        self.url = url


class _FakeVideo:
    def __init__(self, url=""):
        self.url = url


class _FakeFile:
    def __init__(self, name=""):
        self.name = name


class _FakeForward:
    def __init__(self, count=1):
        self.count = count


class _FakeGroup:
    def __init__(self, group_id="1105211570"):
        self.group_id = group_id
        self.group_name = "测试群"


class _FakeMessage:
    def __init__(self, chain, self_id="3631688034", is_mentioned=False, group=None, sender=None):
        self.chain = chain
        self.self_id = self_id
        self.is_mentioned = is_mentioned
        self.group = group
        self.sender = sender or types.SimpleNamespace(user_id="242817150", nickname="测试用户")


class _FakeSession:
    def __init__(self, sid: str):
        self.sid = sid

    def __str__(self):
        return self.sid


class _FakeEvent:
    # Aligned with the real KiraMessageEvent attributes (read by the poke
    # branch); default is a non-notice event
    is_notice = False
    raw_message = None
    adapter = None

    def __init__(self, message: _FakeMessage, sid: str, ctx: "_FakeCtx"):
        self.message = message
        self.session = _FakeSession(sid)
        self._ctx = ctx
        self._buffered = False

    def is_group_message(self):
        return self.message.group is not None

    @property
    def is_mentioned(self):
        return self.message.is_mentioned

    def buffer(self, force=False):
        """Mimic the framework behavior: the message enters the session buffer."""

        self._buffered = True
        self._ctx.get_buffer(str(self.session)).items.append(self)


class _FakeBuffer:
    def __init__(self):
        self.items = []

    def get_length(self):
        return len(self.items)

    def pop(self, count=1):
        del self.items[:count]


class _FakeAdapterManager:
    """Poke-back test stub: returns an adapter object with an injectable client."""

    def __init__(self, client=None):
        self._client = client

    def get_adapter(self, name):
        return types.SimpleNamespace(get_client=lambda: self._client)


class _FakePokeClient:
    """Fake protocol client that records send_poke calls."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    async def send_poke(self, user_id, group_id=None):
        self.calls.append((str(user_id), group_id))


class _FakeMessageProcessor:
    """Mirror of the real MessageProcessor.flush_session_messages: drain the
    session buffer, publish a batch, return whether one was published."""

    def __init__(self, ctx: "_FakeCtx"):
        self._ctx = ctx

    async def flush_session_messages(self, sid: str, extra_event=None) -> bool:
        buffer = self._ctx.buffers.get(sid)
        if buffer is None or buffer.get_length() == 0:
            return False
        buffer.items.clear()
        self._ctx.flushed.append(sid)
        return True


class _FakeCtx:
    """Mirror of the real PluginContext surface the plugin touches.

    Deliberately exposes NO ``flush_session_messages`` method: the real
    PluginContext wrapper swallows the processor's bool return value, so the
    plugin must read the publish result from ``message_processor`` directly —
    if the plugin ever regresses to the wrapper, these tests fail loudly.
    """

    def __init__(self, poke_client: "_FakePokeClient | None" = None):
        self.buffers: dict[str, _FakeBuffer] = {}
        self.flushed: list[str] = []
        self.adapter_mgr = _FakeAdapterManager(poke_client)
        self.message_processor = _FakeMessageProcessor(self)

    def get_buffer(self, sid: str) -> _FakeBuffer:
        if sid not in self.buffers:
            self.buffers[sid] = _FakeBuffer()
        return self.buffers[sid]


def _install_fake_core():
    """Inject the mocked core.* modules into sys.modules (idempotent)."""

    if "core.plugin" in sys.modules:
        return

    core = types.ModuleType("core")
    core_plugin = types.ModuleType("core.plugin")
    core_plugin.Priority = _FakePriority
    core_plugin.on = _FakeOn()
    core_plugin.BasePlugin = _FakeBasePlugin
    core_plugin.logger = logging.getLogger("test.kira.plugin")

    core_chat = types.ModuleType("core.chat")
    core_elements = types.ModuleType("core.chat.message_elements")
    core_elements.At = _FakeAt
    core_elements.Text = _FakeText
    core_elements.Image = _FakeImage
    core_elements.Record = _FakeRecord
    core_elements.Video = _FakeVideo
    core_elements.File = _FakeFile
    core_elements.Forward = _FakeForward
    core_utils = types.ModuleType("core.chat.message_utils")
    core_utils.KiraMessageEvent = type("KiraMessageEvent", (), {})
    core_utils.KiraMessageBatchEvent = type("KiraMessageBatchEvent", (), {})

    core_provider = types.ModuleType("core.provider")
    core_provider.LLMRequest = type("LLMRequest", (), {})

    sys.modules["core"] = core
    sys.modules["core.plugin"] = core_plugin
    sys.modules["core.chat"] = core_chat
    sys.modules["core.chat.message_elements"] = core_elements
    sys.modules["core.chat.message_utils"] = core_utils
    sys.modules["core.provider"] = core_provider


_install_fake_core()

import main as plugin_main  # noqa: E402
from noriengine_trigger_score import evaluate_trigger_score  # noqa: E402

GROUP_SID = "napcat:gm:1105211570"
DM_SID = "napcat:dm:242817150"
SELF_ID = "3631688034"
POKER_ID = "242817150"
GROUP_ID_NUM = 1105211570


def _make_plugin(cfg: dict | None = None):
    ctx = _FakeCtx()
    plugin = plugin_main.NoriEngineChatPlugin(ctx, cfg or {})
    return plugin, ctx


def _group_event(ctx, *chain, is_mentioned=False):
    return _FakeEvent(
        _FakeMessage(list(chain), is_mentioned=is_mentioned, group=_FakeGroup()),
        GROUP_SID,
        ctx,
    )


def _dm_event(ctx, *chain):
    return _FakeEvent(_FakeMessage(list(chain), group=None), DM_SID, ctx)


def _poke_event(
    ctx,
    group=True,
    poker=POKER_ID,
    target=SELF_ID,
    self_id=SELF_ID,
):
    """Build a poke event aimed at the AI (raw is the OneBot notice structure)."""

    target_hit = str(target) == str(self_id)
    msg = _FakeMessage(
        [_FakeText(f"[Poke 用户{poker}(测试用户)戳了戳你]")],
        self_id=self_id,
        is_mentioned=target_hit,
        group=_FakeGroup() if group else None,
    )
    ev = _FakeEvent(msg, GROUP_SID if group else DM_SID, ctx)
    ev.is_notice = True
    ev.adapter = types.SimpleNamespace(name="napcat", platform="QQ")
    ev.raw_message = {
        "post_type": "notice",
        "notice_type": "notify",
        "sub_type": "poke",
        "user_id": int(poker),
        "target_id": int(target),
        "self_id": int(self_id),
        "group_id": GROUP_ID_NUM if group else None,
    }
    return ev


async def _drain(plugin, ctx, wait: float):
    """Wait for the merge window to expire and flush tasks to finish."""

    await asyncio.sleep(wait + 0.3)


class TestPluginBasics:
    def test_default_config_loaded(self):
        plugin, _ = _make_plugin()
        assert plugin.trigger_threshold == 80
        assert plugin.group_reply_chance == 0.5
        assert plugin.private_reply_chance == 1.0
        assert plugin.merge_wait == 2.0
        assert plugin.backlog_norm == 3  # v1.3.0 norm floor of 3
        assert plugin.time_slots == []
        assert plugin.waking_words == ()

    def test_waking_word_marks_mention(self):
        plugin, ctx = _make_plugin({"section_basic": {"waking_words": ["汐祈"]}})
        event = _group_event(ctx, _FakeText("汐祈在吗"))
        asyncio.run(plugin.handle_msg(event))
        assert event.message.is_mentioned is True

    def test_section_config_override(self):
        plugin, _ = _make_plugin(
            {
                "section_basic": {"max_context_messages": 3},
                "section_trigger": {"trigger_threshold": 120, "group_reply_chance": 0.2},
                "section_presence": {"presence_window_size": 8},
            }
        )
        assert plugin.max_context_messages == 3
        assert plugin.trigger_threshold == 120
        assert plugin.group_reply_chance == 0.2
        assert plugin.presence_window_size == 8


class TestGroupGating:
    def test_at_bot_strong_signal_flushes(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})
        event = _group_event(ctx, _FakeAt("3631688034"), _FakeText("在忙吗"))

        async def scenario():
            await plugin.handle_msg(event)
            assert event._buffered is True
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert GROUP_SID in ctx.flushed

    def test_plain_low_signal_stays_silent(self):
        # Low-value messages arriving spread out: burst window set to 0 so
        # each message is scored independently with a single-message backlog
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            for _ in range(3):
                await plugin.handle_msg(_group_event(ctx, _FakeText("哈哈")))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []
        # Single "哈哈": backlog 6 (single message under norm 3) - low-value 25
        # = -19 → verdict 0 (never negative), accumulates 0 < 80
        gate = plugin.gates.get(GROUP_SID)
        assert gate.pending_score == pytest.approx(0.0)

    def test_voice_message_earns_media_points(self):
        # Voice carries no analyzable text at scoring time (the host only
        # runs STT at LLM-round prompt building): previously a Record-only
        # message was an empty batch → low-value -25 → verdict 0; now fixed
        # media points — single message: backlog 6 + media 15 = 21, ×0.5
        # accumulates 10.5
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeRecord()))
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(10.5)

    def test_image_exempts_low_value_penalty(self):
        # "哈哈" + image: the image is real content, so the low-value batch
        # penalty no longer applies — media 15 + backlog 6 = 21 (previously
        # 6 - 25 → clamped 0), ×0.5 accumulates 10.5
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeText("哈哈"), _FakeImage()))
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(10.5)

    def test_video_counts_as_major_media(self):
        # Video is the same major tier as voice/image: backlog 6 + media 15
        # = 21, ×0.5 accumulates 10.5
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeVideo()))
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(10.5)

    def test_file_earns_weak_media_points(self):
        # File/forward are the minor media tier: backlog 6 + weak media 10
        # = 16, ×0.5 accumulates 8.0 (previously empty batch → -25 → 0)
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeFile("报告.pdf")))
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(8.0)

    def test_forward_earns_weak_media_points(self):
        plugin, ctx = _make_plugin(
            {"section_trigger": {"merge_wait_seconds": 0.2, "burst_window_seconds": 0}}
        )

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeForward(count=3)))
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(8.0)

    def test_low_content_burst_stays_silent_under_cap(self):
        # Burst flooding can no longer borrow backlog points: a 1..4 message
        # burst gets backlog 6/22/30/30 (capped), low-value -25 → verdicts
        # 0/0/5/5, ×0.5 chance accumulates only 5 points — far below the
        # threshold
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})

        async def scenario():
            # The first inter-message gap is much larger than the rest, so
            # idle_above_average stays decisively False now that the idle
            # bonus is actually reachable (sampled against the previous
            # external message); this pins the pure burst gradient 1/2/3/4
            for i in range(4):
                await plugin.handle_msg(_group_event(ctx, _FakeText("哈哈")))
                await asyncio.sleep(0.5 if i == 0 else 0.02)
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(5.0)

    def test_backlog_cannot_form_strong_signal(self):
        # Backlog capped at 30 and never scaled by slot probability: an
        # ordinary question's verdict = question 15 + backlog 6/22/30 =
        # 21/37/45 (rising with the 1/2/3-message burst), far below the
        # threshold 80 even in a low-probability slot — no "fake strong
        # signal"; accumulation runs at ×0.1 = 2.1 + 3.7 + 4.5
        plugin, ctx = _make_plugin(
            {"section_trigger": {"group_reply_chance": 0.1, "merge_wait_seconds": 0.2}}
        )

        async def scenario():
            # First gap much larger than the rest: keeps idle decisively
            # False (see test_low_content_burst_stays_silent_under_cap)
            for i in range(3):
                await plugin.handle_msg(
                    _group_event(ctx, _FakeText(f"这个东西要怎么装上去呢，第{i}次问"))
                )
                await asyncio.sleep(0.5 if i == 0 else 0.02)
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(10.3)

    def test_idle_bonus_reachable_through_handle_msg(self):
        # 7th-review P2 regression: idle must be sampled before the current
        # message is recorded; previously idle_above_average was always False
        # in production and the +15 idle bonus was dead code
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})
        gate = plugin.gates.get(GROUP_SID)
        now = time.time()
        # History: two external messages 10s apart (avg interval 10s), then a
        # ~60s quiet gap before the current message
        gate.note_incoming(now - 70)
        gate.note_incoming(now - 60)

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeText("今天天气不错")))
            await plugin.terminate()

        asyncio.run(scenario())
        # Ordinary no-signal message breaking a long silence: single-message
        # backlog 6 + idle 15 = 21, × group chance 0.5 accumulates 10.5
        # (with the dead idle path it would be 6 × 0.5 = 3.0)
        assert gate.pending_score == pytest.approx(10.5)

    def test_accumulation_triggers_flush(self):
        # Threshold 80: an ordinary question's verdict rises 21/37/45
        # (question 15 + backlog 6/22/30) with the burst; at chance 1.0,
        # three messages accumulate 103 ≥ 80 and trigger
        plugin, ctx = _make_plugin(
            {
                "section_basic": {"max_context_messages": 10},
                "section_trigger": {"group_reply_chance": 1.0, "merge_wait_seconds": 0.2},
            }
        )

        async def scenario():
            for i in range(4):
                await plugin.handle_msg(_group_event(ctx, _FakeText(f"这个东西要怎么装上去呢，第{i}次问")))
                await asyncio.sleep(0.05)
                if ctx.flushed:
                    break
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert GROUP_SID in ctx.flushed

    def test_dormant_slot_suppresses_strong_signal(self):
        plugin, ctx = _make_plugin(
            {
                "section_trigger": {
                    "merge_wait_seconds": 0.2,
                    "activity_slots": [
                        {"name": "全天休眠", "start": "00:00", "end": "23:59", "group": 0, "private": 0}
                    ],
                }
            }
        )
        event = _group_event(ctx, _FakeAt("3631688034"), _FakeText("在吗"))

        async def scenario():
            await plugin.handle_msg(event)
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []

    def test_presence_suppression_reduces_score(self):
        # Bot reply share 100% + isolated single message (backlog 6):
        # mention 80 + 6 - 25 (presence) = 61, below the threshold — under
        # full suppression an isolated mention falls back to weak-signal
        # accumulation and needs content signals or a burst to make up the
        # difference (v1.3.0 semantics: presence reduction can hold back an
        # isolated strong mention)
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})
        gate = plugin.gates.get(GROUP_SID)
        now = time.time()
        for i in range(20):
            gate.note_bot_reply(now - 200 + i)

        event = _group_event(ctx, _FakeText("有人吗"), is_mentioned=True)
        event.message.timestamp = int(now)
        suppressed = evaluate_trigger_score(
            plugin._build_snapshot(event, gate, now, False), plugin.wordlists
        )
        assert suppressed.score == 61

        # Control: the same message with no bot replies scores 86
        # (mention 80 + single-message backlog 6)
        plugin2, ctx2 = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})
        event2 = _group_event(ctx2, _FakeText("有人吗"), is_mentioned=True)
        clean = evaluate_trigger_score(
            plugin2._build_snapshot(event2, plugin2.gates.get(GROUP_SID), now, False),
            plugin2.wordlists,
        )
        assert clean.score == 86


class TestPrivateGating:
    def test_private_full_chance_flushes(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})

        async def scenario():
            await plugin.handle_msg(_dm_event(ctx, _FakeText("随便说点什么")))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert DM_SID in ctx.flushed

    def test_private_zero_chance_silent(self):
        plugin, ctx = _make_plugin(
            {"section_trigger": {"private_reply_chance": 0, "merge_wait_seconds": 0.2}}
        )

        async def scenario():
            await plugin.handle_msg(_dm_event(ctx, _FakeText("随便说点什么")))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []

    def test_private_zero_chance_ignores_random(self):
        plugin, ctx = _make_plugin(
            {"section_trigger": {"private_reply_chance": 0, "merge_wait_seconds": 0.2}}
        )

        async def scenario():
            for _ in range(5):
                await plugin.handle_msg(_dm_event(ctx, _FakeText("再说一句")))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert ctx.flushed == []


class TestLifecycle:
    def test_terminate_cancels_tasks(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt("3631688034"), _FakeText("在吗")))
            tasks = list(plugin.flush_tasks.values())
            await plugin.terminate()
            return tasks

        tasks = asyncio.run(scenario())
        assert all(t.cancelled() or t.done() for t in tasks)
        assert plugin.flush_tasks == {}

    def test_message_sent_tracks_bot_reply(self):
        plugin, ctx = _make_plugin()
        batch_event = _FakeEvent(_FakeMessage([], group=_FakeGroup()), GROUP_SID, ctx)
        result = types.SimpleNamespace(ok=True)
        asyncio.run(plugin.track_bot_reply(batch_event, None, result))
        bot, total = plugin.gates.get(GROUP_SID).window_stats(20, 0)
        assert bot == 1
        assert total == 1


class TestPokeDetection:
    def test_poke_notice_detected(self):
        plugin, ctx = _make_plugin()
        raw = plugin._poke_notice_raw(_poke_event(ctx))
        assert raw is not None and raw["sub_type"] == "poke"

    def test_non_notice_event_ignored(self):
        plugin, ctx = _make_plugin()
        assert plugin._poke_notice_raw(_group_event(ctx, _FakeText("普通消息"))) is None

    def test_self_initiated_poke_ignored(self):
        # Self-initiated pokes (some protocol ends echo outbound events) must
        # not enter the branch — prevents a self-poke loop
        plugin, ctx = _make_plugin()
        assert plugin._poke_notice_raw(_poke_event(ctx, poker=SELF_ID)) is None

    def test_poke_targeting_other_ignored(self):
        # Someone poking someone else: the event is still published (empty
        # chain) but must not enter the poke branch
        plugin, ctx = _make_plugin()
        assert plugin._poke_notice_raw(_poke_event(ctx, target="10001")) is None

    def test_other_notice_type_ignored(self):
        # Other notices perceived by qq-enhance (group_increase etc.) must
        # not be misjudged as pokes
        plugin, ctx = _make_plugin()
        event = _poke_event(ctx)
        event.raw_message = {
            "post_type": "notice",
            "notice_type": "group_increase",
            "sub_type": "approve",
            "user_id": 1,
            "self_id": int(SELF_ID),
        }
        assert plugin._poke_notice_raw(event) is None

    def test_gate_disabled_ignored(self):
        plugin, ctx = _make_plugin({"section_poke": {"poke_gate_enabled": False}})
        assert plugin._poke_notice_raw(_poke_event(ctx)) is None


class TestPokeGating:
    def test_single_poke_accumulates_without_flush(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})
        event = _poke_event(ctx)

        async def scenario():
            await plugin.handle_msg(event)
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
        assert event._buffered is True
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(30)
        assert ctx.flushed == []

    def test_poke_accumulation_triggers_flush(self):
        # Base score 30: the 3rd poke accumulates 90 ≥ threshold 80 and
        # triggers a reply
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})

        async def scenario():
            for _ in range(3):
                await plugin.handle_msg(_poke_event(ctx))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert GROUP_SID in ctx.flushed
        assert plugin.gates.get(GROUP_SID).pending_score == 0.0

    def test_poke_score_unaffected_by_group_chance(self):
        # Group chance 0.1: ordinary weak signals accumulate as score×0.1,
        # pokes still add a fixed +30
        plugin, ctx = _make_plugin(
            {"section_trigger": {"group_reply_chance": 0.1, "merge_wait_seconds": 0.2}}
        )
        asyncio.run(plugin.handle_msg(_poke_event(ctx)))
        assert plugin.gates.get(GROUP_SID).pending_score == pytest.approx(30)

    def test_private_poke_same_logic(self):
        # Private and group share the same logic: no probability roll, the
        # threshold triggers once accumulated
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.2}})

        async def scenario():
            for _ in range(3):
                await plugin.handle_msg(_poke_event(ctx, group=False))
            await asyncio.sleep(0.5)

        asyncio.run(scenario())
        assert DM_SID in ctx.flushed

    def test_dormant_slot_poke_silent(self):
        # Dormant slot (probability 0): no score, no poke-back, but still
        # buffered to keep the context
        plugin, ctx = _make_plugin(
            {
                "section_trigger": {
                    "merge_wait_seconds": 0.2,
                    "activity_slots": [
                        {"name": "全天休眠", "start": "00:00", "end": "23:59", "group": 0, "private": 0}
                    ],
                }
            }
        )
        event = _poke_event(ctx)

        async def scenario():
            await plugin.handle_msg(event)
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
        assert event._buffered is True
        assert plugin.gates.get(GROUP_SID).pending_score == 0.0
        assert ctx.flushed == []

    def test_arm_flush_clears_pending_score(self):
        # Unified clearing covers the private probability path: the
        # accumulated score resets once any trigger point arms the flush
        plugin, ctx = _make_plugin()
        gate = plugin.gates.get(GROUP_SID)
        gate.pending_score = 50.0

        async def scenario():
            plugin._arm_flush(GROUP_SID)
            await asyncio.sleep(0.05)
            await plugin.terminate()

        asyncio.run(scenario())
        assert gate.pending_score == 0.0


class TestPokeBack:
    @staticmethod
    def _make_plugin_with_client(cfg: dict | None = None):
        client = _FakePokeClient()
        ctx = _FakeCtx(poke_client=client)
        merged = {
            "section_trigger": {"merge_wait_seconds": 0.2},
            "section_poke": {"poke_back_enabled": True},
        }
        merged.update(cfg or {})
        return plugin_main.NoriEngineChatPlugin(ctx, merged), ctx, client

    def test_poke_back_sent_when_below_threshold(self):
        plugin, ctx, client = self._make_plugin_with_client()

        async def scenario():
            await plugin.handle_msg(_poke_event(ctx))
            await asyncio.sleep(0.1)

        asyncio.run(scenario())
        assert client.calls == [(POKER_ID, GROUP_ID_NUM)]
        assert plugin.gates.get(GROUP_SID).poke_back_count == 1
        assert ctx.flushed == []

    def test_disabled_by_default(self):
        plugin, ctx = _make_plugin()

        async def scenario():
            await plugin.handle_msg(_poke_event(ctx))
            await asyncio.sleep(0.1)

        asyncio.run(scenario())
        assert plugin.poke_back_enabled is False

    def test_global_cooldown_throttles_concurrent_pokes(self):
        # Default 3s global cooldown: two pokes within a short time only get
        # one poke-back
        plugin, ctx, client = self._make_plugin_with_client()

        async def scenario():
            for _ in range(2):
                await plugin.handle_msg(_poke_event(ctx))
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)

        asyncio.run(scenario())
        assert len(client.calls) == 1

    def test_consecutive_cap_limits_poke_back(self):
        # Cooldown 0: pokes 1/2 get a poke-back, poke 3 accumulates 90 and
        # triggers a reply (no poke-back, and _arm_flush clears the score);
        # poke 4 re-accumulates and poke-backs up to the cap, poke 5 is
        # skipped
        plugin, ctx, client = self._make_plugin_with_client(
            {"section_poke": {"poke_back_enabled": True, "poke_back_cooldown_seconds": 0}}
        )

        async def scenario():
            for _ in range(5):
                await plugin.handle_msg(_poke_event(ctx))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
        assert len(client.calls) == 3
        assert plugin.gates.get(GROUP_SID).poke_back_count == 3

    def test_bot_reply_resets_poke_back_count(self):
        plugin, ctx, client = self._make_plugin_with_client(
            {"section_poke": {"poke_back_enabled": True, "poke_back_cooldown_seconds": 0}}
        )

        async def scenario():
            for _ in range(3):
                await plugin.handle_msg(_poke_event(ctx))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            assert plugin.gates.get(GROUP_SID).poke_back_count == 2
            batch = _FakeEvent(_FakeMessage([], group=_FakeGroup()), GROUP_SID, ctx)
            await plugin.track_bot_reply(batch, None, types.SimpleNamespace(ok=True))
            # After the counter resets, another poke can poke-back again
            await plugin.handle_msg(_poke_event(ctx))
            await asyncio.sleep(0.05)

        asyncio.run(scenario())
        assert len(client.calls) == 3

    def test_terminate_cancels_poke_back_tasks(self):
        plugin, ctx, client = self._make_plugin_with_client()

        async def scenario():
            await plugin.handle_msg(_poke_event(ctx))
            tasks = list(plugin._poke_back_tasks)
            await plugin.terminate()
            return tasks

        tasks = asyncio.run(scenario())
        assert all(t.cancelled() or t.done() for t in tasks)
        assert plugin._poke_back_tasks == set()


class TestDebounceIdleExit:
    """P2 fix: the merge loop self-exits on idle and cleans up its
    registrations, so gates become reclaimable by prune."""

    def test_idle_exit_cleans_registrations(self, monkeypatch):
        monkeypatch.setattr(plugin_main, "_DEBOUNCE_IDLE_EXIT_SECONDS", 0.3)
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            # merge_wait (0.1) flush + idle exit (0.3), with ample margin
            await asyncio.sleep(1.0)

        asyncio.run(scenario())
        # task/event registrations were cleaned up with the idle exit and no
        # longer accumulate with the number of sessions
        assert GROUP_SID not in plugin.flush_tasks
        assert GROUP_SID not in plugin.flush_events
        # The flush really happened before the exit (not exit-without-trigger)
        assert ctx.flushed == [GROUP_SID]
        # The gate is no longer protected by an in-flight task and prune can
        # reclaim it (P2 leak trio closed)
        removed = plugin.gates.prune(
            keep_sids=set(plugin.flush_tasks), max_idle_seconds=0.0
        )
        assert removed == 1

    def test_rearm_after_idle_exit_recreates_task(self, monkeypatch):
        monkeypatch.setattr(plugin_main, "_DEBOUNCE_IDLE_EXIT_SECONDS", 0.3)
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            await asyncio.sleep(1.0)
            assert GROUP_SID not in plugin.flush_tasks
            # Simulate the host round-end signal (the fake environment
            # dispatches no on-final-result); otherwise the busy state
            # after the first flush would defer the second trigger instead
            # of recreating the task
            await plugin._on_final_result(
                _batch_event_for(GROUP_SID, ctx), _final_result_event()
            )
            # New trigger after the idle exit: _arm_flush must rebuild the
            # task/event and flush normally
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("还在吗")))
            task = plugin.flush_tasks.get(GROUP_SID)
            assert task is not None and not task.done()
            await asyncio.sleep(0.8)
            return task

        task = asyncio.run(scenario())
        assert task.done()
        assert ctx.flushed == [GROUP_SID, GROUP_SID]
        assert GROUP_SID not in plugin.flush_tasks


class TestUnifiedClearing:
    """P3 closure: clearing happens only in _arm_flush; strong-signal and
    accumulation paths behave unchanged."""

    def test_strong_signal_clears_pending_via_arm_flush(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})
        gate = plugin.gates.get(GROUP_SID)
        gate.pending_score = 50.0

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            await asyncio.sleep(0.3)  # wait out the merge_wait(0.1) flush
            await plugin.terminate()

        asyncio.run(scenario())
        assert gate.pending_score == 0.0
        assert ctx.flushed == [GROUP_SID]

    def test_accumulation_trigger_clears_pending_via_arm_flush(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})
        gate = plugin.gates.get(GROUP_SID)
        gate.pending_score = 77.0

        async def scenario():
            # Any ordinary message on top crosses the threshold 80 via the
            # accumulation path: single-message backlog 6 × group chance 0.5
            # = +3 → 77+3 = 80, exactly at the line
            await plugin.handle_msg(_group_event(ctx, _FakeText("今天天气不错")))
            await asyncio.sleep(0.3)  # wait out the merge_wait(0.1) flush
            await plugin.terminate()

        asyncio.run(scenario())
        assert gate.pending_score == 0.0
        assert ctx.flushed == [GROUP_SID]


class TestConfigWarnings:
    def test_waking_words_invalid_type_warns_and_falls_back(self, caplog):
        # waking_words configured as a plain string: warning + fallback to
        # an empty tuple (P5.4)
        with caplog.at_level(logging.WARNING, logger="test.kira.plugin"):
            plugin, _ = _make_plugin({"section_basic": {"waking_words": "诺里"}})
        assert plugin.waking_words == ()
        assert any("waking_words" in r.message for r in caplog.records)

    def test_initialize_warns_when_default_chat_active(self, caplog):
        # Running alongside the built-in default-chat: initialize logs the
        # mutual-exclusion warning (double-processing risk)
        plugin, ctx = _make_plugin()
        ctx.get_plugin_inst = lambda pid: object() if pid == "default-chat" else None

        async def scenario():
            with caplog.at_level(logging.WARNING, logger="test.kira.plugin"):
                await plugin.initialize()
            await plugin.terminate()

        asyncio.run(scenario())
        assert any("default-chat" in r.message for r in caplog.records)

    def test_now_hhmm_uses_configured_timezone(self):
        # With locale.TZ configured: the wall-clock time for slot resolution
        # follows that timezone
        from datetime import datetime
        from zoneinfo import ZoneInfo

        plugin, ctx = _make_plugin()
        ctx.get_timezone = lambda: ZoneInfo("UTC")

        def _next_minute(hhmm: str) -> str:
            h, m = map(int, hhmm.split(":"))
            total = (h * 60 + m + 1) % (24 * 60)
            return f"{total // 60:02d}:{total % 60:02d}"

        utc = datetime.now(ZoneInfo("UTC")).strftime("%H:%M")
        # Allow a ±1 minute tolerance for crossing a minute boundary; if the
        # timezone were ignored, local time would differ from UTC by hours
        # on a non-UTC deployment and fail
        assert plugin._now_hhmm() in (utc, _next_minute(utc))

    def test_empty_chain_notice_skipped_from_scoring(self):
        # Empty-chain notices such as pokes aimed at a third party: not
        # buffered, not recorded in the timeline, not scored
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})
        event = _group_event(ctx)  # chain is empty
        event.is_notice = True
        event.raw_message = {
            "post_type": "notice", "notice_type": "group_increase",
            "self_id": int(SELF_ID),
        }

        async def scenario():
            await plugin.handle_msg(event)
            await asyncio.sleep(0.3)

        asyncio.run(scenario())
        assert event._buffered is False
        assert plugin.gates.all_sids() == []


def _final_result_event():
    """Minimal stand-in for the host round-end hook payload (KiraFinalResult)."""
    return types.SimpleNamespace(step_results=[])


def _batch_event_for(sid: str, ctx: "_FakeCtx") -> "_FakeEvent":
    """Minimal batch-event stand-in for driving the round-end hook."""
    return _FakeEvent(_FakeMessage([], group=_FakeGroup()), sid, ctx)


def _sid_event(ctx: "_FakeCtx", sid: str, text: str) -> "_FakeEvent":
    """Group text event on an arbitrary sid (cluster-branch sessions)."""
    return _FakeEvent(_FakeMessage([_FakeText(text)], group=_FakeGroup()), sid, ctx)


class TestFollowupWindow:
    """Topic continuity: the per-message follow-up tier anchored at the
    bot's last reply. One trunk scenario — default off → in-window boost →
    threshold crossing → window expiry — with branch assertions inline."""

    def test_followup_window_lifecycle(self):
        # Branch 0: feature is disabled by default (no tasks started, so no
        # terminate needed for the throwaway instance)
        default_plugin, _ = _make_plugin()
        assert default_plugin.followup_window == 0.0

        plugin, ctx = _make_plugin(
            {"section_topic": {"followup_window_seconds": 10.0},
             "section_trigger": {"group_reply_chance": 1.0,
                                 "merge_wait_seconds": 0.1}}
        )
        gate = plugin.gates.get(GROUP_SID)

        async def scenario():
            # Phase 1: bot replied just now → follow-up earns the 50 tier
            # 50(话题延续) + 6(积压) - 18(存在感抑制 1/2 占比) = 38，×1.0 累积
            gate.note_bot_reply(time.time())
            await plugin.handle_msg(_group_event(ctx, _FakeText("今天天气不错")))
            assert gate.pending_score == pytest.approx(38.0)
            assert ctx.flushed == []

            # Phase 2: second follow-up still in window — accumulated heat
            # crosses 80 → 累积触发并刷出（触发点清零累积分）
            await plugin.handle_msg(_group_event(ctx, _FakeText("就是随便聊聊而已")))
            # 38 + [50(话题延续) + 22(2条突发积压) - 6(抑制 1/3)] = 104 ≥ 80
            await asyncio.sleep(0.3)
            assert ctx.flushed == [GROUP_SID]
            assert gate.pending_score == 0.0

            # Phase 3: bot's latest timeline entry is now 30s old — beyond
            # the 10s window → plain scoring, no tier (appending the stale
            # bot entry makes it the latest anchor scanned from the right)
            gate.note_bot_reply(time.time() - 30.0)
            gate.pending_score = 50.0
            await plugin.handle_msg(_group_event(ctx, _FakeText("接着说")))
            await asyncio.sleep(0.3)
            # 无延续档：0 + 30(3条积压封顶) - 11(抑制 2/5) = 19 → 69 < 80 不触发
            assert ctx.flushed == [GROUP_SID]
            assert gate.pending_score == pytest.approx(69.0)
            await plugin.terminate()

        asyncio.run(scenario())


class TestClusterGate:
    """Topic-cluster detection: per-message heat condition sampled from
    prior participants; the bonus accumulates unscaled by slot probability.
    One trunk: default off → hot cluster boosts unscaled → same-user /
    content-less clusters never light → rapid burst reaches the threshold."""

    def test_cluster_gate_lifecycle(self):
        # Branch 0: master switch off by default — the switch (not the
        # window) is the on/off control, so the window default stays 30
        default_plugin, _ = _make_plugin()
        assert default_plugin.cluster_gate_enabled is False
        assert default_plugin.cluster_window == 30.0

        plugin, ctx = _make_plugin(
            {"section_topic": {"cluster_gate_enabled": True,
                               "cluster_window_seconds": 30,
                               "cluster_min_users": 3,
                               "cluster_bonus": 50},
             "section_trigger": {"group_reply_chance": 0.5,
                                 "merge_wait_seconds": 0.1}}
        )

        async def scenario():
            now = time.time()
            # Phase 1: 3 distinct users with real content, spread beyond the
            # 5s burst window → hot. Verdict 50(聚集)+6(积压)+15(闲时)=71 < 80;
            # accumulation keeps the bonus unscaled: (71-50)×0.5+50 = 60.5
            # (diluting the whole verdict by 0.5 would only give 35.5)
            gate = plugin.gates.get(GROUP_SID)
            gate.note_incoming(now - 25, sender="u1", low_value=False)
            gate.note_incoming(now - 20, sender="u2", low_value=False)
            gate.note_incoming(now - 15, sender="u3", low_value=False)
            await plugin.handle_msg(_group_event(ctx, _FakeText("我也有同感")))
            assert gate.pending_score == pytest.approx(60.5)
            assert ctx.flushed == []

            # Phase 2: one user spamming never completes a cluster —
            # verdict 21(6积压+15闲时)×0.5，无加成
            sid_b = "napcat:gm:999999"
            gate_b = plugin.gates.get(sid_b)
            for ts in (now - 25, now - 20, now - 15):
                gate_b.note_incoming(ts, sender="u1", low_value=False)
            await plugin.handle_msg(_sid_event(ctx, sid_b, "我也有同感"))
            assert gate_b.pending_score == pytest.approx(10.5)

            # Phase 3: distinct users but all low-value content → cold
            sid_c = "napcat:gm:888888"
            gate_c = plugin.gates.get(sid_c)
            for idx, ts in enumerate((now - 25, now - 20, now - 15)):
                gate_c.note_incoming(ts, sender=f"u{idx}", low_value=True)
            await plugin.handle_msg(_sid_event(ctx, sid_c, "我也有同感"))
            assert gate_c.pending_score == pytest.approx(10.5)

            # Phase 4: rapid 3-user burst inside the 5s window → backlog
            # caps at 30 → 50+30 = 80 reaches the threshold → strong
            # trigger and flush
            sid_d = "napcat:gm:777777"
            gate_d = plugin.gates.get(sid_d)
            gate_d.note_incoming(now - 3, sender="u1", low_value=False)
            gate_d.note_incoming(now - 2, sender="u2", low_value=False)
            gate_d.note_incoming(now - 1, sender="u3", low_value=False)
            await plugin.handle_msg(_sid_event(ctx, sid_d, "我也有同感"))
            await asyncio.sleep(0.3)
            assert ctx.flushed == [sid_d]
            await plugin.terminate()

        asyncio.run(scenario())


class TestTopicFeatureScope:
    """One shared session scope (topic_focus_sessions) governs BOTH topic
    features: an empty list means global, a non-empty list enables the
    features only for the listed sessions. One trunk with an in-scope /
    out-of-scope branch per feature."""

    def test_topic_features_session_scope(self):
        # Branch 0: default scope is empty → both features act globally
        default_plugin, _ = _make_plugin()
        assert default_plugin.topic_focus_sessions == frozenset()

        sid_scoped = "napcat:gm:444444"
        plugin, ctx = _make_plugin(
            {"section_topic": {
                "followup_window_seconds": 10,
                "cluster_gate_enabled": True,
                # One list scoping BOTH features
                "topic_focus_sessions": [GROUP_SID, sid_scoped],
            },
             "section_trigger": {"group_reply_chance": 1.0}}
        )

        async def scenario():
            now = time.time()
            # Topic-continuity branch: same fresh bot reply on both sessions
            # — only the in-scope one earns the tier (38)；the out-of-scope
            # verdict clamps to 0 (6 积压 - 18 抑制)
            gate_a = plugin.gates.get(GROUP_SID)
            gate_a.note_bot_reply(now)
            await plugin.handle_msg(_group_event(ctx, _FakeText("今天天气不错")))
            assert gate_a.pending_score == pytest.approx(38.0)

            sid_b = "napcat:gm:555555"
            gate_b = plugin.gates.get(sid_b)
            gate_b.note_bot_reply(now)
            await plugin.handle_msg(_sid_event(ctx, sid_b, "今天天气不错"))
            assert gate_b.pending_score == 0.0

            # Cluster branch: identical 3-user hot clusters on both sessions
            # — the SAME shared scope decides (in-scope sid_scoped gets the
            # bonus 71, out-of-scope 21)
            gate_c = plugin.gates.get(sid_scoped)
            gate_c.note_incoming(now - 25, sender="u1", low_value=False)
            gate_c.note_incoming(now - 20, sender="u2", low_value=False)
            gate_c.note_incoming(now - 15, sender="u3", low_value=False)
            await plugin.handle_msg(_sid_event(ctx, sid_scoped, "我也有同感"))
            assert gate_c.pending_score == pytest.approx(71.0)

            sid_d = "napcat:gm:666666"
            gate_d = plugin.gates.get(sid_d)
            gate_d.note_incoming(now - 25, sender="u1", low_value=False)
            gate_d.note_incoming(now - 20, sender="u2", low_value=False)
            gate_d.note_incoming(now - 15, sender="u3", low_value=False)
            await plugin.handle_msg(_sid_event(ctx, sid_d, "我也有同感"))
            assert gate_d.pending_score == pytest.approx(21.0)
            await plugin.terminate()

        asyncio.run(scenario())


class TestBusyQueue:
    """LLM round serialization: busy deferral / round-end release /
    watchdog / self-healing / queue cap."""

    def test_busy_strong_signal_defers_without_flush_or_clear(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})
        gate = plugin.gates.get(GROUP_SID)
        gate.pending_score = 60.0

        async def scenario():
            plugin._mark_busy(GROUP_SID)
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            # terminate cleans the busy state: the deferred assertion must
            # happen before teardown
            assert GROUP_SID in plugin.deferred_sids
            await plugin.terminate()

        asyncio.run(scenario())
        assert ctx.flushed == []
        # The accumulated score is kept during busy (cleared only when
        # released at round end)
        assert gate.pending_score == 60.0

    def test_round_end_event_releases_deferred_and_flushes(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            await plugin.initialize()
            plugin._mark_busy(GROUP_SID)
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            assert GROUP_SID in plugin.deferred_sids
            # Round-end hooks of other sessions must not affect this
            # session's busy state
            await plugin._on_final_result(
                _batch_event_for("napcat:gm:999999", ctx), _final_result_event()
            )
            assert GROUP_SID in plugin.busy_sessions
            # This session's round ends → release the deferred trigger →
            # flush through the merge window
            await plugin._on_final_result(
                _batch_event_for(GROUP_SID, ctx), _final_result_event()
            )
            assert GROUP_SID not in plugin.busy_sessions
            await asyncio.sleep(0.3)
            await plugin.terminate()

        asyncio.run(scenario())
        assert ctx.flushed == [GROUP_SID]
        assert plugin.gates.get(GROUP_SID).pending_score == 0.0

    def test_busy_holds_after_flush_until_round_end(self):
        # P1 regression: PluginContext.flush_session_messages does not forward
        # the processor's bool, so the publish result must come from
        # message_processor directly — busy holds from the flush publish
        # until the round-end signal, instead of being released by the
        # empty-flush path right after every flush
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            await plugin.initialize()
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            await asyncio.sleep(0.3)  # merge window passed, flush published
            assert ctx.flushed == [GROUP_SID]
            # Round still in flight: busy must hold
            assert GROUP_SID in plugin.busy_sessions
            await plugin._on_final_result(
                _batch_event_for(GROUP_SID, ctx), _final_result_event()
            )
            assert GROUP_SID not in plugin.busy_sessions
            await plugin.terminate()

        asyncio.run(scenario())

    def test_watchdog_timeout_forces_release(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            plugin.busy_hold_timeout = 0.15  # bypass the config lower bound to shorten the watchdog
            plugin._mark_busy(GROUP_SID)
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            assert GROUP_SID in plugin.deferred_sids
            await asyncio.sleep(0.4)
            # Watchdog timeout → force-release the deferred trigger → flush
            # through the merge window
            assert ctx.flushed == [GROUP_SID]
            # The post-release flush means a new round has started: busy
            # being set again is by design
            assert GROUP_SID in plugin.busy_sessions
            await plugin.terminate()

        asyncio.run(scenario())

    def test_llm_request_marks_and_refreshes_busy(self):
        plugin, ctx = _make_plugin()
        batch = _FakeEvent(_FakeMessage([], group=_FakeGroup()), GROUP_SID, ctx)
        req = types.SimpleNamespace(system_prompt=[])

        async def scenario():
            # Untracked round: ON_LLM_REQUEST adopts it on arrival
            # (self-heals a lost round-end signal)
            await plugin.inject_group_prompt(batch, req)
            assert GROUP_SID in plugin.busy_sessions
            t1 = plugin.busy_watchdogs[GROUP_SID]
            # New round starts: the watchdog resets (old task cancelled,
            # replaced)
            await plugin.inject_group_prompt(batch, req)
            assert plugin.busy_watchdogs[GROUP_SID] is not t1
            await asyncio.sleep(0.01)
            assert t1.cancelled() or t1.done()
            await plugin.terminate()

        asyncio.run(scenario())
        assert GROUP_SID not in plugin.busy_sessions

    def test_busy_expands_buffer_cap(self):
        plugin, ctx = _make_plugin({"section_basic": {"max_context_messages": 2}})

        async def scenario():
            # Normal state: cap 2, oldest trimmed beyond it
            for i in range(5):
                await plugin.handle_msg(_group_event(ctx, _FakeText(f"闲聊{i}")))
            assert ctx.get_buffer(GROUP_SID).get_length() == 2
            # While busy: enlarged to busy_queue_max (default 10), messages
            # keep queuing without trimming
            plugin._mark_busy(GROUP_SID)
            for i in range(5):
                await plugin.handle_msg(_group_event(ctx, _FakeText(f"排队{i}")))
            assert ctx.get_buffer(GROUP_SID).get_length() == 7
            await plugin.terminate()

        asyncio.run(scenario())

    def test_flush_pending_expands_cap_through_merge_window(self):
        plugin, ctx = _make_plugin(
            {"section_basic": {"max_context_messages": 2},
             "section_trigger": {"merge_wait_seconds": 0.1}}
        )

        async def scenario():
            # Strong trigger arms: messages arriving inside the merge window
            # are not evicted by the normal cap
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            for i in range(4):
                await plugin.handle_msg(_group_event(ctx, _FakeText(f"后续{i}")))
            assert ctx.get_buffer(GROUP_SID).get_length() == 5
            await asyncio.sleep(0.3)
            await plugin.terminate()

        asyncio.run(scenario())
        assert ctx.flushed == [GROUP_SID]

    def test_deferred_skips_rescoring(self):
        plugin, ctx = _make_plugin()
        gate = plugin.gates.get(GROUP_SID)

        async def scenario():
            plugin._mark_busy(GROUP_SID)
            plugin.deferred_sids.add(GROUP_SID)
            gate.pending_score = 70.0
            await plugin.handle_msg(_group_event(ctx, _FakeText("今天天气不错")))
            await plugin.terminate()

        asyncio.run(scenario())
        # Already deferred: subsequent messages only enter the buffer, no
        # repeated scoring/accumulation
        assert gate.pending_score == 70.0
        assert ctx.flushed == []

    def test_private_defers_during_busy(self):
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def scenario():
            plugin._mark_busy(DM_SID)
            await plugin.handle_msg(_dm_event(ctx, _FakeText("在吗")))
            assert DM_SID in plugin.deferred_sids
            await plugin.terminate()

        asyncio.run(scenario())
        assert ctx.flushed == []

    def test_poke_accumulates_and_defers_during_busy(self):
        plugin, ctx = _make_plugin()
        gate = plugin.gates.get(GROUP_SID)
        gate.pending_score = 60.0

        async def scenario():
            plugin._mark_busy(GROUP_SID)
            await plugin.handle_msg(_poke_event(ctx))
            # terminate cleans the busy state: the deferred assertion must
            # happen before teardown
            assert GROUP_SID in plugin.deferred_sids
            await plugin.terminate()

        asyncio.run(scenario())
        # +30 reaches the threshold but busy: deferred without clearing
        assert gate.pending_score == 90.0
        assert ctx.flushed == []

    def test_terminate_cleans_busy_state(self):
        plugin, ctx = _make_plugin()

        async def scenario():
            # Hot-reload re-entry: initialize must stay re-entrant
            await plugin.initialize()
            await plugin.initialize()
            plugin._mark_busy(GROUP_SID)
            plugin.deferred_sids.add(GROUP_SID)
            await plugin.terminate()

        asyncio.run(scenario())
        assert plugin.busy_sessions == set()
        assert plugin.deferred_sids == set()
        assert plugin.busy_watchdogs == {}
        assert plugin.flush_pending == set()

    def test_empty_flush_releases_busy_immediately(self):
        # Flush comes back empty-handed (buffer drained first by another
        # consumer): no batch was published, so no round-end event will
        # arrive — busy must be released immediately instead of waiting for
        # the watchdog timeout
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def fake_flush(sid, extra_event=None):
            ctx.get_buffer(sid).items.clear()
            return False

        ctx.message_processor.flush_session_messages = fake_flush

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            await asyncio.sleep(0.4)
            assert GROUP_SID not in plugin.busy_sessions
            assert GROUP_SID not in plugin.flush_pending
            await plugin.terminate()

        asyncio.run(scenario())

    def test_flush_exception_clears_flush_pending(self):
        # Flush raises: flush_pending must be cleaned up (in the finally
        # backstop), otherwise the session's buffer cap stays permanently
        # enlarged to the busy-queue value
        plugin, ctx = _make_plugin({"section_trigger": {"merge_wait_seconds": 0.1}})

        async def boom(sid, extra_event=None):
            raise RuntimeError("flush boom")

        ctx.message_processor.flush_session_messages = boom

        async def scenario():
            await plugin.handle_msg(_group_event(ctx, _FakeAt(SELF_ID), _FakeText("在吗")))
            await asyncio.sleep(0.4)
            assert GROUP_SID not in plugin.flush_pending
            # The merge loop exited on the exception; its registration was
            # cleaned up too
            assert GROUP_SID not in plugin.flush_tasks
            await plugin.terminate()

        asyncio.run(scenario())
