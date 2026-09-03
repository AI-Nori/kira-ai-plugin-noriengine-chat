"""Real-host smoke test (P6: registration-surface blind spot).

The other tests in this directory inject fake ``core.*`` modules into
``sys.modules`` to isolate the host — at the cost that the real registration
surface of the ``@on`` decorators and the contracts of ``KiraMessageEvent``
/ ``KiraMessageBatchEvent`` / ``LLMRequest`` etc. are never verified: if the
host changes a hook signature or data structure, those tests stay green.

This file imports the REAL KiraAI host in a **subprocess** (the repository
root is located by walking up from this plugin directory), loads the
plugin's main.py and asserts:

1. The three event hooks are registered through the real ``@on`` decorators
   into plugin_components with correct event types and priorities
   (im_message=HIGH / message_sent=LOW / llm_request=MEDIUM);
2. Calling the bound methods with the real event classes upholds the
   signature and data-structure assumptions (the ``buffer`` strategy of
   ``handle_msg``, the poke branch, the ``(event, action, result)``
   positional signature of ``track_bot_reply``, the ``Prompt.name/.content``
   structure of ``LLMRequest.system_prompt``), and that PluginContext still
   exposes ``message_processor`` (the plugin reads the flush publish result
   from it directly).

The subprocess isolates ``sys.modules`` so it does not pollute the
fake-host tests; when the host repository is missing or its dependencies
cannot be installed (``HOST_IMPORTED`` not printed) the test skips instead
of failing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _find_repo_root() -> Path | None:
    """Locate the KiraAI repository root (a directory containing the
    core/plugin package) by walking up from the plugin directory."""

    for candidate in _PLUGIN_DIR.parents:
        if (candidate / "core" / "plugin" / "__init__.py").is_file():
            return candidate
    return None


# Subprocess script: argv[1]=repo_root argv[2]=plugin_dir.
# Only ASCII phase markers are printed (HOST_IMPORTED / SMOKE_OK) to avoid
# console encoding differences.
_SMOKE_SCRIPT = r'''
import asyncio
import importlib.util
import sys
from types import SimpleNamespace

repo_root, plugin_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)
sys.path.insert(0, plugin_dir)

# ---- Phase 1: real host import ----
from core.plugin.plugin_handlers import EventType, Priority
from core.plugin.plugin_registry import _plugin_components
from core.plugin.plugin_context import PluginContext
from core.chat.message_utils import (
    KiraIMMessage, KiraMessageBatchEvent, KiraMessageEvent, MessageChain,
)
from core.chat.message_elements import At, Text
from core.chat.session import Group, Session, User
from core.adapter.adapter_info import AdapterInfo
from core.provider import LLMRequest
from core.prompt_manager import Prompt

# The plugin reads the flush publish result from message_processor directly
# (the PluginContext.flush_session_messages wrapper does not forward it)
assert "message_processor" in PluginContext.__dataclass_fields__

print("HOST_IMPORTED", flush=True)

# ---- Phase 2: load the plugin's main.py (triggers real @on registration) ----
spec = importlib.util.spec_from_file_location(
    "noriengine_host_smoke_main", plugin_dir + "/main.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

PLUGIN_ID = "kira-ai-plugin-noriengine-chat"
comp = _plugin_components[PLUGIN_ID]
hooks = {h.handler.__name__: h for h in comp.hooks}

assert {"handle_msg", "track_bot_reply", "inject_group_prompt"} <= set(hooks), sorted(hooks)
assert hooks["handle_msg"].event_type == EventType.ON_IM_MESSAGE
assert hooks["handle_msg"].priority == Priority.HIGH
assert hooks["track_bot_reply"].event_type == EventType.ON_MESSAGE_SENT
assert hooks["track_bot_reply"].priority == Priority.LOW
assert hooks["inject_group_prompt"].event_type == EventType.ON_LLM_REQUEST
assert hooks["inject_group_prompt"].priority == Priority.MEDIUM

# ---- Phase 3: drive the bound methods with real event classes ----
class _StubProcessor:
    def __init__(self, ctx):
        self._ctx = ctx

    async def flush_session_messages(self, sid, extra_event=None):
        self._ctx.flushed.append(sid)
        return True


class _StubCtx:
    # Mirrors the real PluginContext: flush goes through message_processor
    # (the PluginContext wrapper does not forward the publish result)
    def __init__(self):
        self.flushed = []
        self.message_processor = _StubProcessor(self)

    def get_buffer(self, sid):
        return None


SELF_ID = "3631688034"
GROUP_SID = "napcat:gm:1105211570"
adapter = AdapterInfo(enabled=True, adapter_id="1", name="napcat", platform="QQ")


def make_event(chain, *, group=True, is_notice=False, raw=None, is_mentioned=False):
    msg = KiraIMMessage(
        message_id="m1",
        self_id=SELF_ID,
        chain=chain,
        timestamp=1700000000,
        sender=User(user_id="242817150", nickname="测试用户"),
        group=Group(group_id="1105211570", group_name="测试群") if group else None,
        is_notice=is_notice,
        is_mentioned=is_mentioned,
        raw_message=raw,
    )
    return KiraMessageEvent(
        message_types=["text"], timestamp=msg.timestamp, message=msg, adapter=adapter
    )


async def main():
    plugin = mod.NoriEngineChatPlugin(_StubCtx(), {})
    await plugin.initialize()

    # @bot group message: real KiraMessageEvent + bound method signature
    event = make_event(MessageChain([At(SELF_ID), Text("在吗")]), is_mentioned=True)
    await plugin.handle_msg(event)
    assert event.process_strategy == "buffer", event.process_strategy

    # Poke: real notice raw structure + is_notice flag going through gate scoring
    poke = make_event(
        MessageChain([Text("[Poke 用户242817150(测试用户)戳了戳你]")]),
        is_notice=True,
        is_mentioned=True,
        raw={
            "post_type": "notice", "notice_type": "notify", "sub_type": "poke",
            "user_id": 242817150, "target_id": int(SELF_ID), "self_id": int(SELF_ID),
            "group_id": 1105211570,
        },
    )
    await plugin.handle_msg(poke)
    gate = plugin.gates.get(GROUP_SID)
    assert gate.pending_score == plugin.poke_base_score, gate.pending_score

    # message_sent: real batch event + (event, action, result) positional signature
    batch = KiraMessageBatchEvent(
        message_types=["text"],
        timestamp=1700000000,
        session=Session(
            adapter_name="napcat", session_type="gm",
            session_id="1105211570", session_title="测试群",
        ),
        messages=[poke.message],
    )
    gate.poke_back_count = 2
    await plugin.track_bot_reply(batch, None, SimpleNamespace(ok=True))
    assert gate.poke_back_count == 0

    # llm_request: real LLMRequest.system_prompt Prompt.name/.content structure
    plugin.group_chat_prompt = "（群聊提示）"
    req = LLMRequest(system_prompt=[Prompt("基础人设", name="chat_env")])
    await plugin.inject_group_prompt(batch, req)
    assert req.system_prompt[0].content.endswith("（群聊提示）")

    # No need to wait out the merge_wait flush: verify the task was created,
    # then terminate for teardown
    assert GROUP_SID in plugin.flush_tasks
    await plugin.terminate()
    assert plugin.flush_tasks == {}
    print("SMOKE_OK", flush=True)


asyncio.run(main())
'''


class TestRealHostSmoke:
    def test_plugin_against_real_kira_host(self, tmp_path):
        repo_root = _find_repo_root()
        if repo_root is None:
            pytest.skip("未找到 KiraAI 宿主仓库根（core/plugin 包），跳过真宿主冒烟")

        # Host logging writes to <cwd>/data/log.log: point the subprocess
        # cwd at a temporary directory to avoid writing test logs into the
        # real repository's data/; sys.path is injected explicitly and
        # unaffected
        (tmp_path / "data").mkdir()
        proc = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT, str(repo_root), str(_PLUGIN_DIR)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=180,
            cwd=str(tmp_path),
        )
        if proc.returncode != 0:
            if "HOST_IMPORTED" not in proc.stdout:
                pytest.skip(f"宿主依赖不可用，无法执行真宿主导入：\n{proc.stderr[-2000:]}")
            pytest.fail(
                f"真宿主冒烟失败（注册面或契约断言不通过）：\n{proc.stderr[-4000:]}"
            )
        assert "SMOKE_OK" in proc.stdout
