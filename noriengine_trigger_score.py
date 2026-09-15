"""Reply trigger scoring engine.

Quantitatively scores the current message snapshot as the gating basis for
whether to trigger an LLM reply. Scoring dimensions:

- Direct signals: @bot, mention (nickname / waking word / reply-to-bot),
  follow-up window (topic continuity), private-chat session;
- Content signals: question, request, opinion seeking, text length,
  voice/image/video media bonus, file/forward minor bonus, low-value
  phrase penalty;
- Topic-cluster bonus: ambient heat from several distinct participants,
  additive (not a direct-signal tier) and unscaled by time-slot
  probability at the accumulation call site;
- Backlog pressure: the reply urgency brought by pending message volume;
- Presence suppression: self-restraint when the bot's recent reply share
  is too high;
- Pace scaling: overall scaling of the raw score by the bot's current
  activity level.

Computation::

    backlog = min(cap 30, backlog curve points)                       # global cap
    verdict = max(0, round((direct + content + backlog - suppression) × pace multiplier))

``verdict >= trigger_threshold`` recommends triggering a reply (strong
signal); otherwise stay silent, with weak signals accumulating per message
as ``verdict × time-slot probability``. The backlog is never scaled by
time-slot probability (message density counts even in low-probability
slots); the slot probability only weights the weak-signal accumulation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import log1p
from typing import Sequence

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
BACKLOG_BASE_POINTS = 50
"""Base points of backlog pressure at full saturation within the threshold."""

BACKLOG_CEILING = 100
"""Theoretical ceiling of backlog pressure points."""

BACKLOG_PRESSURE_CAP = 30
"""Global cap of backlog pressure points: however large the backlog, the
pressure score never exceeds this.

Safety boundary: the content-score ceiling of a non-direct-context message
is 65 (question 15 + request 20 + long text 15 + major media 15; the
file/forward minor tier never stacks onto the major tier), so the cap is
chosen as ``threshold 80 - 50 = 30`` (pre-media ceiling): flooding can only
raise the backlog to 30, and an ordinary group message (no @/mention) only
crosses the default threshold 80 in the extreme combination of fully maxed
content signals plus a topped-out backlog (still further reduced by
presence suppression and the pace multiplier). With the backlog
normalization floor of 3 (see
``noriengine_session_state._BACKLOG_NORM_FLOOR``), an isolated single
message gets only 6 backlog points (65+6=71 < 80), so the
threshold-reaching combination requires a real burst of ≥3 messages within
5 seconds.
"""

BACKLOG_SATURATION_RATIO = 5.0
"""At how many times the normalization threshold the backlog approaches its
ceiling."""

IDLE_BACKLOG_BONUS = 15
"""Extra backlog points when the session has been idle longer than the
average message interval."""

FOLLOWUP_BASE_POINTS = 50
"""Default relevance points of the topic-continuity follow-up tier.

A message arriving within the configured follow-up window after the bot's
last reply is likely directed at the bot, so it earns this direct-signal
relevance tier (between mention 80 and private 40) and unlocks the
direct-context-only request/opinion hints. The tier is per-message: every
message scored inside the window earns it (a new bot reply re-anchors the
window). Alone it stays below the default threshold 80 — a plain follow-up
accumulates as a weak signal; two quick follow-ups, or one combined with
content signals, cross the threshold. Feature disabled by default
(``followup_window_seconds = 0`` in the plugin config).
"""

CLUSTER_BONUS_POINTS = 50
"""Default topic-cluster bonus points.

When at least the configured number of distinct external participants
(default 3) have posted real (non-low-value) content within the cluster
window (default 30s when enabled), every subsequently scored message earns
this bonus on top of its own signals. It is an ambient-heat additive
component — NOT a direct-signal tier: a hot topic is not directed at the
bot, so it grants no direct context. Its share of the verdict is reported
as ``unscaled`` and accumulates without the time-slot probability
weighting: diluting it (×0.5 by default) would negate the acceleration it
exists to provide (poke scores accumulate raw the same way). Disabled by
default (``cluster_window_seconds = 0`` in the plugin config).
"""

MEDIA_CONTENT_POINTS = 15
"""Fixed content points for a message carrying voice/image/video segments.

The trigger hook scores the raw message chain before the host runs STT or
image description (those only happen at LLM-round prompt building), so a
voice/image/video-only message has no analyzable text and would otherwise
be classified as an empty low-value batch (-25) — penalizing exactly the
engagement it represents. Media presence instead earns this fixed medium
content score and exempts the low-value penalty."""

WEAK_MEDIA_CONTENT_POINTS = 10
"""Fixed content points for a message carrying file/forward segments.

Same rationale as :data:`MEDIA_CONTENT_POINTS` but a minor tier: a file or
merged-forward bundle is real content yet weaker engagement than voice or
image. Never stacks onto the major tier (a message carrying both is one
message's worth of engagement — the major tier wins), keeping the
non-direct-context content ceiling at 65."""

PRESENCE_FREE_RATIO = 0.25
"""Below this bot-reply share inside the window, no presence suppression is
applied."""

PRESENCE_FULL_RATIO = 0.60
"""At this bot-reply share inside the window, presence suppression reaches
its ceiling."""

PRESENCE_PENALTY_CEILING = 25
"""Theoretical ceiling of presence suppression points."""

# ---------------------------------------------------------------------------
# Default signal wordlists (overridable via WebUI config, see schema.json)
# ---------------------------------------------------------------------------
DEFAULT_QUESTION_TERMS = ("怎么", "如何", "为什么", "有没有")
DEFAULT_SEEK_TERMS = ("帮我", "帮忙", "能不能", "可以吗", "要不要")
DEFAULT_CASUAL_SEEK_TERMS = ("需要", "求", "看看", "试试")
DEFAULT_OPINION_TERMS = ("你觉得", "你认为", "咋看", "有什么建议")
DEFAULT_LOW_CONTENT_REPLIES = (
    "哈哈", "哈哈哈", "草", "笑死", "好", "嗯", "啊", "哦", "6", "666", "？", "?",
)
DEFAULT_RIVAL_AI_NAMES = ("DeepSeek", "ChatGPT", "Grok", "豆包", "千问", "元宝", "通义", "Kimi", "Claude")


@dataclass(frozen=True)
class SignalWordlists:
    """Snapshot of the scoring signal wordlists (built from plugin config,
    immutable at runtime)."""

    question_terms: tuple[str, ...] = DEFAULT_QUESTION_TERMS
    seek_terms: tuple[str, ...] = DEFAULT_SEEK_TERMS
    casual_seek_terms: tuple[str, ...] = DEFAULT_CASUAL_SEEK_TERMS
    opinion_terms: tuple[str, ...] = DEFAULT_OPINION_TERMS
    low_content_replies: frozenset[str] = frozenset(DEFAULT_LOW_CONTENT_REPLIES)
    rival_ai_names: tuple[str, ...] = DEFAULT_RIVAL_AI_NAMES
    # Precompiled cache of the "names another AI at message start" pattern
    # (derived in __post_init__; excluded from construction/repr/comparison)
    _rival_pattern: "re.Pattern | None" = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rival_pattern", self._compile_rival_pattern())

    def _compile_rival_pattern(self) -> "re.Pattern":
        names = "|".join(re.escape(name) for name in self.rival_ai_names)
        return re.compile(rf"^(?:{names})[，,、\s]")

    @classmethod
    def from_config(cls, cfg: dict) -> "SignalWordlists":
        """Build the wordlists from plugin config; missing or invalid items
        fall back to the defaults."""

        def _tuple_of_str(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
            raw = cfg.get(key)
            if not isinstance(raw, (list, tuple)):
                return fallback
            cleaned = tuple(str(item).strip() for item in raw if str(item).strip())
            return cleaned or fallback

        def _frozenset_of_str(key: str, fallback: tuple[str, ...]) -> frozenset[str]:
            return frozenset(_tuple_of_str(key, fallback))

        return cls(
            question_terms=_tuple_of_str("question_terms", DEFAULT_QUESTION_TERMS),
            seek_terms=_tuple_of_str("seek_terms", DEFAULT_SEEK_TERMS),
            casual_seek_terms=_tuple_of_str("casual_seek_terms", DEFAULT_CASUAL_SEEK_TERMS),
            opinion_terms=_tuple_of_str("opinion_terms", DEFAULT_OPINION_TERMS),
            low_content_replies=_frozenset_of_str("low_content_replies", DEFAULT_LOW_CONTENT_REPLIES),
            rival_ai_names=_tuple_of_str("rival_ai_names", DEFAULT_RIVAL_AI_NAMES),
        )

    def rival_pattern(self) -> "re.Pattern":
        """The matching pattern for "names another AI assistant at message
        start" (precompiled cache)."""
        return self._rival_pattern


# ---------------------------------------------------------------------------
# Input/output data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TriggerSnapshot:
    """Runtime snapshot of a session needed for one scoring pass."""

    texts: Sequence[str]
    has_at: bool
    has_mention: bool
    is_group_chat: bool
    pending_count: int
    backlog_norm: int
    bot_recent_replies: int
    recent_window_total: int
    pace_factor: float
    idle_above_average: bool = False
    # Voice/image/video segments present in the raw chain (no text extracted
    # from them at scoring time — see MEDIA_CONTENT_POINTS)
    has_media: bool = False
    # File/forward segments present in the raw chain — minor media tier,
    # see WEAK_MEDIA_CONTENT_POINTS (never stacks onto has_media)
    has_weak_media: bool = False
    # The message arrived within the configured follow-up window after the
    # bot's last reply — topic-continuity direct-signal tier; every message
    # scored inside the window earns it, see FOLLOWUP_BASE_POINTS
    followup: bool = False
    followup_score: int = 0
    # Topic-cluster heat: enough distinct participants with real content
    # posted within the cluster window — every scored message while hot
    # earns the additive bonus, see CLUSTER_BONUS_POINTS
    cluster_hot: bool = False
    cluster_bonus: int = 0


@dataclass(frozen=True)
class TriggerVerdict:
    """Scoring result: verdict score + human-readable breakdown.

    ``score`` serves both the strong-signal threshold check and the base of
    the weak-signal accumulation (the accumulation side applies the
    time-slot probability weighting at the call site — see ``pending_score``
    in main.py). ``unscaled`` is the portion of ``score`` (the
    topic-cluster bonus) that must bypass that weighting when accumulating:
    diluting it by the slot probability would negate its acceleration.
    """

    score: int
    breakdown: str
    unscaled: int = 0


# ---------------------------------------------------------------------------
# Text cleaning and helper predicates
# ---------------------------------------------------------------------------
def clean_chat_text(text: str) -> str:
    """Clean message text: normalize whitespace, strip @ mentions, and drop
    pure @all-style message bodies."""

    normalized = " ".join((text or "").split()).strip()
    if normalized.startswith("@all"):
        return ""
    # Strip @mentions (including the angle-bracket display-name form)
    normalized = re.sub(r"@<[^>]+>|@\S+", "", normalized).strip()
    return normalized


def is_low_content_batch(
    texts: Sequence[str],
    low_content: frozenset[str] = frozenset(DEFAULT_LOW_CONTENT_REPLIES),
) -> bool:
    """Whether the pending messages consist mostly of low-value short
    reactions ("哈哈"/"6" etc.)."""

    normalized_texts = [" ".join(text.split()).strip() for text in texts if text.strip()]
    if not normalized_texts:
        return True
    if any(len(text) > 8 for text in normalized_texts):
        return False
    return all(text in low_content for text in normalized_texts)


def looks_like_question(text: str, terms: Sequence[str] = DEFAULT_QUESTION_TERMS) -> bool:
    """Whether the current utterance forms a real question (as opposed to an
    exclamation or an overly short mood fragment)."""

    if not text:
        return False
    # Overly short pure-mood fragments wrapped in punctuation ("？XX？") are
    # not questions
    if re.fullmatch(r"[？?！!~～…\s]+[\w\u4e00-\u9fff]{1,4}[？?！!~～…\s]+", text):
        return False
    if any(term in text for term in terms):
        return True
    if re.search(r"(?<![这那没])什么", text):
        return True
    if re.search(r"[吗呢](?:[？?。！!~～…]*$)", text) and 4 <= len(text) <= 80:
        return True
    return bool(re.search(r"[？?](?:$|[。！!~～…])", text) and 4 <= len(text) <= 120)


def find_request_hint(
    text: str,
    wordlists: SignalWordlists,
    *,
    direct_context: bool,
) -> str:
    """Return the hit reason of a request-type signal; weak request terms
    only take effect in direct context.

    In non-direct context (no @/mention in a group), a message that names
    another AI at its start ("DeepSeek, help me...") is not treated as a
    request to the bot.
    """

    if not text:
        return ""
    if not direct_context and wordlists.rival_pattern().search(text):
        return ""
    direct_hits = [term for term in wordlists.seek_terms if term in text]
    if "能不能" in direct_hits and not (direct_context or text.startswith("能不能")):
        direct_hits.remove("能不能")
    if not direct_context:
        for weak_direct_term in ("可以吗", "要不要"):
            if weak_direct_term in direct_hits:
                direct_hits.remove(weak_direct_term)
    if direct_hits:
        return "/".join(direct_hits)
    if direct_context:
        casual_hits = [term for term in wordlists.casual_seek_terms if term in text]
        if casual_hits:
            return "/".join(casual_hits)
    return ""


def find_opinion_hint(
    text: str,
    wordlists: SignalWordlists,
    *,
    direct_context: bool,
) -> str:
    """Return the hit reason of an opinion-seeking signal.

    Only effective in direct context (@/mention/private); opinion seeking in
    an ordinary group message only counts after hitting a waking word and
    becoming a mention.
    """

    if "不怎么看" in text:
        return ""
    if not direct_context:
        return ""
    hits = [term for term in wordlists.opinion_terms if term in text]
    if hits:
        return "/".join(hits)
    if re.search(r"你.{0,6}怎么看|怎么看.{0,6}你", text):
        return "怎么看"
    return ""


# ---------------------------------------------------------------------------
# Core scoring flow
# ---------------------------------------------------------------------------
def evaluate_trigger_score(
    snapshot: TriggerSnapshot,
    wordlists: SignalWordlists | None = None,
    low_content_set: frozenset[str] | None = None,
) -> TriggerVerdict:
    """Compute the reply trigger score.

    ``low_content_set`` allows injecting a low-value phrase set different
    from ``wordlists.low_content_replies`` (a compatibility hook kept for
    the old call shape; usually no need to pass it explicitly).
    """

    words = wordlists or SignalWordlists()
    low_content = low_content_set or words.low_content_replies

    # ---- Direct signals ----
    if snapshot.has_at:
        relevance = 100
        relevance_reason = "@"
    elif snapshot.has_mention:
        relevance = 80
        relevance_reason = "提及"
    elif snapshot.followup and snapshot.followup_score > 0:
        relevance = snapshot.followup_score
        relevance_reason = "话题延续"
    elif not snapshot.is_group_chat:
        relevance = 40
        relevance_reason = "私聊"
    else:
        relevance = 0
        relevance_reason = "普通"
    direct_context = relevance > 0

    # ---- Content signals ----
    cleaned_texts = [clean_chat_text(text) for text in snapshot.texts]
    cleaned_texts = [text for text in cleaned_texts if text]
    combined = "\n".join(cleaned_texts)
    content_points, content_reasons = _content_points(
        cleaned_texts,
        combined,
        words,
        low_content,
        direct_context=direct_context,
        has_media=snapshot.has_media,
        has_weak_media=snapshot.has_weak_media,
    )

    # ---- Backlog pressure ----
    pressure = _backlog_pressure(
        pending_count=snapshot.pending_count,
        backlog_norm=max(1, snapshot.backlog_norm),
        idle_above_average=snapshot.idle_above_average,
    )

    # ---- Presence suppression ----
    suppression = _presence_suppression(
        bot_recent_replies=snapshot.bot_recent_replies,
        recent_window_total=snapshot.recent_window_total,
    )

    # ---- Topic-cluster bonus ----
    # Ambient heat from several distinct participants: additive, not a
    # direct-signal tier (grants no direct context)
    cluster_points = (
        snapshot.cluster_bonus if snapshot.cluster_hot and snapshot.cluster_bonus > 0 else 0
    )

    raw = relevance + content_points + pressure + cluster_points - suppression

    # ---- Pace scaling ----
    pace = min(1.0, snapshot.pace_factor)
    multiplier = 0.5 + 0.5 * pace
    final = max(0, int(round(raw * multiplier)))
    # The cluster bonus's share of the final score, reported so the
    # accumulation call site can add it without slot-probability dilution;
    # capped at ``final`` so presence suppression still brakes it
    unscaled = min(final, int(round(cluster_points * multiplier))) if cluster_points else 0

    parts = [f"最终={final}", f"原始={raw}"]
    if relevance != 0:
        parts.append(f"直接信号={relevance}({relevance_reason})")
    if content_points != 0:
        parts.append(f"内容={content_points}({','.join(content_reasons)})")
    if pressure != 0:
        parts.append(f"积压={pressure}")
    if cluster_points:
        parts.append(f"话题聚集={cluster_points}")
    if suppression != 0:
        parts.append(
            f"存在感=-{suppression}(窗口={snapshot.bot_recent_replies}/{snapshot.recent_window_total})"
        )
    if snapshot.idle_above_average:
        parts.append("闲时+15")
    parts.extend([f"节奏={pace:.3f}", f"倍率={multiplier:.2f}"])

    return TriggerVerdict(score=final, breakdown=" ".join(parts), unscaled=unscaled)


# ---------------------------------------------------------------------------
# Per-dimension scoring functions
# ---------------------------------------------------------------------------
def _presence_suppression(*, bot_recent_replies: int, recent_window_total: int) -> int:
    """Presence suppression points from the bot-reply share inside the
    recent window."""

    if bot_recent_replies <= 0 or recent_window_total <= 0:
        return 0

    self_ratio = min(1.0, bot_recent_replies / recent_window_total)
    if self_ratio <= PRESENCE_FREE_RATIO:
        return 0

    ratio_span = PRESENCE_FULL_RATIO - PRESENCE_FREE_RATIO
    progress = min(1.0, (self_ratio - PRESENCE_FREE_RATIO) / ratio_span)
    return int(round(PRESENCE_PENALTY_CEILING * progress))


def _backlog_pressure(*, pending_count: int, backlog_norm: int, idle_above_average: bool) -> int:
    """Backlog pressure points from the pending message count: quadratic
    growth within the threshold, logarithmic growth beyond it.

    The result is globally capped at :data:`BACKLOG_PRESSURE_CAP`.
    """

    ratio = max(0.0, pending_count / backlog_norm)
    if ratio <= 1.0:
        points = int(round(BACKLOG_BASE_POINTS * ratio * ratio))
        if idle_above_average:
            points += IDLE_BACKLOG_BONUS
        points = min(BACKLOG_BASE_POINTS, points)
    else:
        overflow = ratio - 1.0
        full_overflow = BACKLOG_SATURATION_RATIO - 1.0
        factor = min(1.0, log1p(overflow) / log1p(full_overflow))
        points = BACKLOG_BASE_POINTS + int(round((BACKLOG_CEILING - BACKLOG_BASE_POINTS) * factor))
        points = min(BACKLOG_CEILING, points)
    return min(BACKLOG_PRESSURE_CAP, points)


def _content_points(
    cleaned_texts: Sequence[str],
    combined_text: str,
    wordlists: SignalWordlists,
    low_content: frozenset[str],
    *,
    direct_context: bool,
    has_media: bool = False,
    has_weak_media: bool = False,
) -> tuple[int, list[str]]:
    """Compute content points from the cleaned text."""

    points = 0
    reasons: list[str] = []

    if any(looks_like_question(text, wordlists.question_terms) for text in cleaned_texts):
        points += 15
        reasons.append("问题")

    request_hint = find_request_hint(combined_text, wordlists, direct_context=direct_context)
    if request_hint:
        points += 20
        reasons.append(f"请求:{request_hint}")

    opinion_hint = find_opinion_hint(
        combined_text, wordlists, direct_context=direct_context
    )
    if opinion_hint:
        points += 20
        reasons.append(f"征询:{opinion_hint}")

    total_length = len(combined_text)
    if total_length >= 40:
        points += 5
        reasons.append("长文本")
    if total_length >= 120:
        points += 10
        reasons.append("较长文本")

    # Media tiers are mutually exclusive (major wins): one message is one
    # message's worth of engagement no matter how many segments it packs
    if has_media:
        points += MEDIA_CONTENT_POINTS
        reasons.append("语音/图片/视频")
    elif has_weak_media:
        points += WEAK_MEDIA_CONTENT_POINTS
        reasons.append("文件/转发")

    # A media message cannot be judged by its (missing) text — the
    # voice/image/file itself is the engagement, so it never counts as a
    # low-value batch
    if is_low_content_batch(cleaned_texts, low_content) and not (
        has_media or has_weak_media
    ):
        points -= 25
        reasons.append("低价值")

    return points, reasons
