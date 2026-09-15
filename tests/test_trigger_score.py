"""Unit tests for the reply trigger scoring engine."""

import pytest

from noriengine_trigger_score import (
    SignalWordlists,
    TriggerSnapshot,
    clean_chat_text,
    evaluate_trigger_score,
    find_opinion_hint,
    find_request_hint,
    is_low_content_batch,
    looks_like_question,
)


def _snapshot(**overrides) -> TriggerSnapshot:
    """Build a scoring snapshot: group chat, no signals, no backlog,
    pace 1.0 by default."""

    defaults = dict(
        texts=["今天天气不错"],
        has_at=False,
        has_mention=False,
        is_group_chat=True,
        pending_count=1,
        backlog_norm=1,
        bot_recent_replies=0,
        recent_window_total=0,
        pace_factor=1.0,
    )
    defaults.update(overrides)
    return TriggerSnapshot(**defaults)


class TestCleanChatText:
    def test_strip_at_all(self):
        assert clean_chat_text("@all 大家好") == ""

    def test_strip_at_mention(self):
        assert clean_chat_text("@user hello world") == "hello world"

    def test_strip_angle_bracket_mention(self):
        assert clean_chat_text("@<某人> 中午吃什么") == "中午吃什么"

    def test_preserve_normal_text(self):
        assert clean_chat_text("这是一段正常的文本") == "这是一段正常的文本"

    def test_whitespace_normalized(self):
        assert clean_chat_text("  多个   空白  \n 符号  ") == "多个 空白 符号"

    def test_empty_and_none(self):
        assert clean_chat_text("") == ""
        assert clean_chat_text(None) == ""


class TestIsLowContentBatch:
    def test_all_short_reactions(self):
        assert is_low_content_batch(["哈哈", "嗯", "哦"]) is True

    def test_mixed_with_long_text(self):
        assert is_low_content_batch(["哈哈", "这是一段很长的文本"]) is False

    def test_empty_list(self):
        assert is_low_content_batch([]) is True

    def test_single_short_reaction(self):
        assert is_low_content_batch(["6"]) is True

    def test_non_reaction(self):
        assert is_low_content_batch(["hello world"]) is False

    def test_custom_wordlist_injection(self):
        custom = frozenset({"ovar"})
        assert is_low_content_batch(["ovar"], custom) is True
        assert is_low_content_batch(["哈哈"], custom) is False


class TestLooksLikeQuestion:
    def test_question_term(self):
        assert looks_like_question("怎么做？") is True

    def test_what_question(self):
        assert looks_like_question("这是什么？") is True

    def test_no_question(self):
        assert looks_like_question("今天天气不错") is False

    def test_short_question_with_ma(self):
        assert looks_like_question("这个好吗？") is True

    def test_question_mark_only(self):
        assert looks_like_question("？") is False

    def test_pure_interjection_wrap(self):
        assert looks_like_question("？啊？") is False

    def test_custom_terms(self):
        assert looks_like_question("哪能买到", terms=("哪能",)) is True
        assert looks_like_question("怎么弄", terms=("哪能",)) is False


class TestFindRequestHint:
    WORDS = SignalWordlists()

    def test_direct_request_in_group(self):
        assert find_request_hint("帮我查个天气", self.WORDS, direct_context=True) == "帮我"

    def test_direct_request_without_context(self):
        # Without direct context "帮我" still hits (ordinary direct request term)
        assert find_request_hint("帮我查个天气", self.WORDS, direct_context=False) == "帮我"

    def test_weak_request_requires_context(self):
        words = self.WORDS
        assert find_request_hint("需要一些帮助", words, direct_context=True) == "需要"
        assert find_request_hint("需要一些帮助", words, direct_context=False) == ""

    def test_weak_direct_terms_without_context(self):
        words = self.WORDS
        assert find_request_hint("可以吗", words, direct_context=False) == ""
        assert find_request_hint("可以吗", words, direct_context=True) == "可以吗"

    def test_neng_bu_neng_boundary(self):
        words = self.WORDS
        # "能不能" does not hit in non-direct context unless at message start
        assert find_request_hint("他问我能不能这样", words, direct_context=False) == ""
        assert find_request_hint("能不能帮我", words, direct_context=False) == "帮我/能不能"
        assert find_request_hint("你能不能帮我", words, direct_context=True) == "帮我/能不能"

    def test_rival_ai_excluded_without_context(self):
        words = self.WORDS
        assert find_request_hint("DeepSeek，帮我看看", words, direct_context=False) == ""
        # In direct context, naming another AI still hits (already talking
        # to the bot)
        assert find_request_hint("DeepSeek，帮我看看", words, direct_context=True) == "帮我"

    def test_empty_text(self):
        assert find_request_hint("", self.WORDS, direct_context=True) == ""


class TestFindOpinionHint:
    WORDS = SignalWordlists()

    def test_opinion_term_direct(self):
        assert find_opinion_hint("你觉得这个咋样", self.WORDS, direct_context=True) == "你觉得"

    def test_opinion_without_context_never_hits(self):
        # Opinion seeking never hits without direct context; when nickname
        # triggering is desired, add it to the waking-word list
        words = self.WORDS
        assert find_opinion_hint("你们怎么看", words, direct_context=False) == ""
        assert find_opinion_hint("汐祈你怎么看", words, direct_context=False) == ""

    def test_bu_kan_excluded(self):
        assert find_opinion_hint("我不怎么看电影", self.WORDS, direct_context=True) == ""

    def test_zenme_kan_proximity(self):
        words = self.WORDS
        assert find_opinion_hint("你过来说怎么看", words, direct_context=True) == "怎么看"


class TestEvaluateTriggerScore:
    def test_at_direct_hit(self):
        verdict = evaluate_trigger_score(_snapshot(texts=["在吗"], has_at=True))
        # Direct signal 100 + backlog cap 30 = 130 → pace 1.0 → 130
        assert verdict.score >= 80
        assert "直接信号=100(@)" in verdict.breakdown

    def test_mention_reaches_default_threshold_exactly(self):
        # Mention=80 + no content signals + no backlog (pending_count=0)
        # = 80, exactly the default threshold
        verdict = evaluate_trigger_score(
            _snapshot(texts=["好的收到"], has_mention=True, pending_count=0, backlog_norm=1)
        )
        assert verdict.score == 80
        assert "直接信号=80(提及)" in verdict.breakdown

    def test_private_chat_base_relevance(self):
        verdict = evaluate_trigger_score(_snapshot(texts=["嗯"], is_group_chat=False))
        assert "直接信号=40(私聊)" in verdict.breakdown

    def test_plain_group_message_zero_relevance(self):
        verdict = evaluate_trigger_score(_snapshot(texts=["今天天气不错"], pending_count=0))
        assert "直接信号=0(普通)" not in verdict.breakdown  # 0 points leave no breakdown entry
        assert verdict.score < 80

    def test_content_signals_stack(self):
        # Question +15, request +20, long text +5 — and the mention (80)
        # provides direct context
        text = "帮我看看这个问题怎么解决好不好，有一段比较长的描述内容需要处理一下"
        verdict = evaluate_trigger_score(_snapshot(texts=[text], has_mention=True))
        assert "问题" in verdict.breakdown
        assert "请求:帮我" in verdict.breakdown

    def test_long_text_bonus(self):
        short = evaluate_trigger_score(_snapshot(texts=["好"], pending_count=0))
        long_text = "x" * 45
        long_verdict = evaluate_trigger_score(_snapshot(texts=[long_text], pending_count=0))
        assert long_verdict.score > short.score

    def test_low_content_penalty(self):
        verdict = evaluate_trigger_score(_snapshot(texts=["哈哈"], pending_count=0))
        assert "低价值" in verdict.breakdown

    def test_media_message_not_low_value(self):
        # Voice/image/video-only message: no text to analyze, but media
        # presence is engagement — fixed points, never the empty-batch
        # penalty
        verdict = evaluate_trigger_score(_snapshot(texts=[], has_media=True, pending_count=0))
        assert "低价值" not in verdict.breakdown
        assert "语音/图片/视频" in verdict.breakdown
        assert verdict.score == 15

    def test_weak_media_message_not_low_value(self):
        # File/forward-only message: minor tier — fixed 10 points, still
        # never the empty-batch penalty
        verdict = evaluate_trigger_score(
            _snapshot(texts=[], has_weak_media=True, pending_count=0)
        )
        assert "低价值" not in verdict.breakdown
        assert "文件/转发" in verdict.breakdown
        assert verdict.score == 10

    def test_weak_media_never_stacks_onto_media(self):
        # A message packing both tiers is one message's worth of
        # engagement: the major tier wins
        verdict = evaluate_trigger_score(
            _snapshot(texts=[], has_media=True, has_weak_media=True, pending_count=0)
        )
        assert verdict.score == 15
        assert "文件/转发" not in verdict.breakdown

    def test_weak_media_stacks_with_text_content(self):
        base = evaluate_trigger_score(_snapshot(texts=["这是什么"], pending_count=0))
        with_weak = evaluate_trigger_score(
            _snapshot(texts=["这是什么"], has_weak_media=True, pending_count=0)
        )
        assert with_weak.score == base.score + 10

    def test_media_exempts_low_value_reaction_text(self):
        # "哈哈" alongside an image: the image is real content, the batch
        # is not classified as a low-value reaction
        verdict = evaluate_trigger_score(
            _snapshot(texts=["哈哈"], has_media=True, pending_count=0)
        )
        assert "低价值" not in verdict.breakdown

    def test_media_stacks_with_text_content(self):
        base = evaluate_trigger_score(_snapshot(texts=["这是什么"], pending_count=0))
        with_media = evaluate_trigger_score(
            _snapshot(texts=["这是什么"], has_media=True, pending_count=0)
        )
        assert with_media.score == base.score + 15

    def test_backlog_pressure_quadratic(self):
        low = evaluate_trigger_score(_snapshot(texts=["呃"], pending_count=1, backlog_norm=4))
        high = evaluate_trigger_score(_snapshot(texts=["呃"], pending_count=4, backlog_norm=4))
        assert high.score > low.score
        # Full threshold saturation 50, truncated to 30 by the global cap
        assert "积压=30" in high.breakdown

    def test_backlog_global_cap(self):
        # However large the backlog (the log segment can reach 100), the
        # global cap clamps it to 30
        for pending in (5, 20, 100):
            verdict = evaluate_trigger_score(
                _snapshot(texts=["呃"], pending_count=pending, backlog_norm=1)
            )
            assert "积压=30" in verdict.breakdown
            assert verdict.score <= 30

    def test_backlog_overflow_capped(self):
        full = evaluate_trigger_score(_snapshot(texts=["呃"], pending_count=20, backlog_norm=4))
        assert "积压=30" in full.breakdown

    def test_idle_bonus(self):
        base = evaluate_trigger_score(_snapshot(texts=["呃"], pending_count=1, backlog_norm=4))
        idle = evaluate_trigger_score(
            _snapshot(texts=["呃"], pending_count=1, backlog_norm=4, idle_above_average=True)
        )
        assert idle.score > base.score
        assert "闲时+15" in idle.breakdown

    def test_presence_suppression(self):
        clean = evaluate_trigger_score(_snapshot(texts=["在吗"], has_mention=True))
        suppressed = evaluate_trigger_score(
            _snapshot(
                texts=["在吗"],
                has_mention=True,
                bot_recent_replies=15,
                recent_window_total=20,
            )
        )
        assert suppressed.score < clean.score
        assert "存在感=-25" in suppressed.breakdown

    def test_presence_free_below_ratio(self):
        free = evaluate_trigger_score(
            _snapshot(
                texts=["在吗"],
                has_mention=True,
                bot_recent_replies=2,
                recent_window_total=20,
            )
        )
        assert "存在感" not in free.breakdown

    def test_pace_scaling(self):
        full = evaluate_trigger_score(
            _snapshot(texts=["好的"], has_mention=True, pace_factor=1.0, pending_count=0)
        )
        half = evaluate_trigger_score(
            _snapshot(texts=["好的"], has_mention=True, pace_factor=0.5, pending_count=0)
        )
        assert full.score == 80
        # 80 × 0.75 = 60
        assert half.score == 60
        assert "倍率=0.75" in half.breakdown

    def test_score_never_negative(self):
        verdict = evaluate_trigger_score(
            _snapshot(
                texts=["哈哈"],
                pending_count=0,
                bot_recent_replies=15,
                recent_window_total=20,
            )
        )
        assert verdict.score >= 0

    def test_custom_wordlists_apply(self):
        words = SignalWordlists.from_config(
            {"question_terms": ["咋整"], "seek_terms": ["搭把手"], "low_content_replies": ["嘻嘻"]}
        )
        verdict = evaluate_trigger_score(_snapshot(texts=["这事儿咋整"], has_mention=True), words)
        assert "问题" in verdict.breakdown
        low = evaluate_trigger_score(_snapshot(texts=["嘻嘻"], pending_count=0), words)
        assert "低价值" in low.breakdown
        # Defaults no longer apply: a statement containing only "怎么" hits
        # the default wordlist but not the customized one
        assert "问题" not in evaluate_trigger_score(
            _snapshot(texts=["我想学怎么游泳，教练说先练憋气"], pending_count=0), words
        ).breakdown

    def test_wordlists_from_config_fallback(self):
        words = SignalWordlists.from_config({"question_terms": "不是列表"})
        assert words.question_terms == ("怎么", "如何", "为什么", "有没有")
        words2 = SignalWordlists.from_config({"question_terms": []})
        assert words2.question_terms == ("怎么", "如何", "为什么", "有没有")

    def test_custom_rival_names(self):
        words = SignalWordlists.from_config({"rival_ai_names": ["小爱同学"]})
        assert find_request_hint("小爱同学，帮我定闹钟", words, direct_context=False) == ""
        assert find_request_hint("DeepSeek，帮我看看", words, direct_context=False) == "帮我"


class TestFollowupTier:
    """Topic continuity: the follow-up direct-signal tier (per-message,
    between mention 80 and private 40). One trunk scenario with branch
    assertions on the same snapshot shape."""

    def test_followup_tier_scoring(self):
        # Branch 1: base tier — 50 alone stays below the default threshold
        verdict = evaluate_trigger_score(
            _snapshot(texts=["今天天气不错"], followup=True, followup_score=50, pending_count=0)
        )
        assert "直接信号=50(话题延续)" in verdict.breakdown
        assert verdict.score == 50

        # Branch 2: grants direct context — casual seek term "看看"
        # (direct-only) becomes effective, low-value penalty still applies
        unlocked = evaluate_trigger_score(
            _snapshot(texts=["看看这个"], followup=True, followup_score=50, pending_count=0)
        )
        assert "请求:看看" in unlocked.breakdown
        assert unlocked.score == 70
        assert evaluate_trigger_score(
            _snapshot(texts=["哈哈"], followup=True, followup_score=50, pending_count=0)
        ).score == 25

        # Branch 3: stronger direct tiers dominate the followup tier
        assert "直接信号=100(@)" in evaluate_trigger_score(
            _snapshot(texts=["在吗"], has_at=True, followup=True, followup_score=50, pending_count=0)
        ).breakdown
        assert "直接信号=80(提及)" in evaluate_trigger_score(
            _snapshot(
                texts=["好的收到"], has_mention=True, followup=True, followup_score=50,
                pending_count=0, backlog_norm=1,
            )
        ).breakdown


class TestClusterBonus:
    """Topic-cluster bonus: additive (grants no direct context) and its
    verdict share is exposed as ``unscaled`` so the accumulation call site
    can bypass slot-probability dilution."""

    def test_cluster_bonus_scoring(self):
        # Branch 1: hot cluster adds the bonus; casual seek terms stay
        # locked — ambient heat is not directed at the bot
        v = evaluate_trigger_score(
            _snapshot(texts=["看看这个"], cluster_hot=True, cluster_bonus=50, pending_count=0)
        )
        assert v.score == 50 and v.unscaled == 50
        assert "话题聚集=50" in v.breakdown
        assert "请求:看看" not in v.breakdown

        # Branch 2: pace scaling moves score and unscaled together
        # (multiplier 0.5+0.5×0.5=0.75 → round(50×0.75)=38)
        v2 = evaluate_trigger_score(
            _snapshot(
                texts=["看看这个"], cluster_hot=True, cluster_bonus=50,
                pending_count=0, pace_factor=0.5,
            )
        )
        assert v2.score == 38 and v2.unscaled == 38

        # Branch 3: heavy suppression can shrink the unscaled share — it is
        # capped at the final score, so presence suppression still brakes it
        v3 = evaluate_trigger_score(
            _snapshot(
                texts=["哈哈"], cluster_hot=True, cluster_bonus=50, pending_count=0,
                bot_recent_replies=15, recent_window_total=20,
            )
        )
        assert v3.score == 0 and v3.unscaled == 0


class TestBacklogNotScaled:
    """Backlog points are never scaled by slot probability: verdict and
    accumulation share the same score, the cap of 30 is consumed as-is."""

    def test_score_includes_full_capped_backlog(self):
        snap = _snapshot(texts=["这个怎么装"], pending_count=1, backlog_norm=1)  # question 15 + backlog 30
        verdict = evaluate_trigger_score(snap)
        assert verdict.score == 45
        assert "积压=30" in verdict.breakdown

    def test_worst_case_nondirect_exactly_reaches_threshold(self):
        # Threshold-reaching combo: question 15 + request 20 + long text 15
        # + backlog 30 = 80, exactly the default threshold
        long_text = "帮我处理一下这个问题怎么解决，" + "细节" * 60
        snap = _snapshot(texts=[long_text], pending_count=1, backlog_norm=1)
        assert evaluate_trigger_score(snap).score == 80


class TestRivalPatternPrecompile:
    """P4.2: the rival regex is precompiled and cached at construction."""

    def test_pattern_precompiled_and_equivalent(self):
        words = SignalWordlists.from_config({"rival_ai_names": ["小爱", "DeepSeek"]})
        pattern = words.rival_pattern()
        assert pattern.pattern == r"^(?:小爱|DeepSeek)[，,、\s]"
        assert pattern.search("DeepSeek，帮我写诗")
        assert not pattern.search("DeepSeek是谁")
        # The same snapshot returns the cached instance (no re-compiling
        # per call)
        assert words.rival_pattern() is pattern

    def test_default_wordlists_has_pattern(self):
        assert SignalWordlists().rival_pattern().search("ChatGPT 帮我")

    def test_pattern_field_excluded_from_equality(self):
        a = SignalWordlists.from_config({"rival_ai_names": ["A"]})
        b = SignalWordlists.from_config({"rival_ai_names": ["A"]})
        assert a == b
