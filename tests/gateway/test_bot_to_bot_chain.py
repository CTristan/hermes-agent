"""Bot-to-bot exchanges end (#137027): the messaging gateway's chain rules (``BotLoopGuard.chain_admit``).
The Bot Mode hop cap is ``tests/agent/test_bot_chain.py``; the tool entry ``tests/tools/test_bot_mode_dm_chain.py``.

Two Hermes gateways DMing each other on Mattermost ran 17 h because every inbound bot message
produced an outbound one (runtime notices included) and nothing counted the exchange.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings, configured_bot_user, settings_from_config
from gateway.session import Platform, SessionSource

CHAT = "dm_peer"
PEER = "peer_bot"
HUMAN = "alice"


@pytest.fixture
def clock():
    state = {"t": 1_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda s: state.__setitem__("t", state["t"] + s))


def _guard(clock, **overrides):
    settings = BotLoopGuardSettings(**{"max_events": 1000, "window_seconds": 60, "cooldown_seconds": 600, **overrides})
    return BotLoopGuard(settings=lambda: settings, clock=clock.now)


# --- messaging gateway chain rules -------------------------------------------------------------


def test_a_bot_ping_pong_stops_at_max_hops_and_a_human_message_starts_a_new_chain(clock):
    guard = _guard(clock, max_hops=3, bot_replies_expect_reply=True)
    verdicts = [guard.chain_admit(CHAT, PEER, f"ping {n}", is_bot=True)[0] for n in range(10)]
    assert verdicts == [True, True, True] + [False] * 7

    assert guard.chain_admit(CHAT, HUMAN, "keep going", is_bot=False) == (True, "human")
    assert guard.chain_admit(CHAT, PEER, "ping 10", is_bot=True) == (True, "ok")


def test_a_bot_answering_this_bots_reply_starts_no_turn_by_default(clock):
    """The default reply rule: our answer to a bot needs no answer, so its answer to that ends it."""
    guard = _guard(clock)
    assert guard.chain_admit(CHAT, PEER, "what is 2+2?", is_bot=True) == (True, "ok")
    guard.note_reply(CHAT, PEER)  # our "4" went out
    assert guard.chain_admit(CHAT, PEER, "thanks!", is_bot=True) == (False, "reply_to_reply")
    # Another bot in the same chat is a new ask, not a reply to our reply.
    assert guard.chain_admit(CHAT, "other_bot", "hello", is_bot=True) == (True, "ok")


def test_a_second_ask_before_any_reply_is_not_a_reply(clock):
    """Greptile 4467194010: admission is not an answer. A bot's follow-up while its first message waits
    (or after a turn that stayed silent) is a new ask, still capped by max_hops."""
    guard = _guard(clock, max_hops=3)
    assert guard.chain_admit(CHAT, PEER, "question", is_bot=True) == (True, "ok")
    assert guard.chain_admit(CHAT, PEER, "also this", is_bot=True) == (True, "ok")
    guard.note_reply(CHAT, PEER)
    assert guard.chain_admit(CHAT, PEER, "thanks", is_bot=True) == (False, "reply_to_reply")


def test_the_same_ask_repeated_inside_one_chain_is_one_hop(clock):
    guard = _guard(clock, bot_replies_expect_reply=True)
    assert guard.chain_admit(CHAT, PEER, "Interrupting current task", is_bot=True) == (True, "ok")
    assert guard.chain_admit(CHAT, "bot_b", "x", is_bot=True)[0] is True
    assert guard.chain_admit(CHAT, PEER, "  Interrupting   current task ", is_bot=True) == (False, "duplicate")


def test_chain_rules_follow_enabled_and_config(clock):
    off = BotLoopGuard(settings=lambda: BotLoopGuardSettings(enabled=False, max_hops=1), clock=clock.now)
    assert all(off.chain_admit(CHAT, PEER, "same", is_bot=True)[0] for _ in range(5))
    cfg = settings_from_config({"gateway": {"bot_loop_guard": {"max_hops": 7, "bot_replies_expect_reply": "yes"}}})
    assert (cfg.max_hops, cfg.bot_replies_expect_reply) == (7, True)
    assert settings_from_config({}).max_hops == 4 and settings_from_config({}).bot_replies_expect_reply is False


@pytest.mark.asyncio
async def test_ingress_drops_the_bot_reply_and_still_admits_the_human(monkeypatch, clock):
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._bot_loop_guard = _guard(clock)
    runner._adapter_profile_for_source = lambda source: ""

    def _event(user, text, is_bot):
        source = SessionSource(platform=Platform.MATTERMOST, chat_id=CHAT, chat_type="dm", user_id=user, is_bot=is_bot)
        return MessageEvent(text=text, message_id=text, source=source)

    from gateway.session_ingress import executing_admission

    first = _event(PEER, "question", True)
    assert await runner._hm_admit_event(first) is not None
    runner._note_bot_reply(first.source)  # the turn's reply was delivered
    # The session authority executes the admitted row through the same handler: not a second hop.
    token = executing_admission.set(True)
    try:
        assert await runner._hm_admit_event(first) is not None
    finally:
        executing_admission.reset(token)
    assert await runner._hm_admit_event(_event(PEER, "answer to your answer", True)) is None
    assert await runner._hm_admit_event(_event(HUMAN, "hi", False)) is not None
    assert await runner._hm_admit_event(_event(PEER, "new question", True)) is not None


def test_bot_users_marks_a_peer_on_a_platform_without_a_bot_flag():
    config = SimpleNamespace(extra={"bot_users": ["peer_bot", " other "]})
    assert configured_bot_user(config, "peer_bot") and configured_bot_user(config, "other")
    assert not configured_bot_user(config, HUMAN)
    assert configured_bot_user(SimpleNamespace(extra={"bot_users": "a,peer_bot"}), "peer_bot")
    assert not configured_bot_user(SimpleNamespace(extra={}), "peer_bot")


def test_repeated_refused_asks_do_not_refresh_idle_clock(clock):
    """Refused asks (duplicate, reply_to_reply, max_hops) must not update chain.updated."""
    guard = _guard(clock, cooldown_seconds=600)
    assert guard.chain_admit(CHAT, PEER, "initial ask", is_bot=True) == (True, "ok")
    guard.note_reply(CHAT, PEER)

    # 200s later, peer sends a refused reply_to_reply
    clock.advance(200)
    assert guard.chain_admit(CHAT, PEER, "thanks", is_bot=True) == (False, "reply_to_reply")

    # Another 200s later (400s total), another refused message
    clock.advance(200)
    assert guard.chain_admit(CHAT, PEER, "still thanks", is_bot=True) == (False, "reply_to_reply")

    # 201s later (601s since initial ask, but only 201s since last refused ask):
    # The chain should have expired because refused asks did not advance chain.updated.
    clock.advance(201)
    assert guard.chain_admit(CHAT, PEER, "new ask after expiry", is_bot=True) == (True, "ok")


def test_delivered_reply_tracking_and_reference_inversion_on_capable_platforms(clock):
    """On reference-capable platforms with admit_unreferenced_asks, unreferenced top-level asks
    are admitted as new asks, while explicit replies to delivered replies are dropped (#151224)."""
    guard = _guard(clock, admit_unreferenced_asks=True)
    # Hop 1: Peer sends top-level ask 1
    assert guard.chain_admit(
        CHAT, PEER, "task 1", is_bot=True, supports_message_replies=True, reply_to_message_id=None
    ) == (True, "ok")
    guard.note_reply(CHAT, PEER, message_id="reply_msg_101")

    # An explicit reply quoting our reply is dropped
    assert guard.chain_admit(
        CHAT, PEER, "thank you!", is_bot=True, supports_message_replies=True, reply_to_message_id="reply_msg_101"
    ) == (False, "reply_to_reply")

    # A new top-level ask without reply reference is admitted as a new ask (Hop 2)
    assert guard.chain_admit(
        CHAT, PEER, "task 2 (correction)", is_bot=True, supports_message_replies=True, reply_to_message_id=None
    ) == (True, "ok")

    # A message replying to someone else's message is also admitted as a new ask (Hop 3)
    assert guard.chain_admit(
        CHAT, PEER, "task 3 (re: other)", is_bot=True, supports_message_replies=True, reply_to_message_id="other_msg_999"
    ) == (True, "ok")


def test_unsupported_or_disabled_platforms_keep_conservative_drop(clock):
    """Platforms lacking per-message replies (Slack) or with admit_unreferenced_asks=False keep conservative drop."""
    # When admit_unreferenced_asks is False (default), even reference-capable platforms drop top-level asks
    default_guard = _guard(clock, admit_unreferenced_asks=False)
    assert default_guard.chain_admit(CHAT, PEER, "q1", is_bot=True, supports_message_replies=True) == (True, "ok")
    default_guard.note_reply(CHAT, PEER, message_id="rep1")
    assert default_guard.chain_admit(
        CHAT, PEER, "q2", is_bot=True, supports_message_replies=True, reply_to_message_id=None
    ) == (False, "reply_to_reply")

    # When supports_message_replies is False (e.g. Slack), conservative drop applies regardless of setting
    slack_guard = _guard(clock, admit_unreferenced_asks=True)
    assert slack_guard.chain_admit(CHAT, PEER, "q1", is_bot=True, supports_message_replies=False) == (True, "ok")
    slack_guard.note_reply(CHAT, PEER, message_id="rep1")
    assert slack_guard.chain_admit(
        CHAT, PEER, "q2", is_bot=True, supports_message_replies=False, reply_to_message_id=None
    ) == (False, "reply_to_reply")


def test_alternating_fresh_worded_top_level_asks_terminate_at_max_hops(clock):
    """Freshly worded top-level asks under admit_unreferenced_asks cannot loop infinitely; they terminate at max_hops."""
    guard = _guard(clock, max_hops=4, admit_unreferenced_asks=True)
    for n in range(1, 5):
        assert guard.chain_admit(
            CHAT, PEER, f"fresh ask {n}", is_bot=True, supports_message_replies=True, reply_to_message_id=None
        ) == (True, "ok")
        guard.note_reply(CHAT, PEER, message_id=f"rep_{n}")

    # Hop 5 exceeds max_hops (4)
    assert guard.chain_admit(
        CHAT, PEER, "fresh ask 5", is_bot=True, supports_message_replies=True, reply_to_message_id=None
    ) == (False, "max_hops")
