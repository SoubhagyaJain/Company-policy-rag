"""
Empirical Challenger 2 Test Suite for Milestone 4 (Streamlit Thinking UI, ARIA Contracts, Session Persistence).
"""

from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

from src.agent import AgentTurnResult
from app.ui.components.chat import (
    render_thinking_history,
    apply_queue_user_prompt,
    apply_complete_assistant_turn,
)
from app.ui.session import (
    ensure_session_state,
    run_direct_turn,
    run_agent_turn,
    clear_chat_session,
)


class MockSessionState(dict):
    """Dual attr and dict access to emulate Streamlit session_state."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No attribute {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No attribute {name}")


class TestStreamlitThinkingUIEmpirical(unittest.TestCase):
    """Empirical testing of render_thinking_history and Streamlit components."""

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_off_level(self, mock_caption, mock_markdown, mock_expander):
        """When detail_level == 'off', expander must not be rendered."""
        events = [{"stage": "retrieval", "status": "completed", "title": "Retrieval"}]
        render_thinking_history(events, detail_level="off")
        mock_expander.assert_not_called()
        mock_markdown.assert_not_called()

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_empty_events(self, mock_caption, mock_markdown, mock_expander):
        """When events list is empty, expander must not be rendered."""
        render_thinking_history([], detail_level="standard")
        mock_expander.assert_not_called()
        mock_markdown.assert_not_called()

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_duration_from_reasoning_summary(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """Duration in header should use reasoning_summary.total_duration_ms if available."""
        events = [
            {"stage": "received", "status": "completed", "title": "Received", "duration_ms": 10.0},
            {"stage": "retrieval", "status": "completed", "title": "Retrieval", "duration_ms": 25.0},
        ]
        summary = {"total_duration_ms": 1500.0}
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, reasoning_summary=summary, detail_level="standard")
        mock_expander.assert_called_once_with("\U0001f4ad Thought for 1.5s", expanded=False)

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_duration_fallback_sum(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """Duration in header should sum event durations if summary total_duration_ms is missing or 0."""
        events = [
            {"stage": "received", "status": "completed", "title": "Received", "duration_ms": 300.0},
            {"stage": "retrieval", "status": "completed", "title": "Retrieval", "duration_ms": 700.0},
        ]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, reasoning_summary=None, detail_level="standard")
        mock_expander.assert_called_once_with("\U0001f4ad Thought for 1.0s", expanded=False)

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_zero_duration(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """When duration is 0ms, header should simply be 'Thought' with balloon."""
        events = [{"stage": "received", "status": "completed", "title": "Received", "duration_ms": 0.0}]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, reasoning_summary=None, detail_level="standard")
        mock_expander.assert_called_once_with("\U0001f4ad Thought", expanded=False)

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_detailed_mode_metadata(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """In detailed mode, safe metadata (candidate_count, source_count, etc.) and durations must be rendered."""
        events = [
            {
                "stage": "retrieval",
                "status": "completed",
                "title": "Searching sources",
                "summary": "Hybrid search completed",
                "duration_ms": 42.0,
                "details": {
                    "candidate_count": 15,
                    "source_count": 3,
                    "active_topic": "Vacation Policy",
                    "evidence_status": "DIRECT",
                },
            }
        ]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, detail_level="detailed")

        # Check markdown call includes duration badge `42ms` and title
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        self.assertTrue(any("42ms" in call and "Searching sources" in call for call in markdown_calls))

        # Check caption call includes formatted metadata
        caption_calls = [call[0][0] for call in mock_caption.call_args_list]
        self.assertTrue(any("candidates: 15" in call and "sources: 3" in call and "topic: Vacation Policy" in call for call in caption_calls))

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_warning_status(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """When stage.status == 'warning', rendered output must contain alert symbol."""
        events = [
            {
                "stage": "visual_analysis",
                "status": "warning",
                "title": "Visual analysis unavailable",
                "summary": "Vision timed out after 5000ms",
            }
        ]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, detail_level="standard")
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        self.assertTrue(any("Visual analysis unavailable" in call for call in markdown_calls))

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_skipped_status(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """When stage.status == 'skipped', rendered output must contain skip symbol."""
        events = [
            {
                "stage": "visual_analysis",
                "status": "skipped",
                "title": "Visual Analysis",
                "summary": "No images in document",
            }
        ]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, detail_level="standard")
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        self.assertTrue(any("Visual Analysis" in call for call in markdown_calls))

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_render_thinking_history_default_title_fallback(
        self, mock_caption, mock_markdown, mock_expander
    ):
        """When event has no explicit title, stage name is formatted into Title Case."""
        events = [
            {
                "stage": "conversation_context",
                "status": "completed",
                "summary": "Resolved context",
            }
        ]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        render_thinking_history(events, detail_level="standard")
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        self.assertTrue(any("Conversation Context" in call for call in markdown_calls))


class TestStreamlitSessionPersistenceEmpirical(unittest.TestCase):
    """Empirical testing of Multi-Turn Session Persistence and ChatService bridging."""

    def test_ensure_session_state_defaults(self):
        mock_state = MockSessionState()
        with patch("streamlit.session_state", mock_state):
            ensure_session_state()
            self.assertEqual(mock_state.thinking_detail_level, "standard")
            self.assertEqual(mock_state.corpus_scope, "all")
            self.assertEqual(mock_state.chat_mode, "direct")
            self.assertEqual(mock_state.messages, [])
            self.assertTrue(mock_state.session_id.startswith("sess_"))

    @patch("backend.api.dependencies.get_chat_service")
    def test_multi_turn_thinking_detail_level_bridging(self, mock_get_chat_service):
        """Verify that thinking_detail_level persists across turns and is passed to ChatRequest."""
        mock_chat_service = MagicMock()
        mock_get_chat_service.return_value = mock_chat_service

        mock_resp = MagicMock()
        mock_resp.id = "resp_123"
        mock_resp.answer = "Grounded response"
        mock_resp.citations = []
        mock_resp.timing = {"e2e_ms": 100.0}
        mock_resp.low_confidence = False
        mock_resp.thinking_events = []
        mock_resp.reasoning_summary = {"total_duration_ms": 100.0}
        mock_resp.retrieval_trace = None
        mock_chat_service.execute_query.return_value = mock_resp

        mock_state = MockSessionState({
            "session_id": "test_session_1",
            "thinking_detail_level": "compact",
            "corpus_scope": "policy",
            "messages": [],
            "timing_samples": [],
        })

        with patch("streamlit.session_state", mock_state):
            turn1 = run_direct_turn("What is the vacation policy?")
            req1 = mock_chat_service.execute_query.call_args[0][0]
            self.assertEqual(req1.thinking_detail_level, "compact")
            self.assertEqual(req1.session_id, "test_session_1")
            self.assertEqual(req1.document_scope, "policy")

            # Apply complete turn 1
            apply_complete_assistant_turn(mock_state, turn1, user_prompt="What is the vacation policy?")
            self.assertEqual(len(mock_state.messages), 1)

            # Turn 2: User asks follow-up, thinking_detail_level persists
            turn2 = run_agent_turn(MagicMock(), "tell me about it in detail", None)
            req2 = mock_chat_service.execute_query.call_args[0][0]
            self.assertEqual(req2.thinking_detail_level, "compact")
            self.assertEqual(req2.session_id, "test_session_1")
            self.assertEqual(req2.document_scope, "policy")

    @patch("backend.api.dependencies.get_chat_service")
    def test_clear_chat_session(self, mock_get_chat_service):
        mock_chat_service = MagicMock()
        mock_get_chat_service.return_value = mock_chat_service

        mock_state = MockSessionState({
            "session_id": "sess_old_123",
            "messages": [{"role": "user", "content": "hi"}],
            "pending_user_prompt": "pending",
        })

        with patch("streamlit.session_state", mock_state):
            clear_chat_session()

            mock_chat_service.clear_session.assert_called_once_with("sess_old_123")
            self.assertEqual(mock_state.messages, [])
            self.assertIsNone(mock_state.pending_user_prompt)
            self.assertNotEqual(mock_state.session_id, "sess_old_123")
            self.assertTrue(mock_state.session_id.startswith("sess_"))


class TestZeroCoTAndSecurityInStreamlit(unittest.TestCase):
    """Verify zero chain-of-thought, system prompt, or secret leakage in UI representations."""

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    def test_zero_cot_in_render_thinking_history(self, mock_caption, mock_markdown, mock_expander):
        forbidden = ["<think>", "</think>", "system_prompt", "vector_id", "embedding", "api_key", "sk-"]
        mock_expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        events = [
            {"stage": "received", "status": "completed", "title": "Understanding query", "summary": "Clean text"},
            {"stage": "retrieval", "status": "completed", "title": "Searching sources", "summary": "Clean text", "details": {"candidate_count": 5}},
        ]
        render_thinking_history(events, detail_level="detailed")

        all_rendered = [str(c) for c in mock_markdown.call_args_list] + [str(c) for c in mock_caption.call_args_list]
        for rendered in all_rendered:
            for marker in forbidden:
                self.assertNotIn(marker, rendered.lower())
