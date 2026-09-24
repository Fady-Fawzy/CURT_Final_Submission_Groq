import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import _run_requested_tool, app, sessions
from database.seed import initialize_database


def tool_call(name, arguments, call_id="call-123"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class BackendEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_database()
        cls.client = TestClient(app)

    def setUp(self):
        sessions.clear()

    def test_get_inventory_returns_current_parts_as_json(self):
        response = self.client.get("/inventory")

        self.assertEqual(response.status_code, 200)
        inventory = response.json()
        self.assertGreaterEqual(len(inventory), 10)
        self.assertEqual(inventory[0]["name"], "Battery")

    @patch("backend.create_groq_client")
    def test_post_chat_executes_requested_tool_then_returns_groq_reply(self, make_client):
        responses = [
            completion(tool_calls=[tool_call("check_stock", {"item_name": "Brake Pads"})]),
            completion("We have 12 Brake Pads."),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=responses)))
        )
        make_client.return_value = client

        with patch("builtins.print") as debug_print:
            response = self.client.post(
                "/chat",
                json={
                    "message": "How many brake pads do we have?",
                    "session_id": "test-session",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": "We have 12 Brake Pads."})
        debug_print.assert_any_call("Groq requested tool: check_stock")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        follow_up_messages = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        self.assertEqual(follow_up_messages[-1]["role"], "tool")
        tool_result = json.loads(follow_up_messages[-1]["content"])["result"]
        self.assertEqual(
            tool_result,
            {
                "found": True,
                "item_name": "Brake Pads",
                "quantity": 12,
                "location": "Mechanical Workshop",
            },
        )

    @patch("backend.create_groq_client")
    def test_post_chat_returns_groq_text_when_no_tool_is_needed(self, make_client):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=completion(
                            "Please ask me about a CURT inventory item."
                        )
                    )
                )
            )
        )
        make_client.return_value = client

        response = self.client.post(
            "/chat", json={"message": "Hello", "session_id": "test-session"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"reply": "Please ask me about a CURT inventory item."}
        )
        client.chat.completions.create.assert_called_once()

    @patch("backend.create_groq_client")
    def test_follow_up_message_receives_earlier_session_messages(self, make_client):
        responses = [
            completion(tool_calls=[tool_call("check_stock", {"item_name": "Brake Pads"}, "call-first")]),
            completion("We have 12 Brake Pads."),
            completion(tool_calls=[tool_call("check_stock", {"item_name": "Brake Pads"}, "call-follow-up")]),
            completion("They are stored in the Mechanical Workshop."),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=responses)))
        )
        make_client.return_value = client

        first_reply = self.client.post(
            "/chat",
            json={"message": "How many brake pads do we have?", "session_id": "chat-a"},
        )
        follow_up_reply = self.client.post(
            "/chat",
            json={"message": "Where are they stored?", "session_id": "chat-a"},
        )

        self.assertEqual(first_reply.json()["reply"], "We have 12 Brake Pads.")
        self.assertEqual(
            follow_up_reply.json()["reply"],
            "They are stored in the Mechanical Workshop.",
        )
        second_request_messages = client.chat.completions.create.call_args_list[2].kwargs[
            "messages"
        ]
        self.assertEqual(
            [(message["role"], message.get("content")) for message in second_request_messages[1:4]],
            [
                ("user", "How many brake pads do we have?"),
                ("assistant", "We have 12 Brake Pads."),
                ("user", "Where are they stored?"),
            ],
        )

    @patch("backend.create_groq_client")
    def test_different_session_ids_do_not_share_conversation_history(self, make_client):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        side_effect=[
                            completion("We have 12 Brake Pads."),
                            completion("Please name an item."),
                        ]
                    )
                )
            )
        )
        make_client.return_value = client

        self.client.post(
            "/chat",
            json={"message": "How many brake pads do we have?", "session_id": "chat-a"},
        )
        self.client.post(
            "/chat",
            json={"message": "Where are they stored?", "session_id": "chat-b"},
        )

        second_request_messages = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        self.assertEqual(second_request_messages[1:], [{"role": "user", "content": "Where are they stored?"}])

    def test_post_chat_requires_message_and_session_id(self):
        response = self.client.post("/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 422)

    @patch("backend.create_groq_client")
    def test_post_chat_rejects_blank_message(self, make_client):
        for message in ("", "   "):
            with self.subTest(message=message):
                response = self.client.post(
                    "/chat", json={"message": message, "session_id": "chat-a"}
                )
                self.assertEqual(response.status_code, 422)
        make_client.assert_not_called()
        self.assertEqual(sessions, {})

    @patch("backend.create_groq_client")
    def test_post_chat_rejects_blank_session_id(self, make_client):
        for session_id in ("", "   "):
            with self.subTest(session_id=session_id):
                response = self.client.post(
                    "/chat", json={"message": "Hello", "session_id": session_id}
                )
                self.assertEqual(response.status_code, 422)
        make_client.assert_not_called()
        self.assertEqual(sessions, {})

    @patch("backend.create_groq_client")
    def test_post_chat_reports_missing_key_without_internal_details(self, make_client):
        make_client.side_effect = RuntimeError(
            "GROQ_API_KEY is missing; diagnostic-token-not-for-response"
        )
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/chat", json={"message": "Hello", "session_id": "chat-a"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.json()["detail"])
        self.assertNotIn("diagnostic-token", response.text)
        self.assertEqual(sessions, {})

    @patch("backend.create_groq_client")
    def test_post_chat_reports_provider_error_without_internal_details(self, make_client):
        make_client.side_effect = RuntimeError("private-provider-diagnostic")
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/chat", json={"message": "Hello", "session_id": "chat-a"})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private-provider-diagnostic", response.text)
        self.assertEqual(sessions, {})

    def test_malformed_tool_argument_returns_error_instead_of_raising(self):
        function_call = SimpleNamespace(
            function=SimpleNamespace(name="check_stock", arguments=json.dumps({"item_name": 123}))
        )

        result = _run_requested_tool(function_call)

        self.assertEqual(result, {"error": "The tool request had invalid arguments."})


if __name__ == "__main__":
    unittest.main()
