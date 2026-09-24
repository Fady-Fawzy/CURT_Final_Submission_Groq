import os
import unittest
from unittest.mock import patch

from groq_client import MODEL_NAME, create_groq_client


class GroqClientTests(unittest.TestCase):
    def test_missing_api_key_has_a_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                create_groq_client()

    def test_api_key_is_passed_to_the_groq_client_with_retries(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            with patch("groq_client.Groq") as mock_client:
                create_groq_client()

        mock_client.assert_called_once_with(
            api_key="test-key", max_retries=3, timeout=45.0
        )

    def test_model_name_matches_the_enabled_groq_chat_model(self):
        self.assertEqual(MODEL_NAME, "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
