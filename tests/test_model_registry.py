import unittest
from types import SimpleNamespace
from unittest.mock import patch

from model_registry import ModelConfig, send_request


class SendRequestSettingsTests(unittest.TestCase):
    def make_config(self, provider):
        return ModelConfig(
            provider_name=provider,
            model_id="test-model",
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
            average_latency_ms=0,
            quality_tier="medium",
        )

    def test_openai_compatible_providers_receive_settings(self):
        expected_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Explain testing."},
        ]

        for provider in ("openai", "groq", "openrouter"):
            with self.subTest(provider=provider):
                with patch("model_registry.OpenAI") as openai_client:
                    with patch(
                        "model_registry.required_env",
                        return_value="test-key",
                    ):
                        openai_client.return_value.chat.completions.create.return_value = (
                            SimpleNamespace(
                                choices=[
                                    SimpleNamespace(
                                        message=SimpleNamespace(content="answer")
                                    )
                                ],
                                usage=None,
                            )
                        )

                        send_request(
                            "Explain testing.",
                            self.make_config(provider),
                            system_prompt="Be concise.",
                            temperature=0.3,
                        )

                        openai_client.return_value.chat.completions.create.assert_called_once_with(
                            model="test-model",
                            messages=expected_messages,
                            temperature=0.3,
                        )

    def test_gemini_receives_system_instruction_and_temperature(self):
        with patch("model_registry.genai.Client") as gemini_client:
            with patch(
                "model_registry.required_env",
                return_value="test-key",
            ):
                with patch(
                    "model_registry.types.GenerateContentConfig"
                ) as generation_config:
                    gemini_client.return_value.models.generate_content.return_value = (
                        SimpleNamespace(text="answer", usage_metadata=None)
                    )

                    send_request(
                        "Explain testing.",
                        self.make_config("gemini"),
                        system_prompt="Be concise.",
                        temperature=0.3,
                    )

                    gemini_client.return_value.models.generate_content.assert_called_once_with(
                        model="test-model",
                        contents="Explain testing.",
                        config=generation_config.return_value,
                    )
                    generation_config.assert_called_once_with(
                        max_output_tokens=1024,
                        system_instruction="Be concise.",
                        temperature=0.3,
                    )

    def test_ollama_providers_receive_settings(self):
        expected_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Explain testing."},
        ]

        for provider in ("ollama_local", "ollama_cloud"):
            with self.subTest(provider=provider):
                with patch("model_registry.ollama.Client") as ollama_client:
                    with patch(
                        "model_registry.required_env",
                        return_value="test-key",
                    ):
                        ollama_client.return_value.chat.return_value = SimpleNamespace(
                            message=SimpleNamespace(content="answer"),
                            prompt_eval_count=None,
                            eval_count=None,
                        )

                        send_request(
                            "Explain testing.",
                            self.make_config(provider),
                            system_prompt="Be concise.",
                            temperature=0.3,
                        )

                        ollama_client.return_value.chat.assert_called_once_with(
                            model="test-model",
                            messages=expected_messages,
                            options={"temperature": 0.3},
                        )


if __name__ == "__main__":
    unittest.main()