from audit_agent.nim_client import NIMClient


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_chat_falls_back_to_next_model(tmp_path, monkeypatch):
    client = NIMClient(api_key="test-key", cache_dir=str(tmp_path), timeout_seconds=1, max_retries=0)
    attempts = []

    def fake_create(model, messages, temperature, max_tokens, extra):
        attempts.append(model)
        if model == "google/gemma-4-31b-it":
            raise TimeoutError("slow model")
        return DummyResponse(
            {
                "choices": [{"message": {"content": '{"verdict": "agree"}'}}],
                "usage": {"total_tokens": 12},
            }
        )

    monkeypatch.setattr(client, "_create_completion", fake_create)

    result = client.chat(
        model="google/gemma-4-31b-it,google/gemma-3n-e2b-it",
        messages=[{"role": "user", "content": "hi"}],
        use_cache=False,
    )

    assert attempts == ["google/gemma-4-31b-it", "google/gemma-3n-e2b-it"]
    assert result["choices"][0]["message"]["content"] == '{"verdict": "agree"}'
    assert client.stats()["disabled_reason"] is None


def test_chat_json_uses_reasoning_content_when_content_is_empty(tmp_path, monkeypatch):
    client = NIMClient(api_key="test-key", cache_dir=str(tmp_path), timeout_seconds=1, max_retries=0)

    def fake_chat(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": '{"verdict": "agree", "reason": "supported"}',
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "chat", fake_chat)

    parsed = client.chat_json(
        model="google/gemma-4-31b-it",
        messages=[{"role": "user", "content": "hi"}],
        use_cache=False,
    )

    assert parsed == {"verdict": "agree", "reason": "supported"}