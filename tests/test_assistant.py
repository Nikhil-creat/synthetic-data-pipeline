from fastapi.testclient import TestClient

from assistant import LLMRouter, detect_provider, rank_gemini_models, rank_groq_models


def test_detect_provider():
    assert detect_provider("gsk_abc") == "groq"
    assert detect_provider("AIzaSyX") == "gemini"
    assert detect_provider("sk-other") is None and detect_provider("") is None


def test_groq_ranking_skips_non_chat_models():
    ids = ["whisper-large-v3", "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
           "llama-guard-4-12b", "playai-tts"]
    ranked = rank_groq_models(ids)
    assert ranked == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_gemini_ranking_prefers_flash_and_skips_others():
    gc = ["generateContent"]
    models = [
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.0-flash-lite", "supportedGenerationMethods": gc},
        {"name": "models/gemini-2.5-flash-image-preview", "supportedGenerationMethods": gc},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]
    assert rank_gemini_models(models) == ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro"]


def test_router_detects_provider_even_if_keys_are_swapped():
    r = LLMRouter(groq_key="AIzaFAKE", gemini_key="gsk_FAKE")
    assert set(r.providers) == {"gemini", "groq"}


def test_assistant_falls_back_to_offline_rag(monkeypatch):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from main import app
    with TestClient(app) as c:
        r = c.post("/assistant/chat", json={"message": "which resampler for extreme imbalance?"})
        assert r.status_code == 200 and r.json()["provider"] == "offline-rag"
        assert c.get("/assistant/status").json()["configured"] == []
        assert "sdp_http_requests_total" in c.get("/metrics").text
