from backend.rag_pipeline import HybridGeospatialRAGPipeline


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = responses

    def create(self, model, messages, temperature, max_tokens, timeout, extra_body):
        return _FakeResponse(self._responses[model])


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)


def _build_pipeline(responses, available_models):
    pipeline = HybridGeospatialRAGPipeline.__new__(HybridGeospatialRAGPipeline)
    pipeline.llm_model = available_models[0]
    pipeline.available_models = list(available_models)
    pipeline.fallback_model_preferences = list(available_models[1:])
    pipeline.llm_client = _FakeClient(responses)
    pipeline.llm_base_url = "http://127.0.0.1:11434/v1"
    pipeline._list_available_models = lambda: list(available_models)
    return pipeline


def test_choose_default_model_prefers_qwen25_before_qwen3() -> None:
    pipeline = HybridGeospatialRAGPipeline.__new__(HybridGeospatialRAGPipeline)
    pipeline.available_models = ["qwen3:8b", "qwen2.5:14b", "llama3.1:8b"]

    assert pipeline._choose_default_model() == "qwen2.5:14b"


def test_generate_answer_retries_empty_response_with_next_model() -> None:
    pipeline = _build_pipeline(
        responses={
            "qwen3:4b": "",
            "llama3.1:8b": "Direct answer with support [C1]\nEvidence Gaps: limited.",
        },
        available_models=["qwen3:4b", "llama3.1:8b"],
    )
    retrieved_chunks = [
        {
            "chunk_id": "C1",
            "engine": "vector",
            "company": "Alpha",
            "chunk_type": "company_profile",
            "score": 0.8,
            "text": "Alpha is a supplier.",
            "meta": {},
        }
    ]

    answer = pipeline._generate_answer_with_llm(
        question="What is Alpha?",
        context="Question: What is Alpha?\nRetrieved Chunks:\n[C1] Alpha is a supplier.",
        retrieved_chunks=retrieved_chunks,
        preferred_companies=["Alpha"],
    )

    assert answer.startswith("Direct answer with support [C1]")
    assert pipeline.llm_model == "llama3.1:8b"


def test_generate_answer_returns_fallback_when_all_models_are_empty() -> None:
    pipeline = _build_pipeline(
        responses={
            "qwen3:4b": "",
            "llama3.1:8b": "",
        },
        available_models=["qwen3:4b", "llama3.1:8b"],
    )
    retrieved_chunks = [
        {
            "chunk_id": "C1",
            "engine": "vector",
            "company": "Alpha",
            "chunk_type": "company_profile",
            "score": 0.8,
            "text": "Alpha is a supplier.",
            "meta": {},
        }
    ]

    answer = pipeline._generate_answer_with_llm(
        question="What is Alpha?",
        context="Question: What is Alpha?\nRetrieved Chunks:\n[C1] Alpha is a supplier.",
        retrieved_chunks=retrieved_chunks,
        preferred_companies=["Alpha"],
    )

    assert "Alpha" in answer
    assert "Evidence Gaps" in answer
