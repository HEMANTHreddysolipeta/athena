from athena.rag import (
    NOT_FOUND_MESSAGE,
    answer_question,
    build_prompt,
    format_answer,
    parse_citations,
    select_context,
)
from athena.vector_store import RetrievedChunk


def chunk(text, filename="doc.pdf", page=1, distance=0.1):
    return RetrievedChunk(id=text[:8], text=text,
                          metadata={"filename": filename, "page": page}, distance=distance)


class RecordingLLM:
    """Fake LLM: returns a canned reply and remembers the prompt it got."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def seed(store):
    store.upsert_chunks(
        ["a", "b", "c"],
        ["Technical SEO covers site speed and crawlability.",
         "Keyword research finds terms people search for.",
         "Backlinks are links from other websites."],
        [{"filename": "seo.pdf", "filepath": "x", "page": p} for p in (1, 2, 3)],
    )


def test_context_respects_character_budget():
    chunks = [chunk("x" * 400), chunk("y" * 400), chunk("z" * 400)]
    assert len(select_context(chunks, max_chars=900)) == 2


def test_first_chunk_is_kept_even_if_over_budget():
    assert len(select_context([chunk("x" * 5000)], max_chars=100)) == 1


def test_prompt_numbers_sources_with_filename_and_page():
    sources = select_context([chunk("alpha", "a.pdf", 3), chunk("beta", "b.docx", None)], 10_000)
    prompt = build_prompt("what?", sources)
    assert "[1] a.pdf, page 3\nalpha" in prompt
    assert "[2] b.docx\nbeta" in prompt


def test_answer_cites_only_referenced_sources(store):
    seed(store)
    llm = RecordingLLM("Technical SEO covers site speed [1].")

    result = answer_question("what is technical seo", store=store, generate=llm, top_k=3)

    assert result.found
    assert [s.number for s in result.cited] == [1]
    assert len(result.context) == 3
    assert "QUESTION: what is technical seo" in llm.prompts[0]
    assert "seo.pdf, page" in format_answer(result)


def test_comma_separated_citations_are_parsed():
    """Regression: gemma3 writes [1, 2, 4]; the first parser missed those."""
    sources = select_context([chunk(f"t{i}", f"f{i}.pdf") for i in range(4)], 10_000)
    cited = parse_citations("Parts: panel, battery [1, 3]. Wires [4].", sources)
    assert [s.number for s in cited] == [1, 3, 4]


def test_question_is_the_only_query(store):
    """Regression: old code added 5 hardcoded resume queries to every question."""
    seed(store)
    llm = RecordingLLM("ok [1]")
    answer_question("backlinks", store=store, generate=llm, top_k=1)
    assert "Backlinks are links" in llm.prompts[0]
    assert "resume" not in llm.prompts[0].lower()


def test_not_found_is_reported_without_an_ungrounded_retry(store):
    seed(store)
    llm = RecordingLLM("NOT_FOUND")

    result = answer_question("capital of France?", store=store, generate=llm)

    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert len(llm.prompts) == 1  # no second, context-free LLM call


def test_empty_store_skips_the_llm(store):
    llm = RecordingLLM("should not be called")
    result = answer_question("anything", store=store, generate=llm)
    assert not result.found and llm.prompts == []


# ---- phase 2: abstention, follow-ups, streaming, grounding -------------

class FixedReranker:
    def __init__(self, score):
        self.value = score

    def score(self, query, texts):
        return [self.value] * len(texts)


def test_low_relevance_abstains_without_calling_the_llm(store):
    seed(store)
    llm = RecordingLLM("should not be called")
    result = answer_question("bitcoin price", store=store, generate=llm, rerank=True,
                             reranker=FixedReranker(-9.0))
    assert not result.found and result.abstained_early
    assert llm.prompts == []


def test_confident_retrieval_goes_to_the_llm(store):
    seed(store)
    llm = RecordingLLM("Backlinks are links from other websites [1].")
    result = answer_question("backlinks", store=store, generate=llm, rerank=True,
                             reranker=FixedReranker(3.0))
    assert result.found and not result.abstained_early
    assert result.context[0].relevance > 0.9


def test_grounding_flags_invented_numbers_in_the_answer(store):
    seed(store)
    result = answer_question("technical seo", store=store, top_k=1,
                             generate=RecordingLLM("Technical SEO has 7 pillars [1]."))
    assert result.found
    assert result.grounding.unsupported == ["7"]
    assert "Not found in the cited sources: 7" in format_answer(result)


def test_follow_up_is_rewritten_using_history(store):
    seed(store)

    class TwoStepLLM:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            if "STANDALONE QUESTION" in prompt:
                return "What are backlinks?"
            return "Links from other websites [1]."

    llm = TwoStepLLM()
    result = answer_question("what are they?", store=store, generate=llm,
                             history=[("Tell me about backlinks", "Backlinks are ...")])
    assert result.standalone_question == "What are backlinks?"
    assert "QUESTION: What are backlinks?" in llm.prompts[1]


def test_standalone_question_is_not_rewritten(store):
    seed(store)
    llm = RecordingLLM("Keyword research finds terms [1].")
    answer_question("What is keyword research used for in SEO?", store=store, generate=llm,
                    history=[("hi", "hello")])
    assert len(llm.prompts) == 1  # no rewrite call


def test_stream_emits_sources_then_tokens_then_done(store):
    from athena.rag import stream_answer

    seed(store)

    def fake_stream(prompt):
        yield from ["Backlinks ", "are links ", "[1]."]

    events = list(stream_answer("backlinks", store=store, generate_stream=fake_stream))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Backlinks are links [1]."


def test_stream_never_shows_the_not_found_token(store):
    from athena.rag import stream_answer

    seed(store)
    events = list(stream_answer("x", store=store, generate_stream=lambda p: iter(["NOT_", "FOUND"])))
    assert not [e for e in events if e["type"] == "token"]
    assert events[-1]["result"].answer == NOT_FOUND_MESSAGE


def test_stream_answer_starting_with_n_is_not_truncated(store):
    """Regression: holding back a possible NOT_FOUND must not drop text."""
    from athena.rag import stream_answer

    seed(store)
    events = list(stream_answer("x", store=store, generate_stream=lambda p: iter(["N", "o, links [1]."])))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "No, links [1]."
