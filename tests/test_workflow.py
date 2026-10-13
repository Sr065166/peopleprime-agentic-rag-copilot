from unittest.mock import MagicMock, patch
import pytest
from app.rag.workflow import (
    run_agentic_workflow,
    route_question,
    retrieve_kb,
    grade_kb,
    search_web,
    grade_web,
    rewrite_query,
    generate_from_kb,
    generate_from_web,
    direct_answer,
    insufficient_evidence,
)
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import RetrievedChunk


def make_mock_structured_llm(return_value):
    """Helper to mock an LLM with structured output."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = return_value
    return mock_llm


# ============================================================================
# TEST A: PRIVATE KB SUCCESS
# ============================================================================
def test_workflow_private_kb_success():
    """
    Question routes to KB -> Pinecone returns relevant chunks ->
    KB grader returns GOOD -> Grounded answer generated from private KB ->
    Tavily is NOT called -> Citations preserved.
    """
    mock_llm = MagicMock()

    # Router returns "kb"
    # KB grader returns "GOOD"
    # Generation returns answer text
    def mock_with_structured(schema, **kwargs):
        mock_invoker = MagicMock()
        if schema == RouteDecision:
            mock_invoker.invoke.return_value = RouteDecision(route="kb", reason="Policy inquiry")
        elif schema == EvidenceGrade:
            mock_invoker.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Found exact policy")
        return mock_invoker

    mock_llm.with_structured_output.side_effect = mock_with_structured
    mock_llm.invoke.return_value = MagicMock(
        content="Employees at PeoplePrime receive 25 days of annual paid leave."
    )

    mock_chunks = [
        RetrievedChunk(
            text="Employees receive 25 days of annual leave.",
            document_name="PeoplePrime_Leave_Policy.pdf",
            document_id="leave_01",
            page_number=3,
            chunk_index=0,
            score=0.91,
            source_type="pdf",
        )
    ]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=mock_chunks), \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow("How many days of annual leave do employees receive?")

        assert final_state["source_used"] == "private_kb"
        assert final_state["kb_grade"] == "GOOD"
        assert "25 days" in final_state["answer"]
        assert len(final_state["citations"]) == 1
        assert final_state["citations"][0]["type"] == "private_kb"
        assert "PeoplePrime_Leave_Policy.pdf (Page 3)" in final_state["citations"][0]["title"]
        # CRITICAL ASSERTION: Tavily must NOT be called on KB success
        mock_tavily.assert_not_called()


# ============================================================================
# TEST B: WEB FALLBACK SUCCESS
# ============================================================================
def test_workflow_web_fallback_success():
    """
    KB evidence is WEAK -> Web fallback called -> Web grader returns GOOD ->
    Answer generated with external disclaimer and real URLs.
    """
    mock_llm = MagicMock()

    def mock_with_structured(schema, **kwargs):
        mock_invoker = MagicMock()
        if schema == RouteDecision:
            mock_invoker.invoke.return_value = RouteDecision(route="kb", reason="External law query")
        elif schema == EvidenceGrade:
            # First call is KB grade (WEAK), second call is web grade (GOOD)
            pass
        return mock_invoker

    # Let structured output toggle: KB grade -> WEAK, Web grade -> GOOD
    kb_grader_mock = MagicMock()
    kb_grader_mock.invoke.return_value = EvidenceGrade(grade="WEAK", reason="KB does not cover statutory laws")

    web_grader_mock = MagicMock()
    web_grader_mock.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Web has valid government guidelines")

    router_mock = MagicMock()
    router_mock.invoke.return_value = RouteDecision(route="kb", reason="Policy question")

    def side_effect_structured(schema, **kwargs):
        mock_inst = MagicMock()
        if schema == RouteDecision:
            return router_mock
        elif schema == EvidenceGrade:
            # We differentiate by prompt or sequential call
            return mock_inst
        return mock_inst

    # We patch grade_kb and grade_web directly to verify node transitions
    mock_web_results = [
        {
            "title": "National Public Holidays 2026",
            "url": "https://gov.in/holidays/2026",
            "content": "There are 14 gazetted public holidays in 2026.",
        }
    ]

    mock_llm.invoke.return_value = MagicMock(
        content="Based on external public records, there are 14 gazetted holidays. Note: Please verify with HR."
    )

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=[]), \
         patch("app.rag.workflow.execute_tavily_search", return_value=mock_web_results) as mock_tavily:

        # Structured mocks
        mock_llm.with_structured_output.side_effect = [
            router_mock,      # route_question
            kb_grader_mock,   # grade_kb
            web_grader_mock,  # grade_web
        ]

        final_state = run_agentic_workflow("What are the official national public holidays for 2026?")

        assert final_state["source_used"] == "web_search"
        assert final_state["kb_grade"] == "WEAK"
        assert final_state["web_grade"] == "GOOD"
        assert len(final_state["citations"]) == 1
        assert final_state["citations"][0]["url"] == "https://gov.in/holidays/2026"
        assert final_state["citations"][0]["type"] == "web"
        assert mock_tavily.called


# ============================================================================
# TEST C: QUERY REWRITE & BOUNDED RETRY SUCCESS
# ============================================================================
def test_workflow_query_rewrite_retry_success():
    """
    KB weak -> Web search 1 weak -> Query rewritten once (retry_count=1) ->
    Web search 2 returns GOOD -> Grounded web answer.
    """
    mock_llm = MagicMock()

    invoker_router = MagicMock()
    invoker_router.invoke.return_value = RouteDecision(route="kb", reason="Vague policy question")

    invoker_web_1 = MagicMock()
    invoker_web_1.invoke.return_value = EvidenceGrade(grade="WEAK", reason="Web search 1 irrelevant")

    invoker_web_2 = MagicMock()
    invoker_web_2.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Web search 2 found specific labor rule")

    # In this test, retrieve_relevant_chunks returns [], so grade_kb short-circuits
    # to WEAK without invoking the LLM. Thus, with_structured_output is called 3 times:
    # 1) route_question, 2) grade_web (1st), 3) grade_web (retry)
    mock_llm.with_structured_output.side_effect = [
        invoker_router,
        invoker_web_1,
        invoker_web_2,
    ]

    # LLM invoke for rewrite and final generation
    rewrite_resp = MagicMock(content="standard probation period statutory employment guidelines")
    gen_resp = MagicMock(content="Standard statutory probation is 3 to 6 months. External source.")
    mock_llm.invoke.side_effect = [rewrite_resp, gen_resp]

    # Web search returns empty or poor first, then valid second
    tavily_results_1 = [{"title": "Vague Article", "url": "https://example.com/vague", "content": "general stuff"}]
    tavily_results_2 = [{"title": "Probation Law", "url": "https://gov.in/probation", "content": "Statutory probation standard is 3 months."}]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=[]), \
         patch("app.rag.workflow.execute_tavily_search", side_effect=[tavily_results_1, tavily_results_2]) as mock_tavily:

        final_state = run_agentic_workflow("How long can probation last?")

        assert final_state["retry_count"] == 1
        assert final_state["source_used"] == "web_search"
        assert final_state["web_grade"] == "GOOD"
        assert mock_tavily.call_count == 2
        assert "Probation Law" in final_state["citations"][0]["title"]


# ============================================================================
# TEST D: INSUFFICIENT EVIDENCE FALLBACK
# ============================================================================
def test_workflow_insufficient_evidence_fallback():
    """
    KB weak -> Web attempt 1 weak -> Rewrite (retry=1) -> Web attempt 2 weak ->
    Terminates at insufficient_evidence node. No hallucinated answer.
    """
    mock_llm = MagicMock()

    router_mock = MagicMock()
    router_mock.invoke.return_value = RouteDecision(route="kb", reason="Obscure question")

    kb_grader_weak = MagicMock()
    kb_grader_weak.invoke.return_value = EvidenceGrade(grade="WEAK", reason="KB has nothing")

    web_grader_weak_1 = MagicMock()
    web_grader_weak_1.invoke.return_value = EvidenceGrade(grade="WEAK", reason="Web 1 has nothing")

    web_grader_weak_2 = MagicMock()
    web_grader_weak_2.invoke.return_value = EvidenceGrade(grade="WEAK", reason="Web 2 still has nothing")

    mock_llm.with_structured_output.side_effect = [
        router_mock,
        kb_grader_weak,
        web_grader_weak_1,
        web_grader_weak_2,
    ]

    mock_llm.invoke.return_value = MagicMock(content="quantum computing corporate regulations")

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=[]), \
         patch("app.rag.workflow.execute_tavily_search", return_value=[]) as mock_tavily:

        final_state = run_agentic_workflow("What is the company policy on Martian travel insurance?")

        assert final_state["source_used"] == "insufficient_evidence"
        assert "could not find enough reliable evidence" in final_state["answer"]
        assert final_state["retry_count"] == 1
        assert mock_tavily.call_count == 2  # exactly 1 original + 1 retry max


# ============================================================================
# TEST E: DIRECT GREETING ROUTE
# ============================================================================
def test_workflow_direct_greeting_route():
    """
    Greeting -> Router chooses 'direct' -> Direct answer -> Pinecone and Tavily not called.
    """
    with patch("app.rag.workflow.get_llm") as mock_get_llm, \
         patch("app.rag.workflow.retrieve_relevant_chunks") as mock_pinecone, \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow("Good morning!")

        assert final_state["source_used"] == "direct"
        assert "Hello!" in final_state["answer"]
        mock_get_llm.assert_not_called()
        mock_pinecone.assert_not_called()
        mock_tavily.assert_not_called()


@pytest.mark.parametrize("greeting_input", ["hello", "hello!", "Hi", "hey", "good afternoon", "how are you"])
def test_greetings_zero_external_calls(greeting_input):
    """
    Verify that common greetings like 'hello', 'Hi', 'how are you' receive a friendly response
    without calling Groq LLM, Pinecone, or Tavily.
    """
    with patch("app.rag.workflow.get_llm") as mock_get_llm, \
         patch("app.rag.workflow.retrieve_relevant_chunks") as mock_pinecone, \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow(greeting_input)

        assert final_state["source_used"] == "direct"
        assert "PeoplePrime" in final_state["answer"]
        assert len(final_state["citations"]) == 0
        assert any("DIRECT" in t for t in final_state["execution_trace"])
        mock_get_llm.assert_not_called()
        mock_pinecone.assert_not_called()
        mock_tavily.assert_not_called()


def test_hr_question_with_greeting_still_routes_to_kb():
    """
    Verify questions containing greetings along with policy questions (e.g. 'Hello, what is the leave policy?')
    do NOT get captured by simple greeting check, and continue to use Pinecone KB retrieval.
    """
    mock_llm = MagicMock()
    router_mock = MagicMock()
    router_mock.invoke.return_value = RouteDecision(route="kb", reason="Leave policy question")

    grader_mock = MagicMock()
    grader_mock.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Found policy")

    mock_llm.with_structured_output.side_effect = [router_mock, grader_mock]
    mock_llm.invoke.return_value = MagicMock(content="Employees get 25 days leave.")

    mock_chunks = [
        RetrievedChunk(
            text="25 days annual leave",
            document_name="leave.pdf",
            document_id="id1",
            page_number=1,
            chunk_index=0,
            score=0.9,
            source_type="pdf",
        )
    ]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=mock_chunks) as mock_pinecone, \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow("Hello, what is our annual leave policy?")

        assert final_state["source_used"] == "private_kb"
        assert mock_pinecone.called
        mock_tavily.assert_not_called()


# ============================================================================
# TEST F: ERROR HANDLING & FAIL SAFE
# ============================================================================
def test_workflow_malformed_grader_fails_safely():
    """
    If LLM grader throws or returns unparsable output, it safely defaults to WEAK
    rather than crashing or spuriously approving bad evidence.
    """
    state: AgentState = {
        "question": "Sample question",
        "search_query": "Sample question",
        "route": "kb",
        "kb_docs": [{"document_name": "doc.pdf", "text": "random text"}],
        "kb_grade": None,
        "web_results": [],
        "web_grade": None,
        "answer": "",
        "citations": [],
        "source_used": "",
        "retry_count": 0,
        "execution_trace": [],
        "error_status": None,
    }

    mock_llm = MagicMock()
    # Structured output raises Exception (simulating bad JSON / API failure)
    mock_llm.with_structured_output.side_effect = RuntimeError("Bad Groq JSON output")

    with patch("app.rag.workflow.get_llm", return_value=mock_llm):
        out = grade_kb(state)
        assert out["kb_grade"] == "WEAK"


# ============================================================================
# TEST G: MULTI-TOPIC COMPOUND QUERY DETECTION & RETRIEVAL
# ============================================================================
def test_extract_compound_topics():
    """Verify that multi-topic compound questions are detected across policy domains."""
    from app.rag.workflow import extract_compound_topics

    q1 = "Summarize the annual leave, sick leave, and remote-work allowances in the uploaded demonstration HR policy."
    topics1 = extract_compound_topics(q1)
    assert "annual leave" in topics1
    assert "sick leave" in topics1
    assert "remote work" in topics1

    q2 = "How many days of annual leave do employees receive?"
    topics2 = extract_compound_topics(q2)
    assert topics2 == ["annual leave"]


def test_retrieve_kb_multi_topic_retrieval_and_deduplication():
    """Verify retrieve_kb executes multi-topic sub-retrievals and deduplicates chunks."""
    question = "Summarize the annual leave, sick leave, and remote-work allowances in the uploaded policy."
    state: AgentState = {
        "question": question,
        "search_query": question,
        "route": "kb",
        "kb_docs": [],
        "kb_grade": None,
        "web_results": [],
        "web_grade": None,
        "answer": "",
        "citations": [],
        "source_used": "",
        "retry_count": 0,
        "execution_trace": [],
        "error_status": None,
    }

    c_leave = RetrievedChunk(
        text="25 days annual leave.",
        document_name="HR_Policy.pdf",
        document_id="doc1",
        page_number=2,
        chunk_index=1,
        score=0.88,
        source_type="pdf",
    )
    c_sick = RetrievedChunk(
        text="10 days sick leave.",
        document_name="HR_Policy.pdf",
        document_id="doc1",
        page_number=2,
        chunk_index=2,
        score=0.85,
        source_type="pdf",
    )
    c_remote = RetrievedChunk(
        text="Remote work up to 3 days per week.",
        document_name="HR_Policy.pdf",
        document_id="doc1",
        page_number=2,
        chunk_index=3,
        score=0.82,
        source_type="pdf",
    )
    # Duplicate with lower score
    c_leave_dup = RetrievedChunk(
        text="25 days annual leave.",
        document_name="HR_Policy.pdf",
        document_id="doc1",
        page_number=2,
        chunk_index=1,
        score=0.75,
        source_type="pdf",
    )

    def mock_retrieve_side_effect(query, top_k=None):
        if "annual leave" in query:
            return [c_leave, c_leave_dup]
        elif "sick leave" in query:
            return [c_sick]
        elif "remote work" in query:
            return [c_remote]
        return [c_leave]

    with patch("app.rag.workflow.retrieve_relevant_chunks", side_effect=mock_retrieve_side_effect) as mock_ret:
        result = retrieve_kb(state)
        docs = result["kb_docs"]
        assert len(docs) == 3  # Duplicate c_leave_dup should be deduplicated
        texts = [d["text"] for d in docs]
        assert "25 days annual leave." in texts
        assert "10 days sick leave." in texts
        assert "Remote work up to 3 days per week." in texts
        # Metadata preserved
        assert all(d["document_name"] == "HR_Policy.pdf" for d in docs)
        assert all(d["page_number"] == 2 for d in docs)


def test_workflow_multi_topic_policy_question_end_to_end():
    """Verify complete end-to-end workflow handles multi-topic questions with private KB table."""
    mock_llm = MagicMock()

    def mock_with_structured(schema, **kwargs):
        invoker = MagicMock()
        if schema == RouteDecision:
            invoker.invoke.return_value = RouteDecision(route="kb", reason="Multi-topic HR inquiry")
        elif schema == EvidenceGrade:
            invoker.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Found leave, sick, and remote policies")
        return invoker

    mock_llm.with_structured_output.side_effect = mock_with_structured
    table_answer = (
        "Here is the summary of allowances:\n\n"
        "| Policy Area | Allowance / Entitlement | Source Document |\n"
        "|---|---|---|\n"
        "| Annual Leave | 25 days paid leave | NovaCore_Policy.pdf (Page 2) |\n"
        "| Sick Leave | 10 days paid leave | NovaCore_Policy.pdf (Page 2) |\n"
        "| Remote Work | Up to 3 days per week | NovaCore_Policy.pdf (Page 2) |"
    )
    mock_llm.invoke.return_value = MagicMock(content=table_answer)

    chunks = [
        RetrievedChunk("Annual leave: 25 days", "NovaCore_Policy.pdf", "doc1", 2, 1, 0.9, "pdf"),
        RetrievedChunk("Sick leave: 10 days", "NovaCore_Policy.pdf", "doc1", 2, 2, 0.88, "pdf"),
        RetrievedChunk("Remote work: 3 days/week", "NovaCore_Policy.pdf", "doc1", 2, 3, 0.85, "pdf"),
    ]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=chunks), \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow(
            "Summarize the annual leave, sick leave, and remote-work allowances in the uploaded demonstration HR policy. Use a table and cite the source."
        )

        assert final_state["source_used"] == "private_kb"
        assert final_state["kb_grade"] == "GOOD"
        assert "| Annual Leave |" in final_state["answer"]
        assert len(final_state["citations"]) >= 1
        assert any(c["title"] == "NovaCore_Policy.pdf (Page 2)" for c in final_state["citations"])
        mock_tavily.assert_not_called()


def test_workflow_missing_topic_explicit_unavailable():
    """Verify that an unsupported topic is marked unavailable rather than rejected or hallucinated."""
    mock_llm = MagicMock()

    def mock_with_structured(schema, **kwargs):
        invoker = MagicMock()
        if schema == RouteDecision:
            invoker.invoke.return_value = RouteDecision(route="kb", reason="Policy inquiry with unlisted topic")
        elif schema == EvidenceGrade:
            invoker.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Supported leave present; missing topic marked unavailable")
        return invoker

    mock_llm.with_structured_output.side_effect = mock_with_structured
    mock_answer = (
        "| Policy Area | Entitlement | Source |\n"
        "|---|---|---|\n"
        "| Annual Leave | 25 days | HR_Policy.pdf (Page 2) |\n"
        "| Sabbatical Leave | Not specified in the uploaded document | N/A |"
    )
    mock_llm.invoke.return_value = MagicMock(content=mock_answer)

    chunks = [
        RetrievedChunk("Annual leave: 25 days", "HR_Policy.pdf", "doc1", 2, 1, 0.9, "pdf"),
    ]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=chunks), \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow(
            "Summarize the annual leave and sabbatical leave allowances in the policy."
        )

        assert final_state["source_used"] == "private_kb"
        assert "Not specified in the uploaded document" in final_state["answer"]
        mock_tavily.assert_not_called()


def test_workflow_execution_trace_includes_timing_instrumentation():
    """Verify that execution trace logs include elapsed_ms timing brackets."""
    mock_llm = MagicMock()

    def mock_with_structured(schema, **kwargs):
        invoker = MagicMock()
        if schema == RouteDecision:
            invoker.invoke.return_value = RouteDecision(route="kb", reason="Policy query")
        elif schema == EvidenceGrade:
            invoker.invoke.return_value = EvidenceGrade(grade="GOOD", reason="Found evidence")
        return invoker

    mock_llm.with_structured_output.side_effect = mock_with_structured
    mock_llm.invoke.return_value = MagicMock(content="Employees receive 20 days annual leave.")

    chunks = [
        RetrievedChunk("Annual leave: 20 days", "HR_Policy.pdf", "doc1", 1, 0, 0.95, "pdf"),
    ]

    with patch("app.rag.workflow.get_llm", return_value=mock_llm), \
         patch("app.rag.workflow.retrieve_relevant_chunks", return_value=chunks):

        final_state = run_agentic_workflow("What is the annual leave policy?")

        trace = final_state.get("execution_trace", [])
        assert len(trace) >= 4
        # Verify timing brackets [xx.xms] in trace entries
        assert any("Intent Classifier [" in entry and "ms]" in entry for entry in trace)
        assert any("Private Pinecone KB Retrieval [" in entry and "ms]" in entry for entry in trace)
        assert any("KB Evidence Grade [" in entry and "ms]" in entry for entry in trace)
        assert any("Answer Generation [" in entry and "ms]" in entry for entry in trace)


def test_web_search_expands_results_for_calendar_holiday_queries():
    """Verify execute_tavily_search requests >=6 results when calendar/holiday terms appear."""
    from app.services.web_search import execute_tavily_search

    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "Republic Day", "url": "https://example.com/1", "content": "Jan 26 Gazetted"}
        ]
    }

    # Standard query uses default max_results=4
    execute_tavily_search("employment law update", max_results=4, tavily_client=mock_client)
    _, kwargs = mock_client.search.call_args
    assert kwargs["max_results"] == 4

    # Holiday calendar query expands to at least 6
    execute_tavily_search("public holidays 2026 gazetted calendar", max_results=4, tavily_client=mock_client)
    _, kwargs2 = mock_client.search.call_args
    assert kwargs2["max_results"] >= 6


def test_groq_llm_configured_with_timeouts_and_retries():
    """Verify ChatGroq client initializes with bounded timeout and max retries."""
    import app.rag.workflow as wf

    with patch("app.rag.workflow.ChatGroq") as mock_chat_groq_cls, \
         patch("app.rag.workflow.get_settings") as mock_settings:
        settings_mock = MagicMock()
        settings_mock.groq_api_key = "gsk_test_key"
        settings_mock.groq_model = "test-model"
        mock_settings.return_value = settings_mock

        # Reset singleton for testing
        wf._groq_llm = None
        client = wf.get_llm()

        mock_chat_groq_cls.assert_called_once_with(
            model="test-model",
            temperature=0.0,
            api_key="gsk_test_key",
            request_timeout=30.0,
            max_retries=2,
        )
        assert client == mock_chat_groq_cls.return_value


@pytest.mark.parametrize("query", ["Thank you", "thanks a lot", "okay got it", "understood"])
def test_conversational_thanks_and_acknowledgements_zero_external_calls(query):
    """Verify thanks and conversational acknowledgements route directly with zero external API calls."""
    with patch("app.rag.workflow.get_llm") as mock_get_llm, \
         patch("app.rag.workflow.retrieve_relevant_chunks") as mock_retrieve, \
         patch("app.rag.workflow.execute_tavily_search") as mock_tavily:

        final_state = run_agentic_workflow(query)

        assert final_state["route"] == "direct"
        assert final_state["source_used"] == "direct"
        assert len(final_state["citations"]) == 0
        mock_get_llm.assert_not_called()
        mock_retrieve.assert_not_called()
        mock_tavily.assert_not_called()


def test_pinecone_index_caching_across_calls():
    """Verify get_or_verify_index caches verified index to avoid repeated list_indexes/describe_index."""
    import app.rag.vectorstore as vs

    mock_pc = MagicMock()
    mock_idx_item = MagicMock()
    mock_idx_item.name = "peopleprime-hr-kb-384"
    mock_pc.list_indexes.return_value = [mock_idx_item]
    mock_desc = MagicMock()
    mock_desc.dimension = 384
    mock_pc.describe_index.return_value = mock_desc

    with patch("app.rag.vectorstore.get_pinecone_client", return_value=mock_pc):
        vs._verified_index = None  # Reset cache

        idx1 = vs.get_or_verify_index()
        idx2 = vs.get_or_verify_index()

        assert idx1 == idx2
        # list_indexes and describe_index should be called exactly once
        assert mock_pc.list_indexes.call_count == 1
        assert mock_pc.describe_index.call_count == 1
