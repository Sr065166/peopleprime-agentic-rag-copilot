import pytest
from app.rag.embeddings import embed_texts, embed_query, get_embedding_model


def test_embedding_model_singleton():
    model1 = get_embedding_model()
    model2 = get_embedding_model()
    assert model1 is model2


def test_embed_query_dimension_and_normalization():
    query = "What is the annual leave allowance for employees?"
    vector = embed_query(query)

    assert isinstance(vector, list)
    assert len(vector) == 384
    # Check that it's normalized (L2 norm ~= 1.0)
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_embed_texts_batch():
    texts = [
        "PeoplePrime attendance and core working hours policy.",
        "Medical insurance benefits coverage for full-time staff.",
    ]
    vectors = embed_texts(texts)

    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 384
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-4


def test_embed_query_empty_fails():
    with pytest.raises(ValueError, match="empty"):
        embed_query("   ")
