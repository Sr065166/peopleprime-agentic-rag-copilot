import pytest
import numpy as np
from app.rag.embeddings import embed_texts, embed_query, get_embedding_model, OnnxEmbeddingModel


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


def test_embed_texts_empty_list():
    assert embed_texts([]) == []


def test_onnx_embedding_engine_direct():
    model = OnnxEmbeddingModel()
    sentences = ["HR benefits policy", "Annual performance evaluation"]
    embeddings = model.encode(sentences)
    assert embeddings.shape == (2, 384)
    # Norm check
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4


def test_onnx_and_pytorch_cosine_similarity():
    """Verify that ONNX runtime produces virtually identical vectors to SentenceTransformer (>0.9999)."""
    from sentence_transformers import SentenceTransformer
    onnx_model = OnnxEmbeddingModel()
    torch_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    test_queries = [
        "What is the annual leave allowance for employees?",
        "Medical insurance coverage for full-time staff.",
    ]

    onnx_vecs = onnx_model.encode(test_queries)
    torch_vecs = torch_model.encode(test_queries, normalize_embeddings=True, convert_to_numpy=True)

    for i in range(len(test_queries)):
        cos_sim = float(np.dot(onnx_vecs[i], torch_vecs[i]))
        assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} too low for query: {test_queries[i]}"
