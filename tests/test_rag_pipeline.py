import shutil

import os
import pytest

from app import constants, rag_pipeline


@pytest.fixture(scope="function")
def clean_index(tmp_path) -> None:  # noqa: ARG001
    """
    Ensure a clean chroma index directory before each test.
    """
    index_path = "data/chroma_index"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    yield index_path
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


def test_rag_pipeline_creates_index(clean_index) -> None:
    """
    Test that running the RAG pipeline creates the Chroma index directory.
    """
    os.makedirs(constants.CHROMA_DIR, exist_ok=True)
    rag_pipeline.get_rag_chain()  # Run the pipeline
    assert os.path.exists(clean_index), "Chroma index directory was not created"
    assert any(os.scandir(clean_index)), "Chroma index directory is empty"


def test_load_documents_returns_strings() -> None:
    """
    Ensure that `rag_pipeline.load_documents` returns a list of textual items.
    This test validates the preprocessing phase before embedding.
    """
    docs = rag_pipeline.load_documents()

    assert isinstance(docs, list), "Expected a list of documents."
    assert all(isinstance(d, str) for d in docs), "All documents should be strings."
    assert len(docs) > 0, "Documents list should not be empty."
