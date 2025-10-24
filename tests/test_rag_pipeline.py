import shutil

import os
import pytest

from app import rag_pipeline


@pytest.fixture(scope="function")
def clean_index(tmp_path):
    """Ensure a clean chroma index directory before each test."""
    index_path = "data/chroma_index"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    yield index_path
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


def test_rag_pipeline_creates_index(clean_index):
    """Test that running the RAG pipeline creates the Chroma index directory."""
    rag_pipeline.get_rag_chain()  # Run the pipeline
    assert os.path.exists(clean_index), "Chroma index directory was not created"
    assert any(os.scandir(clean_index)), "Chroma index directory is empty"
