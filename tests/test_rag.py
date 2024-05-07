import pytest
from nomad_nlp_engine import RAG
from langchain_community.chat_models import ChatOllama

llm_remote = ChatOllama(model="llama3:70b")
llm_remote.base_url = 'http://172.28.105.30/backend'

@pytest.mark.parametrize("model, query, expected_answer", [
    # (None, "What is a plugin entry point?", ),
    (None, "What is NOMAD?", "NOMAD is a data management system that processes files to extract data from various formats, allowing for the creation of search interfaces, APIs, visualization tools, and analysis tools independent of specific file formats. It's based on a bottom-up approach to data management, converting heterogeneous files into homogeneous machine-actionable processed data to make data FAIR (Findable, Accessible, Interoperable, and Reusable)."),
])
def test_rag(model, query, expected_answer):
    rag = RAG(model=model)
    answer = rag.query(query)
    assert answer == expected_answer
