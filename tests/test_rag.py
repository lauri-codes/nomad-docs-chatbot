import pytest
from nomad_docs_chatbot import RAG


@pytest.mark.parametrize("model, query, expected_answer", [
    (None, "What is NOMAD?", "NOMAD is a data management system that processes files to extract data from various formats, allowing for the creation of search interfaces, APIs, visualization tools, and analysis tools independent of specific file formats. It's based on a bottom-up approach to data management, converting heterogeneous files into homogeneous machine-actionable processed data to make data FAIR (Findable, Accessible, Interoperable, and Reusable)."),
    # (None, "What is a schema in NOMAD?", "In NOMAD, a schema defines the data structures and organization of Processed Data. Schemas can be defined in yaml or python formats, and are used to define sections that contain data and more sections, allowing for browsing complex data like files and directories on your computer."),
])
def test_rag(model, query, expected_answer):
    rag = RAG(model=model, k=4)
    answer = rag.query(query)
    assert answer == expected_answer
