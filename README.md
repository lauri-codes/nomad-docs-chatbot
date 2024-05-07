# nomad-docs-chatbot

This is a Python package containing a RAG (Retrieval Augmented Generation) based chatbot for NOMAD documentation. The code can be run completely locally with open-source models.

This implementation uses a form of [self-correction](https://arxiv.org/abs/2310.11511), where the LLM filters out un-relevant context documents and performs a check for hallusinations and usefullness of the answer. If the generated answer does not pass the checks, a hard-coded failure response will be returned.

By default, the LLM model and embedding run on the local GPU. The default Llama 3 8B model requires roughly 3GB of GPU RAM. On a laptop with `GeForce GTX 1650 Mobile` answering a question takes roughly 1 minute (a single query with default settings can contain up to 7 LLM calls due to self-correction).

## Usage

```python
from nomad_docs_chatbot import RAG

# Run with default model
rag = RAG()
answer = RAG.query("What is NOMAD?")
print(answer)

# Use custom model
llm_remote = ChatOllama(model="llama3:70b")
llm_remote.base_url = 'http://172.28.105.30/backend'
rag = RAG(model=model)

# Modify temperature and number of source documents used as context.
rag = RAG(temperature=0.1, k=3)
```

## Examples

```text
Q: What is NOMAD?
A: NOMAD is a data management system that processes files to extract data from various formats, allowing for the creation of search interfaces, APIs, visualization tools, and analysis tools independent of specific file formats.
It's based on a bottom-up approach to data management, converting heterogeneous files into homogeneous machine-actionable processed data to make data FAIR (Findable, Accessible, Interoperable, and Reusable)."

Q: What is a schema in NOMAD?
A: In NOMAD, a schema defines the data structures and organization of Processed Data. Schemas can be defined in yaml or python formats, and are used to define sections that contain data and more sections, allowing for browsing complex data like files and directories on your computer.
```

## Installation

By default the implementation uses a local LLM model based on Llama 3 8B. For this you will need to install the following pre-requisite libraries:

- [`ollama`](https://ollama.com/): Package manager for open-source LLM models. Install with:
    
    ```sh
    curl -fsSL https://ollama.com/install.sh | sh
    ```

- [`llama3`](https://ai.meta.com/blog/meta-llama-3/): The open-source LLM model that is used. Install with ollama:
    
    ```sh
    ollama pull llama3
    ```

The python package itself can be installed with:

```sh
pip install .
```