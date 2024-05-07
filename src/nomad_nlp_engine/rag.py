from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List
import time
import logging

from nomad_nlp_engine.embeddings import get_retriever


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


class RAG():
    """Class used for Retrieval Augmented Generation."""
    def __init__(
            self,
            model = None,
            embedding: str = 'gpt4all',
            k: int = 5,
            logger=None
        ):
        """
        Args:
            model: The LLM base model to use.
            embedding: The embedding function used for retrieval.
            k: Amount of documents to retrieve for context.
            temperature: The temperature that controls the final
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        if model is None:
            model = ChatOllama(model='llama3', temperature=0)
        self.model = model
        self.retriever = get_retriever(embedding, k)

        # Retrieval grader. This part uses LLM to assess the relevance of a retrieved
        # document.
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )
        self.retrieval_grader = prompt | self.model.bind(temperature=0, format="json") | JsonOutputParser()
        self.k = k

        # This is the final LLM question prompt where the context is embedded into.
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question}
            Context: {context}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "document"],
        )
        self.rag_chain = prompt | self.model | StrOutputParser()

        # Hallucination Grader
        prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )
        self.hallucination_grader = prompt | self.model.bind(temperature=0, format="json") | JsonOutputParser()

        # Answer Grader
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation}
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )

        self.answer_grader = prompt | self.model.bind(temperature=0, format="json") | JsonOutputParser()
        self.app = self.get_app()

    def get_app(self):

        def retrieve(state):
            """
            Retrieve documents from vectorstore

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            self.logger.info("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = self.retriever.invoke(question)[0:self.k]
            return {"documents": documents, "question": question}

        def generate(state):
            """
            Generate answer using RAG on retrieved documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            self.logger.info("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            # RAG generation
            generation = self.rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}

        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Filtered out irrelevant documents
            """

            self.logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            # Score each doc
            filtered_docs = []
            for d in documents:
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score["score"]
                # Document relevant
                if grade.lower() == "yes":
                    self.logger.info("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                # Document not relevant
                else:
                    self.logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                    # We do not include the document in filtered_docs
                    continue
            return {"documents": filtered_docs, "question": question}

        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """

            self.logger.info("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]

            score = self.hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score["score"]

            # Check hallucination
            if grade == "yes":
                self.logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                self.logger.info("---GRADE GENERATION vs QUESTION---")
                score = self.answer_grader.invoke({"question": question, "generation": generation})
                grade = score["score"]
                if grade == "yes":
                    self.logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    self.logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                self.logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not useful"

        def failure(state):
            """
            Generate answer using RAG on retrieved documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            self.logger.info("---FAILED TO GENERATE USEFUL ANSWER---")

            return {"generation": "Infortunately I'm not able to provide a good answer to your question based on my current knowledge."}

        # Build graph
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generate
        workflow.add_node("failure", failure)  # 

        workflow.set_entry_point('retrieve')
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "useful": END,
                "not useful": "failure",
            },
        )
        workflow.add_edge("failure", END)

        # Compile
        app = workflow.compile()
        return app

    def query(self, input):
        inputs = {"question": input}
        for output in self.app.stream(inputs):
            for key, value in output.items():
                self.logger.info(f"Finished running: {key}:")
        return value["generation"]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rag = RAG()
    start = time.time()
    print(rag.query("What is NOMAD?"))
    end = time.time()
    print(end-start)
