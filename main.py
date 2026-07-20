import os
from typing import List, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, END

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------- Load and ingest the default document ----------
DEFAULT_DOC_PATH = os.path.join(os.path.dirname(__file__), "data", "alice_in_wonderland.txt")

def load_default_document():
    docs = []
    if os.path.exists(DEFAULT_DOC_PATH):
        with open(DEFAULT_DOC_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": "alice_in_wonderland.txt"}))
    return docs

raw_docs = load_default_document()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(raw_docs)

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="agentic_rag",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
vectorstore = QdrantVectorStore(client=client, collection_name="agentic_rag", embedding=embeddings)
if chunks:
    vectorstore.add_documents(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- Graph state ----------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    steps: List[str]
    sources: List[str]
    retries: int

class YesNo(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

doc_grader = ChatPromptTemplate.from_messages([
    ("system", "You are grading whether a document chunk is relevant to a user's question. "
               "The chunk may be in a different language than the question — judge by meaning, not language. "
               "If the chunk is part of the same book/document the question is about, or touches the topic "
               "even loosely, answer 'yes'. Only answer 'no' if the chunk is clearly about something unrelated."),
    ("human", "Document: {document}\n\nQuestion: {question}")
]) | llm.with_structured_output(YesNo)

hallucination_grader = ChatPromptTemplate.from_messages([
    ("system", "Is the answer grounded in the given documents? Answer 'yes' or 'no'."),
    ("human", "Documents: {documents}\n\nAnswer: {generation}")
]) | llm.with_structured_output(YesNo)

answer_grader = ChatPromptTemplate.from_messages([
    ("system", "Does the answer directly address the question? Answer 'yes' or 'no'."),
    ("human", "Question: {question}\n\nAnswer: {generation}")
]) | llm.with_structured_output(YesNo)

rag_chain = ChatPromptTemplate.from_messages([
    ("system", "Answer only based on the given documents. If the answer isn't in the documents, say: "
               "'I could not find an answer to this in the documents.'"),
    ("human", "Question: {question}\n\nDocuments:\n{context}")
]) | llm

# ---------- Graph nodes ----------
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "steps": state.get("steps", []) + ["retrieve"]}

def grade_documents(state):
    filtered = [d for d in state["documents"]
                if doc_grader.invoke({"document": d.page_content, "question": state["question"]}).binary_score == "yes"]
    return {"documents": filtered, "steps": state["steps"] + ["grade_documents"]}

def web_search(state):
    return {"documents": state["documents"], "steps": state["steps"] + ["web_search_skipped"]}

def generate(state):
    if not state["documents"]:
        return {"generation": "I could not find an answer to this in the documents.",
                "sources": [], "steps": state["steps"] + ["generate"],
                "retries": state.get("retries", 0) + 1}
    context = "\n\n".join(d.page_content for d in state["documents"])
    result = rag_chain.invoke({"question": state["question"], "context": context})
    sources = list({d.metadata.get("source", "document") for d in state["documents"]})
    return {"generation": result.content, "sources": sources,
            "steps": state["steps"] + ["generate"], "retries": state.get("retries", 0) + 1}

def route_after_grade(state):
    return "generate" if state["documents"] else "web_search"

def route_after_generate(state):
    if state.get("retries", 0) >= 3 or not state["documents"]:
        return "useful"
    if hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]}).binary_score == "no":
        return "not_grounded"
    useful = answer_grader.invoke({"question": state["question"], "generation": state["generation"]}).binary_score
    return "useful" if useful == "yes" else "not_useful"

# ---------- Build the graph ----------
g = StateGraph(GraphState)
g.add_node("retrieve", retrieve)
g.add_node("grade_documents", grade_documents)
g.add_node("web_search", web_search)
g.add_node("generate", generate)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "grade_documents")
g.add_conditional_edges("grade_documents", route_after_grade, {"web_search": "web_search", "generate": "generate"})
g.add_edge("web_search", "generate")
g.add_conditional_edges("generate", route_after_generate,
                        {"useful": END, "not_grounded": "generate", "not_useful": "web_search"})
rag_app = g.compile()

# ---------- FastAPI ----------
api = FastAPI(title="Agentic RAG API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    question: str

@api.post("/chat")
def chat(body: ChatIn):
    r = rag_app.invoke({"question": body.question, "steps": [], "retries": 0})
    return {"answer": r["generation"], "steps": r["steps"], "sources": r["sources"]}

@api.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks)}