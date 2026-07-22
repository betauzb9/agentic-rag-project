import os
import tempfile
from typing import List, TypedDict

import fitz
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, END

# ---------- Provider config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not OPENAI_API_KEY and not GOOGLE_API_KEY:
    raise RuntimeError("Set at least one of OPENAI_API_KEY or GOOGLE_API_KEY.")

# Embedding dimensions differ per provider
EMBED_DIM = {"openai": 1536, "google": 768}

current_provider = "openai" if OPENAI_API_KEY else "google"

def get_llm(temperature=0):
    if current_provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=OPENAI_API_KEY)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=GOOGLE_API_KEY)

def get_vision_llm():
    if current_provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

def get_embeddings():
    if current_provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    else:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def switch_provider():
    """Switch to the other provider if available. Returns True if switched."""
    global current_provider
    if current_provider == "openai" and GOOGLE_API_KEY:
        current_provider = "google"
        print("Switched to Google Gemini (fallback).")
        return True
    if current_provider == "google" and OPENAI_API_KEY:
        current_provider = "openai"
        print("Switched to OpenAI (fallback).")
        return True
    return False

def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "quota" in msg or "rate limit" in msg or "429" in msg or "resourceexhausted" in msg

def call_with_fallback(func):
    """Calls func(); if a quota/rate-limit error occurs, switches provider and retries once."""
    try:
        return func()
    except Exception as e:
        if is_quota_error(e) and switch_provider():
            rebuild_pipeline()
            return func()
        raise

# ---------- Vector store (dimension depends on active provider) ----------
client = QdrantClient(location=":memory:")
COLLECTION_NAME = "agentic_rag"

vectorstore = None
retriever = None
document_loaded = False
stored_chunks = []  # keep raw chunks so we can re-embed if provider switches

def reset_collection():
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM[current_provider], distance=Distance.COSINE),
    )

def rebuild_pipeline():
    """Rebuilds embeddings/vectorstore/retriever after a provider switch, re-embedding stored chunks."""
    global vectorstore, retriever
    reset_collection()
    embeddings = get_embeddings()
    vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    if stored_chunks:
        vectorstore.add_documents(stored_chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rebuild_pipeline()

# ---------- Ingestion helpers ----------
def caption_image(image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    def _call():
        vision_llm = get_vision_llm()
        msg = vision_llm.invoke([{"role": "user", "content": [
            {"type": "text", "text": "Describe what is shown in this image, briefly."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}])
        return msg.content
    return call_with_fallback(_call)

def ingest_pdf(path: str, filename: str):
    docs = []
    pdf = fitz.open(path)
    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": page_num}))
        for img in page.get_images(full=True):
            base_image = pdf.extract_image(img[0])
            caption = caption_image(base_image["image"])
            docs.append(Document(page_content=f"[Image] {caption}", metadata={"source": filename, "page": page_num}))
    return docs

def ingest_text(path: str, filename: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": filename})]

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

def grade_doc(document, question):
    def _call():
        llm = get_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are grading whether a document chunk is relevant to a user's question. "
                       "Judge by meaning, not language. If the chunk touches the topic even loosely, answer 'yes'. "
                       "Only answer 'no' if the chunk is clearly about something unrelated."),
            ("human", "Document: {document}\n\nQuestion: {question}")
        ]) | llm.with_structured_output(YesNo)
        return chain.invoke({"document": document, "question": question})
    return call_with_fallback(_call)

def grade_hallucination(documents, generation):
    def _call():
        llm = get_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "Is the answer grounded in the given documents? Answer 'yes' or 'no'."),
            ("human", "Documents: {documents}\n\nAnswer: {generation}")
        ]) | llm.with_structured_output(YesNo)
        return chain.invoke({"documents": documents, "generation": generation})
    return call_with_fallback(_call)

def grade_answer(question, generation):
    def _call():
        llm = get_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "Does the answer directly address the question? Answer 'yes' or 'no'."),
            ("human", "Question: {question}\n\nAnswer: {generation}")
        ]) | llm.with_structured_output(YesNo)
        return chain.invoke({"question": question, "generation": generation})
    return call_with_fallback(_call)

def generate_answer(question, context):
    def _call():
        llm = get_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "Answer only based on the given documents. If the answer isn't in the documents, say: "
                       "'I could not find an answer to this in the documents.'"),
            ("human", "Question: {question}\n\nDocuments:\n{context}")
        ]) | llm
        return chain.invoke({"question": question, "context": context})
    return call_with_fallback(_call)

# ---------- Graph nodes ----------
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "steps": state.get("steps", []) + ["retrieve"]}

def grade_documents(state):
    filtered = [d for d in state["documents"]
                if grade_doc(d.page_content, state["question"]).binary_score == "yes"]
    return {"documents": filtered, "steps": state["steps"] + ["grade_documents"]}

def web_search(state):
    return {"documents": state["documents"], "steps": state["steps"] + ["web_search_skipped"]}

def generate(state):
    if not state["documents"]:
        return {"generation": "I could not find an answer to this in the documents.",
                "sources": [], "steps": state["steps"] + ["generate"],
                "retries": state.get("retries", 0) + 1}
    context = "\n\n".join(d.page_content for d in state["documents"])
    result = generate_answer(state["question"], context)
    sources = list({d.metadata.get("source", "document") for d in state["documents"]})
    return {"generation": result.content, "sources": sources,
            "steps": state["steps"] + ["generate"], "retries": state.get("retries", 0) + 1}

def route_after_grade(state):
    return "generate" if state["documents"] else "web_search"

def route_after_generate(state):
    if state.get("retries", 0) >= 3 or not state["documents"]:
        return "useful"
    if grade_hallucination(state["documents"], state["generation"]).binary_score == "no":
        return "not_grounded"
    useful = grade_answer(state["question"], state["generation"]).binary_score
    return "useful" if useful == "yes" else "not_useful"

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

@api.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_loaded, stored_chunks
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        docs = ingest_pdf(tmp_path, file.filename)
    else:
        docs = ingest_text(tmp_path, file.filename)

    os.remove(tmp_path)

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    stored_chunks = chunks

    def _call():
        rebuild_pipeline()

    call_with_fallback(_call)
    document_loaded = True

    return {"status": "ok", "filename": file.filename, "chunks": len(chunks), "provider": current_provider}

@api.post("/chat")
def chat(body: ChatIn):
    if not document_loaded:
        return {"answer": "Please upload a document first using /upload.", "steps": [], "sources": []}
    r = rag_app.invoke({"question": body.question, "steps": [], "retries": 0})
    return {"answer": r["generation"], "steps": r["steps"], "sources": r["sources"]}

@api.get("/health")
def health():
    return {"status": "ok", "document_loaded": document_loaded, "current_provider": current_provider}
