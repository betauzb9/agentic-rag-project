import os
import tempfile
from typing import List, TypedDict, Optional, Any

import fitz
import base64
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # SDK class only; points at DeepSeek's endpoint below
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, END

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")

# Chat + grading run on DeepSeek V4 Flash.
# (deepseek-chat is retired 2026/07/24; deepseek-v4-flash is its replacement, non-thinking mode.)
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# DeepSeek has no vision model, so image captioning is disabled — PDFs still ingest fine,
# embedded images just get a placeholder instead of a real caption.
vision_llm = None

# DeepSeek has no embeddings endpoint, so embeddings run on a free local model
# (requires: pip install fastembed). No API key needed for this part.
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384

client = QdrantClient(location=":memory:")

# ---------- Per-session storage ----------
# Each browser gets its own session_id (sent as the X-Session-Id header).
# Every session has its own Qdrant collection + retriever, so uploads from
# one user never overwrite or leak into another user's chat.
SESSIONS: dict[str, dict] = {}

def collection_name_for(session_id: str) -> str:
    # Qdrant collection names are restricted to a safe charset; session_id is a UUID so this is safe.
    return f"agentic_rag_{session_id}"

def create_session_collection(session_id: str) -> str:
    name = collection_name_for(session_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    return name

def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"retriever": None, "document_loaded": False}
    return SESSIONS[session_id]

def caption_image(image_bytes):
    if vision_llm is None:
        return "[Image could not be captioned: no vision-capable API key available]"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    msg = vision_llm.invoke([{"role": "user", "content": [
        {"type": "text", "text": "Describe what is shown in this image, briefly."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ]}])
    return msg.content

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

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    steps: List[str]
    sources: List[str]
    retries: int
    retriever: Any  # session-specific retriever, injected per request

class YesNo(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")

doc_grader = ChatPromptTemplate.from_messages([
    ("system", "You are grading whether a document chunk is relevant to a user's question. "
               "Judge by meaning, not language. If the chunk touches the topic even loosely, answer 'yes'. "
               "Only answer 'no' if the chunk is clearly about something unrelated."),
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

def retrieve(state):
    docs = state["retriever"].invoke(state["question"])
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

api = FastAPI(title="Agentic RAG API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies/auth used; safe (and required) to combine with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    question: str

@api.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str = Header(..., alias="X-Session-Id"),
):
    session = get_session(x_session_id)

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

    collection_name = create_session_collection(x_session_id)
    vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    vectorstore.add_documents(chunks)

    session["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 4})
    session["document_loaded"] = True

    return {"status": "ok", "filename": file.filename, "chunks": len(chunks)}

@api.post("/chat")
def chat(body: ChatIn, x_session_id: str = Header(..., alias="X-Session-Id")):
    session = get_session(x_session_id)
    if not session["document_loaded"]:
        return {"answer": "Please upload a document first using /upload.", "steps": [], "sources": []}
    r = rag_app.invoke({
        "question": body.question,
        "steps": [],
        "retries": 0,
        "retriever": session["retriever"],
    })
    return {"answer": r["generation"], "steps": r["steps"], "sources": r["sources"]}

@api.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}
