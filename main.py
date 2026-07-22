import os
from typing import List, TypedDict

import openai
from fastapi import FastAPI, UploadFile, File
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

# ---------- Multiple API keys with fallback ----------
API_KEYS = [
    os.environ.get("OPENAI_API_KEY"),
    os.environ.get("OPENAI_API_KEY_2"),
    os.environ.get("OPENAI_API_KEY_3"),
]
API_KEYS = [k for k in API_KEYS if k]  # remove empty ones

if not API_KEYS:
    raise RuntimeError("No OPENAI_API_KEY environment variables are set.")

current_key_index = 0

def get_working_llm(model="gpt-4o-mini", temperature=0):
    """Returns an LLM instance using the current working API key."""
    global current_key_index
    key = API_KEYS[current_key_index]
    return ChatOpenAI(model=model, temperature=temperature, api_key=key)

def get_working_embeddings():
    global current_key_index
    key = API_KEYS[current_key_index]
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)

def switch_to_next_key():
    """Switches to the next available API key. Returns True if switched, False if no more keys."""
    global current_key_index
    if current_key_index + 1 < len(API_KEYS):
        current_key_index += 1
        print(f"Switching to backup API key #{current_key_index + 1}")
        return True
    return False

def call_with_fallback(func, *args, **kwargs):
    """Calls func(*args, **kwargs); on quota/rate-limit error, retries with next key."""
    global current_key_index
    attempts = 0
    while attempts < len(API_KEYS):
        try:
            return func(*args, **kwargs)
        except openai.RateLimitError as e:
            print(f"API key #{current_key_index + 1} failed: {e}")
            if not switch_to_next_key():
                raise RuntimeError("All API keys exhausted their quota.") from e
            attempts += 1
    raise RuntimeError("All API keys failed.")
    def generate(state):
    if not state["documents"]:
        return {"generation": "I could not find an answer to this in the documents.",
                "sources": [], "steps": state["steps"] + ["generate"],
                "retries": state.get("retries", 0) + 1}
    context = "\n\n".join(d.page_content for d in state["documents"])
    
    def _call():
        llm = get_working_llm()
        chain = ChatPromptTemplate.from_messages([
            ("system", "Answer only based on the given documents. If the answer isn't in the documents, say: "
                       "'I could not find an answer to this in the documents.'"),
            ("human", "Question: {question}\n\nDocuments:\n{context}")
        ]) | llm
        return chain.invoke({"question": state["question"], "context": context})
    
    result = call_with_fallback(_call)
    sources = list({d.metadata.get("source", "document") for d in state["documents"]})
    return {"generation": result.content, "sources": sources,
            "steps": state["steps"] + ["generate"], "retries": state.get("retries", 0) + 1}
