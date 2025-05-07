from fastapi import FastAPI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # เปลี่ยนจาก OpenAIEmbeddings
import os

app = FastAPI()

# ใช้โมเดลฟรีจาก HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@app.post("/crawl")
async def crawl(url: str):
    loader = WebBaseLoader([url])
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        embeddings=embeddings  # ใช้ local embeddings
    )
    chunks = splitter.split_documents(docs)
    
    return {
        "content": [chunk.page_content for chunk in chunks],
        "embeddings": embeddings.embed_documents([chunk.page_content for chunk in chunks]),  # สร้าง embeddings
        "status": "SUCCESS"
    }