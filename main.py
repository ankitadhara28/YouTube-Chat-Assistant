from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()


# Allow Chrome Extension to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


ytt_api = YouTubeTranscriptApi()

# This will store the retriever of the CURRENT video
retriever = None


# -----------------------------
# Request models
# -----------------------------

class VideoRequest(BaseModel):
    video_id: str


class QuestionRequest(BaseModel):
    question: str


# -----------------------------
# Load LLM and Embedding model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

chat_model = ChatHuggingFace(llm=llm)


# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "YouTube Chat Assistant Backend is running"
    }


# -----------------------------
# Process Video
# -----------------------------

@app.post("/process-video")
def process_video(request: VideoRequest):

    global retriever

    video_id = request.video_id

    try:

        # 1. Fetch transcript
        transcript_list = ytt_api.fetch(
            video_id,
            languages=["en", "hi", "bn"]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        # 2. Split transcript
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([transcript])

        # 3. Create FAISS vector store
        vector_store = FAISS.from_documents(
            chunks,
            embedding_model
        )

        # 4. Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return {
            "success": True,
            "video_id": video_id,
            "message": "Transcript loaded successfully",
            "chunks": len(chunks)
        }

    except TranscriptsDisabled:

        return {
            "success": False,
            "message": "No captions available for this video."
        }


# -----------------------------
# Ask Question
# -----------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    global retriever

    if retriever is None:
        return {
            "success": False,
            "answer": "Please load a YouTube video first."
        }

    question = request.question

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Combine chunks
    context_text = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    # Ask LLM
    answer = chat_model.invoke(final_prompt)

    return {
        "success": True,
        "answer": answer.content
    }