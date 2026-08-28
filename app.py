from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
# step 1 : Indexing ( here using yt api we will load any video's transcript, and will bring it as string in our code)
ytt_api = YouTubeTranscriptApi()

def get_transcript(video_id):

    try:
        transcript_list = ytt_api.fetch(
            video_id,
            languages=["en", "hi" , "bn"]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        return transcript

    except TranscriptsDisabled:
        return "No captions available for this video."
    
# step 2: Chunk Division using text splitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))

# step 3: convert chunks to vectors and store in vector store

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks,embedding_model)

print(vector_store.index_to_docstore_id)

# step 4: form a retriver 

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
# retriever.invoke("What is neural network")

# Step 5: Augmentation

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

chat_model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context. 
                If the conetxt is insufficient, just say you dont know.
                
                {context}
                Question: {question}
                """,
                input_variables = ['context','question']    
)

question = "give me the summary of the whole video , also tell me  what topic the video is"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context" : context_text, "question" : question})

# step 6: Generation
answer = chat_model.invoke(final_prompt)
print(answer.content)
