from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import asyncio

try:
    # Check if a running event loop exists
    asyncio.get_running_loop()
except RuntimeError:
    # No event loop, so create and set one
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(google_api_key = GOOGLE_API_KEY,model="gemini-embedding-001")

def Create_vector_db_from_youtube(video_url: str) -> FAISS:
    # Extract video id
    video_id = video_url.split("v=")[-1].split("&")[0]

    api = YouTubeTranscriptApi()
    transcript_obj = api.fetch(video_id)
    full_text = " ".join([entry.text for entry in transcript_obj])

    # Wrap text as a LangChain Document
    transcript = [Document(page_content=full_text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_vector_db(query,db,k=2):

    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])
    llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,model="gemini-2.5-pro")
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""
        Answer the question based on the context below:

        Context: {context}

        Question: {query}

        Answer:
        """
    )
    chain = LLMChain(llm=llm,prompt=prompt_template)
    response = chain.invoke({
        "context": docs_page_content,
        "query": query
    })
    
    response_text = response.get("text", "")
    response = response.replace("\n", " ")
    return response

