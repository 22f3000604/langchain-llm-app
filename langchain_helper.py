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
import random

try:
    # Check if a running event loop exists
    asyncio.get_running_loop()
except RuntimeError:
    # No event loop, so create and set one
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="gemini-embedding-001")


def load_proxies(proxy_file_path: str):
    """
    Load proxies from a CSV or text file, one proxy per line, in format:
    http://ip:port
    socks4://ip:port
    """
    if not os.path.exists(proxy_file_path):
        print(f"Proxy file {proxy_file_path} not found. Running without proxies.")
        return []

    with open(proxy_file_path, "r") as f:
        proxies = [line.strip() for line in f.readlines() if line.strip()]
    return proxies
import concurrent.futures

def fetch_with_timeout(api, video_id, timeout_seconds=30):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(api.fetch, video_id)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Fetching transcript timed out after {timeout_seconds} seconds")
            return None

# Usage in your function
def Create_vector_db_from_youtube(video_url: str, proxy_list=None, timeout=60) -> FAISS:
    video_id = video_url.split("v=")[-1].split("&")[0]

    api = YouTubeTranscriptApi()

    # Use the timeout wrapper
    transcript_obj = fetch_with_timeout(api, video_id, timeout_seconds=timeout)
    if not transcript_obj:
        raise TimeoutError(f"Fetching transcript for video {video_id} timed out")

    full_text = " ".join([entry.text for entry in transcript_obj])

    transcript = [Document(page_content=full_text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_vector_db(query, db, k=2):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])
    llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-pro")
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""
        Answer the question based on the context below:

        Context: {context}

        Question: {query}

        Answer:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.invoke({
        "context": docs_page_content,
        "query": query
    })

    response_text = response.get("text", "")
    response_text = response_text.replace("\n", " ")
    return response_text


# Usage example:
# Load proxies from a local file 'proxies.txt' (one proxy URL per line like http://ip:port or socks4://ip:port)
proxies = load_proxies('proxies.txt')

# Create vector db with proxies enabled (if list empty, runs without proxy)
# db = Create_vector_db_from_youtube("https://www.youtube.com/watch?v=VIDEO_ID", proxy_list=proxies)
