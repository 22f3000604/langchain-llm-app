from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
Google_Api_key = os.getenv("Google_Api_Key")

def generate_pet_name():
    llm = ChatGoogleGenerativeAI(
        google_api_key = Google_Api_key,
        model="gemini-1.5-flash",
        temperature = 0.2,
    )               

    name = llm.invoke("i have a pet dog, can you generate me 7 cool superhero  names for it ?")

    return name.content

if __name__ == "__main__":
    print(generate_pet_name())    