from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType

from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_pet_name(pet_type, number_of_names, style):
    llm = ChatGoogleGenerativeAI(
        google_api_key = GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature = 1.2,
    )               

    prompt_template = PromptTemplate(
    input_variables = ["pet_type", "number_of_names", "style"],
    template = "i have a pet {pet_type}, can you generate me {number_of_names}  {style}  names for it ?"
    )

     # Modern way: use prompt | llm
    chain = prompt_template | llm
    response = chain.invoke({
        "pet_type": pet_type,
        "number_of_names": number_of_names,
        "style": style
    })

    return response.content

def langchain_agent():
    llm = ChatGoogleGenerativeAI(temperature = 0.9,google_api_key = GOOGLE_API_KEY,model="gemini-1.5-flash")
    tools = load_tools(["wikipedia","llm-math","google-search"], llm=llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent.run("what is average age of dog. multilply it by 2")
    
    print(result)


if __name__ == "__main__":
    langchain_agent()
    #print(generate_pet_name("cat", 5, "elegant"))