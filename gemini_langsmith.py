import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load env
load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# Chain
chain = prompt | llm

# Run
response = chain.invoke({"topic": "Machine Learning"})

print(response.content)