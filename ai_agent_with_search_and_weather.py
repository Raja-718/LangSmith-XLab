from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

#  FIXED DuckDuckGo tool (stable version)
search = DuckDuckGoSearchAPIWrapper()

search_tool = Tool(
    name="duckduckgo_search",
    func=lambda q: search.run(q)[:1000],  # limit noisy output
    description="Search the web using DuckDuckGo and extract useful information from results."
)

# Weather tool (unchanged)
@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()

# Gemini model (unchanged)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Load ReAct prompt
prompt = hub.pull("hwchase17/react")

#  FIX: guide agent to stop looping
prompt.template += "\nAlways extract the final answer from the search result. Do not keep searching repeatedly."

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

#  FIX: increase iteration limit
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=10
)

# Run query
response = agent_executor.invoke({
    "input": "What is the release date of Pushpa 2 movie?"
})

print(response)
print(response['output'])