import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent, WebSearchTool

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to the .env file.")

model = OpenAIServerModel(model_id="gpt-4o-mini", api_key=api_key)
agent = ToolCallingAgent(tools=[WebSearchTool()], model=model)
result = agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
print(result)
# CLI examples:
# smolagent "Plan a trip to Tokyo, Kyoto and Osaka between Mar 28 and Apr 7." --model-type "InferenceClientModel" --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" --imports "pandas numpy" --tools "web_search"
# smolagent