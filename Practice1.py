import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import list_models
from smolagents import (
    CodeAgent,
    GradioUI,
    InferenceClientModel,
    LiteLLMModel,
    WebSearchTool,
    load_tool,
    tool,
)


HF_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def gemini_fibonacci() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

    model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key=api_key)
    agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    print(agent.run("Could you give me the 118th number in the Fibonacci sequence?"))


def get_huggingface_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HuggingFaceHub_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add your Hugging Face token to the .env file."
        )
    return token


@tool
def model_download_tool(task: str) -> str:
    """Return the most-downloaded Hugging Face model for a task.

    Args:
        task: Hugging Face model task to search for, such as text-to-video.
    """
    token = get_huggingface_token()
    most_downloaded_model = next(
        iter(list_models(filter=task, sort="downloads", limit=1, token=token)),
        None,
    )
    if most_downloaded_model is None:
        return f"No Hugging Face models found for task: {task}"
    return most_downloaded_model.id


def huggingface_model_downloads() -> None:
    token = get_huggingface_token()
    model = InferenceClientModel(model_id=HF_MODEL_ID, token=token)
    agent = CodeAgent(tools=[model_download_tool], model=model)
    print(
        agent.run(
            "Find the most-downloaded model for the 'text-to-video' task "
            "on the Hugging Face Hub."
        )
    )


def managed_web_search() -> None:
    token = get_huggingface_token()
    model = InferenceClientModel(model_id=HF_MODEL_ID, token=token)
    web_agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        name="web_search_agent",
        description="Searches the web and reports findings.",
    )
    manager_agent = CodeAgent(tools=[], model=model, managed_agents=[web_agent])
    print(manager_agent.run("Who is the CEO of Hugging Face?"))


def launch_image_generation_ui() -> None:
    token = get_huggingface_token()
    image_generation_tool = load_tool(
        "m-ric/text-to-image",
        token=token,
        trust_remote_code=True,
    )
    model = InferenceClientModel(model_id=HF_MODEL_ID, token=token)
    agent = CodeAgent(tools=[image_generation_tool], model=model)
    GradioUI(agent).launch()


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    actions = {
        "1": ("Gemini Fibonacci example", gemini_fibonacci),
        "2": ("Most-downloaded Hugging Face model by task", huggingface_model_downloads),
        "3": ("Managed web-search agent", managed_web_search),
        "4": ("Image-generation Gradio UI", launch_image_generation_ui),
    }

    while True:
        print("\nChoose a practice:")
        for number, (label, _) in actions.items():
            print(f"{number}. {label}")
        print("q. Quit")

        choice = input("> ").strip().lower()
        if choice == "q":
            return
        action = actions.get(choice)
        if action is None:
            print("Choose 1, 2, 3, 4, or q.")
            continue
        action[1]()


if __name__ == "__main__":
    main()
