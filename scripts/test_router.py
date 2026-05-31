import os
from dotenv import load_dotenv
from agent.router import classify_query
from langchain_openai import ChatOpenAI

load_dotenv()

NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
ROUTER_MODEL = os.getenv("NEBIUS_ROUTER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

def main():
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("Error: NEBIUS_API_KEY is not set. Please set it in .env or your environment.")
        return

    model = ChatOpenAI(
        model=ROUTER_MODEL,
        api_key=api_key,
        base_url=NEBIUS_BASE_URL,
    )

    queries = [
        "How many refund requests?",
        "Show me 3 examples from the REFUND category",
        "Summarize the FEEDBACK category",
        "Tell me about the common themes in complaints",
        "Who won the 2024 Champions League?",
        "Write me a poem"
    ]

    print("Testing router classifications:\n")
    for q in queries:
        try:
            res = classify_query(q, model)
            print(f"Q: {q}")
            print(f"A: {res}\n")
        except Exception as e:
            print(f"Error classifying '{q}': {e}\n")

if __name__ == "__main__":
    main()
