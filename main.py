import asyncio
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm_name: str = "rwkv7-g1d"
provider_url: str = "http://127.0.0.1:65530/api/oai"

model = ChatOpenAI(model=llm_name, base_url=provider_url, api_key="Anything")

template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates English to Japanese. Translate the user sentence.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Translate the word into Japanese language: {query} (think less)",
        ),
    ]
)

MAX_ITERATIONS = 1000
MAX_HISTORY = 10


async def process_chat_stream(chain, query: str, history: List[Tuple]) -> str:
    """Helper function of run_model()."""

    assert query and len(query) > 0, "[Program] Query cannot be empty."

    content_acc = ""
    try:
        async for chunk in chain.astream({"query": query, "chat_history": history}):
            if chunk and chunk.content:
                print(chunk.content, end="", flush=True)
                content_acc += chunk.content
        print()
        return content_acc
    except Exception as e:
        print(f"\nStream Error: {e}")
        return ""


async def run_model(model: ChatOpenAI, template: ChatPromptTemplate):
    assert model is not None, "[Program] Model initialization failed."
    assert template is not None, "[Program] Template is missing."

    chat_history: List[Tuple] = []
    chain = template | model

    print("--- AI AGENT: Program Online ---")
    print("Type /exit or /quit to off the system.")

    for _ in range(MAX_ITERATIONS):
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "> "
            )

            if not user_input or user_input.lower() in ["/exit", "/quit"]:
                break

            ai_response = await process_chat_stream(chain, user_input, chat_history)

            if ai_response:
                chat_history.append(("human", user_input))
                chat_history.append(("ai", ai_response))

            if len(chat_history) > MAX_HISTORY * 2:
                chat_history = chat_history[-(MAX_HISTORY * 2) :]

        except asyncio.CancelledError:
            print("\n[Program] Signal received: Terminating Program.")
            break
        except Exception as e:
            print(f"\n[Recoverable Error]: {e}")
            continue

    print("--- AI AGENT: Program Offline ---")
    print("Press Enter to continue.")


if __name__ == "__main__":
    try:
        asyncio.run(run_model(model, template))
    except KeyboardInterrupt:
        pass
