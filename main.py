import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, chat

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


async def run_model(model: ChatOpenAI, template: ChatPromptTemplate):
    chat_history = []

    context = template | model

    for _ in range(100):
        user_query = input("> ")

        gen_content = ""

        async for chunk in context.astream(
            {"query": user_query, "chat_history": chat_history}
        ):
            # gen_content += chunk.content
            print(chunk.content, end="", flush=True)

        chat_history.append(("human", user_query))
        chat_history.append(("ai", gen_content))
        print()


asyncio.run(run_model(model, template))
