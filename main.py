from langchain_openai import ChatOpenAI

llm_name: str = "rwkv7-g1d"
provider_url: str = "http://127.0.0.1:65530/api/oai"

model = ChatOpenAI(model=llm_name, base_url=provider_url, api_key="Anything")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)
print(ai_msg)
