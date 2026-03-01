from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm_name: str = "rwkv7-g1d"
provider_url: str = "http://127.0.0.1:65530/api/oai"

model = ChatOpenAI(model=llm_name, base_url=provider_url, api_key="Anything")

template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query} (think less)"),
    ]
)

chat_history = []

context = template | model

for i in range(100):
    user_query = input("> ")
    chat_history.append(user_query)

    ai_msg = context.invoke({"query": user_query, "chat_history": chat_history})
    chat_history.append(ai_msg.content)
    print(ai_msg.content)
