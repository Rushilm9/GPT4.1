import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Azure GPT-4.1 Model Configuration ---
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2024-05-01-preview",
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.0
)

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="messages")
])

# --- Initialize or Load Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LCEL Chain ---
chain = prompt | llm

# --- Handle User Input and Update Chat History ---
user_input = st.chat_input("Ask me anything!")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["type"]):
        st.write(msg["content"])

if user_input:
    # Add user input to history
    st.session_state.messages.append({"type": "human", "content": user_input})

    # Display the user's message immediately
    with st.chat_message("human"):
        st.write(user_input)

    # Show the "Thinking..." message while AI is processing
    with st.chat_message("ai"):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("Thinking... Please wait.")

        # Run chain with updated message history
        response = chain.invoke(
            {"messages": st.session_state.messages},
            config={"configurable": {"session_id": "default"}}
        )

        # Replace the "Thinking..." message with the AI response
        thinking_placeholder.write(response.content)

    # Add AI response to history
    st.session_state.messages.append({"type": "ai", "content": response.content})
