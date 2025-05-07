import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import time  # For simulating delay (you can adjust or remove this)

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
        temperature=0.0  # Ensures predictable tool usage
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

if user_input:
    # Add user input to history
    st.session_state.messages.append({"type": "human", "content": user_input})

    # Show the "thinking" message temporarily while AI is processing
    with st.chat_message("ai"):
        st.write("Thinking... Please wait.")

    # Run chain with updated message history
    response = chain.invoke(
        {"messages": st.session_state.messages},
        config={"configurable": {"session_id": "default"}}
    )

    # Remove the "Thinking..." message and display the AI response
    st.session_state.messages.append({"type": "ai", "content": response.content})

# --- Display previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["type"]):
        st.write(msg["content"])
