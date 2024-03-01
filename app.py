from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from agents.simple_agent import initialize_simple_agent
from agents.react_agent import initialize_react_agent

from tavily import TavilyClient
import getpass
import os
import pprint

# If API key is not set in the environment, prompt for it
#os.environ["TAVILY_API_KEY"] = getpass.getpass()

import streamlit as st

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)

def handle_submit_agent(agent_executor, message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Chroma.
    """

    # Handle the response
    with st.spinner('Thinkin\' about it...'):
        response = agent_executor.invoke({"input": message})
        print(response)
        write_message('assistant', response)

def handle_submit_simple(llm_chain, message, db, search_client):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Chroma.
    """

    # Handle the response
    with st.spinner('Thinkin\' about it...'):
        search_response = search_client.search(message, search_depth="advanced")["results"]
        print("Search response:")
        pprint.pprint(search_response)
        db_response = db.similarity_search(message)
        print("Vector DB response:")
        pprint.pprint(db_response)
        context = db_response[0].page_content
        for result in search_response:
            context += " \n "+result["content"]
        response = llm_chain.invoke({"context": context,"question": message})
        pprint.pprint(response)
        write_message('assistant', response["text"])

keep_it_simple = True

llm = Ollama(model="dolphin-mistral")

embeddings = OllamaEmbeddings(
    model = "llama2",
    num_thread = 4,
    num_gpu = 1
)

vec_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

st.set_page_config("Sparkplug", page_icon=":mechanic:")

agent_executor = None
if keep_it_simple:
    agent_executor = initialize_simple_agent(llm)
else:
    agent_executor = initialize_react_agent(llm, vec_db)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Yo, what do you want?"},
    ]

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if user_prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', user_prompt)

    # Generate a response
    if keep_it_simple:
        handle_submit_simple(agent_executor, user_prompt, vec_db, search_client)
    else:
        handle_submit_agent(agent_executor, user_prompt)
