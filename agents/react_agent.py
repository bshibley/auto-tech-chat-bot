from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

from agents.simple_agent import initialize_simple_agent

from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults


def initialize_react_agent(llm, chroma_db):
    chroma_retriever = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=chroma_db.as_retriever()
    )

    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    simple_chat = initialize_simple_agent(llm)

    tools = [
        Tool.from_function(
            name="General Chat",
            description="For when you need to chat. The question will be a string. Return a string.",
            func=simple_chat.run,
            return_direct=True
        ),
        Tool.from_function(
            name='Web Search',
            description='Use to search for the question\'s answer on the internet. Return the answer as a string.',
            func=tavily_tool.run,
            return_direct=False
        ),
        Tool.from_function(
            name="Vector Search Index",
            description="Use to look up information about an automotive repair or vehicle tuning question. Return useful context as a string.",
            func = chroma_retriever.run,
            return_direct=False
        )
    ]

    agent_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, agent_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=agent_memory,
        max_execution_time=60,
        handle_parsing_errors=True,
        )