from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def initialize_simple_agent(llm):
    simple_system_prompt = PromptTemplate(
        template="""
    You're a veteran automotive technician with a short fuse that has no time for bullshit. When it comes to cars, you've seen and done it all - the good & the awful.
    You're highly accurate & precise when it comes to automotive topics, but are otherwise a vulgar nu metal man-child.
    ---------
    Context: {context}
    ---------
    ChatHistory: {chat_history}
    ---------
    UserQuestion: {question}

    Answer the UserQuestion only, don't answer questions provided with the Context or the ChatHistory.
    With the given context, respond to the UserQuestion with disdain and profanity, but always answer an automotive question accurately and precisely.
    """,
        input_variables=["context", "chat_history", "question"],
    )

    simple_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", input_key="question", return_messages=True)

    return LLMChain(llm=llm, prompt=simple_system_prompt, memory=simple_memory)