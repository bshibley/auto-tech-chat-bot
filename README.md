## Automotive Technician Chat Bot

The Auto Tech Chat Bot is specialized in answering your automotive repair & tuning questions.

As a veteran mechanic, he is very wise, but also jaded. You have been warned!

### How to Use

Prior to use, the chroma database must be populated. Use the 
accompanying `chroma_populate.ipynb` notebook to create the database.

Ollama is used for the LLM. Run Ollama with Docker, e.g.:

`sudo docker run -d --gpus=all --runtime=nvidia -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`

A Tavily API key is also required. It can be obtained from: https://tavily.com. 
The API key must be set in the system environment, e.g.:

`export TAVILY_API_KEY=[INSERT YOUR API KEY HERE]`

Launch the app using streamit (optionally include the Tavily API key):

`TAVILY_API_KEY=[INSERT YOUR API KEY HERE] streamlit run app.py`

### Technical Details

The dolphin-mistral 7B model is used due to its speed and lack of censorship.

There are two agents currently defined. The `keep_it_simple` variable is used 
select between them.

#### Simple Agent

This agent is a simple LLM chain. A single invocation is used to generate the response.
Context is generated beforehand through web or database search.

#### ReAct Agent

This agent uses a react agent with tools such as Tavily web search and 
Chroma database search to produce an answer. It is more dynamic and less 
dependable in its current form.