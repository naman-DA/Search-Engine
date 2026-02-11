import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import (ArxivAPIWrapper,WikipediaAPIWrapper,)
from langchain_community.tools import (ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun,)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# STREAMLIT UI

st.set_page_config(page_title="Search Agent (Groq)", layout="centered")
st.title("Search Agent (Streaming + Citations + Memory)")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

# LLM (STREAMING ENABLED)

llm = ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant",temperature=0,streaming=True,)

# TOOLS

arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300,))

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300,))

search_tool = DuckDuckGoSearchRun(name="Search")

# TOOL ROUTER + CITATIONS

def tool_router(question: str) -> dict:
    q = question.lower()

    if "arxiv" in q or "paper" in q or "research" in q:
        result = arxiv_tool.invoke(question)
        return {"context": result, "source": "arXiv"}

    if "wikipedia" in q or "who is" in q or "define" in q:
        result = wiki_tool.invoke(question)
        return {"context": result, "source": "Wikipedia"}

    result = search_tool.invoke(question)
    return {"context": result, "source": "DuckDuckGo"}

# PROMPT (WITH MEMORY)

prompt = ChatPromptTemplate.from_messages([
    (
      "system",
      "You are a helpful AI assistant.\n"
      "Use the provided context to answer accurately.\n"
      "If context is insufficient, say so clearly.\n"
      "Keep answers concise."
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# LCEL PIPELINE (STREAMING SAFE)

chain = (RunnableLambda(
        lambda x: {"context": tool_router(x["question"])["context"],"question": x["question"],}
    )
    | prompt
    | llm
    | StrOutputParser()
)

# CHAT MEMORY

if "messages" not in st.session_state:
    st.session_state.messages = [{ "role": "assistant", "content": "Hi! I can search the web, Wikipedia, and arXiv. Ask me anything."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# USER INPUT

user_input = st.chat_input("Ask a question...")

if user_input:
  st.session_state.messages.append({"role": "user", "content": user_input})
  st.chat_message("user").write(user_input)

  with st.chat_message("assistant"):
    placeholder = st.empty()
    final_answer = ""

    for chunk in chain.stream({"question": user_input}):
      final_answer += chunk
      placeholder.markdown(final_answer)

      # Citation
      citation = tool_router(user_input)["source"]
      placeholder.markdown(final_answer + f"\n\n **Source:** {citation}")

  st.session_state.messages.append({"role": "assistant", "content": final_answer})