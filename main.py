import streamlit as st
import pinecone
import os
import time


def initRAG(vector_store):
 if 'code_executed' not in st.session_state:
  from langchain import hub
  from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import PromptTemplate, MessagesPlaceholder
  from langchain.chains import ConversationalRetrievalChain
  from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessage
   )
  from langchain.schema.messages import AIMessage, HumanMessage
  from langchain.memory import ConversationBufferMemory
  from langchain.agents.types import AgentType
  from langchain.agents import initialize_agent
  from langchain.tools import Tool
  from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
  from langchain.schema import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]  
  
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.15, max_tokens=512, openai_api_key=openai_api_key)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':1})
  condense_q_system_prompt = """When creating multiple choices quiz, set the choices in bullet points and put the answer below it with explanation. Example:
   What is the purpose of the document "General Standardization Development Guideline"?
   A) Translation 
   B) Bug fixing 
   C) Notification reference 
   D) Standardization development
   Answer: D) Standardization development
  """
  condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
  )
  condense_q_chain = condense_q_prompt | llm | StrOutputParser()
  qa_system_prompt = """
  When creating multiple choices quiz, set the choices in bullet points and put the answer below it with explanation. Example:
   What is the purpose of the document "General Standardization Development Guideline"?
   A) Translation 
   B) Bug fixing 
   C) Notification reference 
   D) Standardization development
   Answer: D) Standardization development
  {context}
  """
  qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
  def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]

  rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
  )
  st.session_state.rag_chain = rag_chain
  st.session_state.ai_msg_early = rag_chain.invoke({"question": 'hello', "chat_history": []})
  st.session_state.code_executed = True
  st.session_state.chat_history = []
  st.session_state.convo_history = []

def insert_or_fetch_embeddings(index_name):
  global isVector
  import pinecone
  from langchain.vectorstores import Pinecone
  from langchain.embeddings.openai import OpenAIEmbeddings

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]    
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

  if index_name in pinecone.list_indexes():
   print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
   pinecone.init(api_key='bbb687a2-cfb9-4b3e-8210-bece030f2776', environment='gcp-starter')
   vector_store = Pinecone.from_existing_index(index_name, embeddings)
   isVector = True
   print('OK')
  return vector_store

pinecone.init(api_key='bbb687a2-cfb9-4b3e-8210-bece030f2776', environment='gcp-starter')
chat_history = []
isVector = False
vector_store = None
question = None

def ask_and_get_answer_v3(question, chat_history=[]):
  from langchain.schema.messages import AIMessage, HumanMessage
  rag_chain = st.session_state.rag_chain
  ai_msg_early = st.session_state.ai_msg_early
  convo_history = st.session_state.convo_history
  ai_msg_early.content = ''
  ai_msg = rag_chain.stream({"question": question, "chat_history": chat_history})
  result_container = st.empty()
  # for convo in convo_history:
  #  st.write(f"**Question:** {convo.question}")
  #  st.write(f"**Answer:** {convo.answer}")
  for chunk in ai_msg:
    print(chunk.content, end="", flush=True)
    ai_msg_early.content += chunk.content
    result_container.markdown(f" **Answer:** {ai_msg_early.content}", unsafe_allow_html=True)
  # st.write(convo_history)
  st.session_state.convo_history.insert(0,{'question': question, 'answer': ai_msg_early.content})
  st.session_state.chat_history.extend([HumanMessage(content=question), ai_msg_early])

def extractWords(words):
 import re
 pattern = re.compile(r'func\s(.*?)\s+is not callable', re.DOTALL)
 match = pattern.search(str(words))
 if match:
  extracted_value = match.group(1).strip()
  return(extracted_value)
 else:
  print("Pattern not found.")

# Function to generate answers based on questions
def generate_answer(question):
    # Replace this with your logic to generate answers
    # For now, it's hardcoded answers
    if question.lower() == 'what is your name?':
        return "My name is ChatGPT."
    elif question.lower() == 'how does it work?':
        return "I use natural language processing to understand and generate text."
    else:
        return "I don't have an answer to that question."



def ask_and_get_answer_v2(vector_store, query):
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI
  from langchain.agents.types import AgentType
  from langchain.agents import initialize_agent
  from langchain.tools import Tool

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]
  answer = ""
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=openai_api_key)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
  chain=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  system_message = """
        answer in French."
        "if answer has some points, divide it to be bullet points"
        """
  tools = [
    Tool(
        name="qa-vet",
        func=chain.run(query),
        description="Useful when you need to answer vet questions",
    )
  ]
  executor = initialize_agent(
    agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    agent_kwargs={"system_message": system_message},
    verbose=False,
  )
  result = executor.run({'input': query, 'chat_history': []})
  return(result) 

def ask_and_get_answer(vector_store, query):
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]
  answer = ""
  llm = ChatOpenAI(model='gpt-3.5-turbo-0301', temperature=0, openai_api_key=openai_api_key)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
  chain=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

  answer = chain.run(query)
  return(answer)

def ask_with_memory(vector_store, question, chat_history=[]):
  from langchain.chains import ConversationalRetrievalChain
  from langchain.chat_models import ChatOpenAI

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]

  llm = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

  crc = ConversationalRetrievalChain.from_llm(llm, retriever)
  result = crc({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))

  return result, chat_history

# Streamlit app
def main():
    st.write(st.session_state.chat_history)
    from dotenv import load_dotenv
    global vector_store
    global chat_history
    global question
    api_config = st.secrets["api"]
    openai_api_key = api_config["openai_api_key"]
    load_dotenv()
    index_name = 'demo-langchain'
    vector_store = insert_or_fetch_embeddings(index_name)
    initRAG(vector_store)
    # Create a layout with two columns
    left_column, right_column = st.columns([1, 3])

    # Add an image to the left column
    left_column.image("rubo-bot.png", use_column_width=True)

    # Input field for the question
    with right_column:
        st.title("FE DEV GUIDELINES - Ask me questions...")
        question = st.text_input(f"Enter question: ")
        if st.button("Get Answer") or question:
            # Check if a question is provided
            if not question:
                st.warning("Please enter a question.")
            else:
             while True:
              try:
               st.write(f"**Question:** {question}")     
               ask_and_get_answer_v3(question, st.session_state.chat_history)
               break  
              except Exception as e:
               print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
 main()
