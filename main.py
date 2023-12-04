import streamlit as st
import pinecone
import os
import time
import re


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
    HumanMessagePromptTemplate
  )
  from langchain.schema.messages import AIMessage, HumanMessage
  from langchain.memory import ConversationBufferMemory
  from langchain.agents.types import AgentType
  from langchain.agents import initialize_agent
  from langchain.tools import Tool
  from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
  from langchain.schema import StrOutputParser
  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.messages import AIMessage, HumanMessage
  import openai
  import os

  api_config = st.secrets["api"]
  openai_api_key = api_config["openai_api_key"]
  os.environ['OPENAI_API_KEY'] = openai_api_key
  fine_tuned_model_id = api_config["fine_tuned_model_id"]
  
  job = openai.fine_tuning.jobs.retrieve(fine_tuned_model_id)
  model_id = job.fine_tuned_model
  llm = ChatOpenAI(model=model_id, temperature=0.6, max_tokens=512, openai_api_key=openai_api_key)
  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3}, filters={'metadata': {'source': 'emarketing_textbook_download'}})
  condense_q_system_prompt = """ You are a quiz creator. Do not create quiz more than the user asks. Do not duplicate created quiz.
  """
  condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
  )
  condense_q_chain = condense_q_prompt | llm | StrOutputParser()
  qa_system_prompt = """You are a quiz creator. Do not create quiz more than the user asks. Do not duplicate created quiz.
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

def detect_and_create_quizzes(text, chat_history):
    from langchain.schema.messages import AIMessage, HumanMessage
     st.write(f"Goes Here")
    rag_chain = st.session_state.rag_chain
    ai_msg_early = st.session_state.ai_msg_early
    convo_history = st.session_state.convo_history
    ai_msg_early.content = ''
    number_of_quiz_per_iteration = 4
    keywords = ['generate', 'create', 'quiz', 'question']
    number_match = re.search(r'\b\d+\b', text)

    if any(keyword in text.lower() for keyword in keywords) and number_match:
        original_number = int(number_match.group())
        print(f"Original Number: {original_number}")
        st.write(f"Original Number: {original_number}")
        isFirst = True
        while original_number > 0:
            time.sleep(1)
            if isFirst:
             question = re.sub(r'\b\d+\b', str(min(original_number, number_of_quiz_per_iteration)), text)
             with get_openai_callback() as cb:
                ai_msg = rag_chain.stream({"question": question, "chat_history": chat_history})
                for chunk in ai_msg:
                  ai_msg_early.content += chunk.content
                  formatted_content = ai_msg_early.content.replace('\n', '<br>')
                  result_container.markdown(f" **Answer:** {formatted_content}", unsafe_allow_html=True)
                # print(f"{ai_msg.content}")
                # print(cb)
             isFirst = False
            else:
             question = f"Continue the number. Don't jump the number. Create {min(original_number, number_of_quiz_per_iteration)} again different quizzes"
             try:
               with get_openai_callback() as cb:
                ai_msg = rag_chain.stream({"question": question, "chat_history": chat_history})
                for chunk in ai_msg:
                  ai_msg_early.content += chunk.content
                  formatted_content = ai_msg_early.content.replace('\n', '<br>')
                  result_container.markdown(f" **Answer:** {formatted_content}", unsafe_allow_html=True)
                # print(f"{ai_msg.content}")
                # print(cb)
             except Exception as e:
              print(f"An error occurred: {e}")
            chat_history.extend([HumanMessage(content=question), ai_msg_early])
            print("")
            original_number -= number_of_quiz_per_iteration
    else:
      return False

def insert_or_fetch_embeddings(index_name):
  global isVector
  import pinecone
  from langchain.vectorstores import Pinecone, AstraDB
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
   vstore = AstraDB(
        embedding=embeddings,
        collection_name="astra_vector_demo",
        api_endpoint="https://288a909a-e845-4ebc-a371-c4fa12e5f11e-us-east1.apps.astra.datastax.com",
        token="AstraCS:nUUlGWiZPdBeIMoDgelSEJFk:eb5ebad132a13502a8ea60942c655b4d5b31baee3efc19df69ded0e326206b59",
    )
  return vstore

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
    formatted_content = ai_msg_early.content.replace('\n', '<br>')
    result_container.markdown(f" **Answer:** {formatted_content}", unsafe_allow_html=True)
  # st.write(convo_history)
  st.session_state.convo_history.insert(0,{'question': question, 'answer': ai_msg_early.content})
  st.session_state.chat_history.extend([HumanMessage(content=question), ai_msg_early])


# Streamlit app
def main():
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
        st.title("E-Marketing Textbook - Ask me questions...")
        question = st.text_input(f"Enter question: ")
        if st.button("Get Answer") or question:
            # Check if a question is provided
            if not question:
                st.warning("Please enter a question.")
            else:
             # while True:
              try:
               st.write(f"**Question:** {question}")
               detect_and_create_quizzes(question,  st.session_state.chat_history)
                # ask_and_get_answer_v3(question, st.session_state.chat_history)
               # break  
              except Exception as e:
               print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
 main()
