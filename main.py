import streamlit as st
import pinecone
import os
import time

pinecone.init(api_key='bbb687a2-cfb9-4b3e-8210-bece030f2776', environment='gcp-starter')
chat_history = []
isVector = False
vector_store = None

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
    from dotenv import load_dotenv
    global vector_store
    global chat_history
    api_config = st.secrets["api"]
    openai_api_key = api_config["openai_api_key"]
    load_dotenv()
    index_name = 'demo-langchain'

    if vector_store is None:
     vector_store = insert_or_fetch_embeddings(index_name)
        
    # Create a layout with two columns
    left_column, right_column = st.columns([1, 3])

    # Add an image to the left column
    left_column.image("rubo-bot.png", use_column_width=True)

    # Input field for the question
    with right_column:
        st.title("CPEB 2024 - Posez-nous vos questions...")
        question = st.text_input(f"Entrez votre question: ")
        if st.button("Obtenir la réponse"):
            # Check if a question is provided
            if not question:
                st.warning("Please enter a question.")
            else:
                # Generate and display the answer
                result = ask_and_get_answer(vector_store, question + " au format puces")
                if '403' in result:
                 result2 = ask_and_get_answer(vector_store, question + " au format puces")
                 st.write(f"**Question:** {question}")
                 st.write(f"**Answer:** {result2}")
                else:
                 st.write(f"**Question:** {question}")
                 st.write(f"**Answer:** {result}")



if __name__ == "__main__":
 main()
