import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env if exists
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template_text):
    return PromptTemplate(template=template_text, input_variables=["context", "question"])

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    user_input = st.chat_input("Pass your prompt here")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content': user_input})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know.
        Only answer based on the given context.

        Context: {context}
        Question: {question}

        Answer directly:
        """

        HF_TOKEN = os.environ.get("HF_TOKEN")
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY environment variable is not set!")
            return
        if not HF_TOKEN:
            st.error("HF_TOKEN environment variable is not set!")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            llm = ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=GROQ_API_KEY
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': user_input})
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant','content':result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
