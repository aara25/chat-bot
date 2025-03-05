import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

load_dotenv()
import google.generativeai as genai


genai.configure(api_key=GEMINI_API_KEY)

try:
    models = genai.list_models()
    print([model.name for model in models])
except Exception as e:
    print("Error:", e)

if "chat_history" not in st.session_state: # session state - object that is going to keep all of the variables persistent throughtout the streamlit session 
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Bot", page_icon=":/")

st.title("Streaming Bot")

openai_api_key = os.getenv("OPENAI_API_KEY")


# conversation 

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# get response 

def get_response(query, chat_history):
    template = """
        You are a helpful assistant. Answer the following questions considering the  

        Chat History: {chat_history}

        User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", google_api_key=GEMINI_API_KEY)
    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo",  # You can change this to "gpt-3.5-turbo"
    #     temperature=0.7
    # )

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query,
    })



# user input

user_query =st.chat_input("Your Message")

if user_query is not None and user_query !="":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # ai_response = get_response(user_query, st.session_state.chat_history)
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))


    