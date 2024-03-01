from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import openai
from PIL import Image
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from utils import *


st.set_page_config(
    page_title='WisdomBot',
    layout='wide'
)

col1, col2 = st.columns([1, 1])

with col1:
    logo = Image.open("conversative.png")
    st.image(logo, width=100)

with col2:
    st.subheader("_WisdomBot_")

st.header(" ")
with st.sidebar:
    selected = option_menu(menu_title='Main menu',options=['About', 'Chatbot','Community'], 
    icons=['house-fill', 'chat-fill','globe'],
    menu_icon="cast", default_index=0,)
    
if selected == "About":
    st.write(" ")
    st.header(":blue[Hello, I'm Wisdom]")
    st.write("""
             an Indigenous Languages AI Assistant designed to teach you Yoruba, Igbo, Hausa and more.
                 I was developed by Conversative AI (a subsidiary of Okubo Wisdom Legacies).
             """)
    
    st.header(":blue[About the Company]")
    st.write("""
             Conversative AI is an Artificial Intelligence Research company founded in 2023 by Wisdom Okubo. Its Parent Company, Okubo Wisdom Legacies, has a mission of curating Africa's Indigenous and Endangered Languages.

             """)
    
    st.write(" ")
    
    
    
    
    
elif selected == "Chatbot":
    
 st.write(" In Development. ")
  
elif selected == "Community":
    st.write("""
    Our community is a platform for our users to interact with other users on their language learning journey. It is also a place to get updates on new features, exclusive offers and other essential information. Users can take screenshots of their interactions with the Chatbot and send to the Forum for Feedback, Reviews and Development Purposes.
    
    """)
    


st.markdown(
    "`Created by` Conversative AI | 2024"
)
