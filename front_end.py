# Importing Libraries
import time 
import streamlit as st
import os 
import joblib 
import fitz
import requests
from io import BytesIO
from PIL import Image
from streamlit_option_menu import option_menu
from pymongo import MongoClient
import text_summary
import TTS_STT
import os
# ---------------------------------------------------
# Session State Variables
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = str(time.time())
    # st.session_state.cgat_title = None
    st.session_state.chat_titles = {}
    st.session_state.speaker = False
    st.session_state.speakerText = None

# ---------------------------------------------------
# MongoDB Connection
client = MongoClient('localhost', 27017)

db = client.neuraldb
current_session = db[st.session_state.chat_id]

# ---------------------------------------------------
# AI Response Generator
def response_generator(query):
    print(f"Resposne generator has query {query}")
    endpoint='http://localhost:6000/response'
    response = requests.post(endpoint, json={"Prompt": query})
    print(response)
    if response.status_code == 200:
        output = response.json()
    else:
        output = f"Error: {response.status_code}"
    if output['image']==False:
        output['image']=[]
    else:
        output['image']=[]
        for i in os.listdir(os.path.join(os.getcwd(), 'images')):
            output['image'].append(os.path.join(os.getcwd(),i))
    print(output)
    return output

# ---------------------------------------------------
# Web App Settings 
ai_AVATAR_ICON='🤖'
chatting_AVATAR_ICON='💬'
st.set_page_config(page_title='Yamaha AI Hackathon', page_icon=ai_AVATAR_ICON, layout='centered')
st.title("Yamaha AI Hackathon " + chatting_AVATAR_ICON)

# ---------------------------------------------------
# Theme Settings
# hide_streamlit_style = """
# <style>     
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------
# Option Selection for Chatting vs Reasoning
options_placeholder = st.empty()
if 'mode' not in st.session_state:
    st.session_state.mode = 'CHAT'

# ---------------------------------------------------
# Sidebar
# with st.sidebar:
#     # Create a button for a new chat
#     if st.button('New Chat'):
#         new_chat_id = time.time()
#         st.session_state.chat_id = new_chat_id
#         st.session_state.chat_title = f'New Chat'
#         # st.rerun()
#     st.write('# Past Chats')
#     if st.session_state.get('chat_id') is None:
#         # Create buttons for past chats
#         for chat_id, chat_title in past_chats.items():
#             if st.button(chat_title):
#                 st.session_state.chat_id = chat_id
#                 st.session_state.chat_title = chat_title
#                 # Load the messages of the selected chat session
#                 try:
#                     st.session_state.messages = joblib.load(
#                         f'data/{st.session_state.chat_id}-st_messages'
#                     )
#                 except:
#                     st.session_state.messages=[]
#     else:
#         # If a chat is already selected, display its title and allow switching
#         st.write(f'Current Chat: {st.session_state.chat_title}')
        
#         # Create buttons for past chats (excluding the current one)
#         for chat_id, chat_title in past_chats.items():
#             if chat_id != st.session_state.chat_id:
#                 if st.button(chat_title):
#                     st.session_state.chat_id = chat_id
#                     st.session_state.chat_title = chat_title
#                     try:
#                         st.session_state.messages = joblib.load(
#                         f'data/{st.session_state.chat_id}-st_messages'
#                     )
#                     except:
#                         st.session_state.messages=[]
#     # Save new chats after a message has been sent to AI
#     # TODO: Give user a chance to name chat
#     st.session_state.chat_title = f'Chat Session - {st.session_state.chat_id}'

with st.sidebar:
    # Create a button for a new chat
    if st.button('New Chat'):
        new_chat_id = str(time.time())
        st.session_state.chat_id = new_chat_id
        # st.session_state.chat_title = None
        st.session_state.messages = []
        current_session = db[new_chat_id]
        st.rerun()

    st.write('# Past Chats')

    # Create buttons for past chats
    for chat_id in db.list_collection_names():
        try:
            if st.button(st.session_state.chat_titles[chat_id]):
                st.session_state.chat_id = chat_id
                # st.session_state.chat_title = f'Chat Session - {chat_id}'
                current_session = db[chat_id]
                st.session_state.messages = current_session.find_one()['chat_history']
                st.rerun()
        except:
            pass


# ---------------------------------------------------
# Main Section
st.session_state['rerun_flag'] = False

# ---------------------------------------------------
# PDF Upload Section
def pdf_upload():
    uploaded_file = st.file_uploader("Upload PDFs", type=['pdf'])

    if 'preview_clicked' not in st.session_state:
        st.session_state.preview_clicked = False
    if 'close_preview_clicked' not in st.session_state:
        st.session_state.close_preview_clicked = False

    image_placeholder = st.empty()

    if uploaded_file is not None:
        if st.button('Preview'):
            st.session_state.preview_clicked = True
            st.session_state.close_preview_clicked = False
            # Load the PDF
            pdf = fitz.open(stream=uploaded_file.read(), filetype='pdf')
            # Convert each page to an image and display it
            for page_num in range(min(len(pdf), 1)):
                page = pdf.load_page(page_num)
                pix = page.get_pixmap()
                # Convert Pixmap to bytes
                img_bytes = pix.tobytes()
                # Create PIL image from bytes
                pil_image = Image.open(BytesIO(img_bytes))
                st.session_state['pdf_image'] = pil_image
        if st.session_state.preview_clicked and not st.session_state.close_preview_clicked:
            if 'pdf_image' in st.session_state:
                image_placeholder.image(st.session_state['pdf_image'], caption='Page 1')
            if st.button('Close Preview'):
                # Clear the image
                image_placeholder.empty()
                st.session_state.close_preview_clicked = True
                st.session_state.preview_clicked = False
# pdf_upload()

# ---------------------------------------------------
# Load the chat history
if current_session.count_documents({}) != 0:
    st.session_state.messages = current_session.find_one()['chat_history']
else:
    st.session_state.messages = []

# ---------------------------------------------------
# Display the chat history
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            # if message['role'] == 'AI':
            #     # for image in message['images']:
            #     #     st.image(image)
            st.markdown(message['content'])

# ---------------------------------------------------
# Display the Chain of Thoughts
def display_chain_of_thoughts():
    # st.header("Reasoning Section")
    for message in st.session_state.messages:
        if message['role'] == 'AI':
            with st.chat_message(message['role']):
                st.markdown(message['cots'])
                for imagedbqw in message['images']:
                    print(imagedbqw)
                    st.image(imagedbqw)
            
        elif message['role'] == 'user':
            with st.chat_message(message['role']):
                st.write(message['content'])

# ---------------------------------------------------
# Chat Input Stream
if prompt :=st.chat_input("Enter the Prompt"):
    loading_message = st.empty()
    loading_message.info("Loading AI response...")
    
    # Generate the AI's response
    response = response_generator(prompt)
    loading_message.empty()

    # Append the user's message to the chat history        
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )

    # Append the AI's response to the chat history
    st.session_state.messages.append(
        dict(
            role='AI',
            content=response['output'],
            images=response['image'],
            cots=response['intermediate'],
        )
    )
    
    # Save the chat history in MongoDB
    if current_session.count_documents({}) == 0:
        db_structure = dict(
            user_prompt1 = prompt,
            ai_response1 = response,
            prompt_counter = 1,
            chat_history = st.session_state.messages,
        )
        current_session.insert_one(db_structure)
        st.session_state.speaker = True
        st.session_state.speakerText = response
        st.session_state.chat_titles[st.session_state.chat_id] = text_summary.extract_keywords(prompt, num_keywords=2)
        st.rerun()
    else:
        current_db = current_session.find_one()
        current_db['user_prompt'+str(current_db['prompt_counter']+1)] = prompt
        current_db['ai_response'+str(current_db['prompt_counter']+1)] = response
        current_db['prompt_counter'] += 1
        current_db['chat_history'] = st.session_state.messages
        current_session.update_one({}, {"$set": current_db})
    
if st.session_state.mode == "CHAT":
    display_chat_history()
    if st.session_state.speaker:
        if st.button('🔊'):
            # Play an audio file when button is pressed
            TTS_STT.SpeakText(st.session_state.speakerText['output'], savefile='audio_file', saveOnly=True)
            audio_file = open('audio_file.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            # st.rerun()

if st.session_state.mode == "REASON":
    display_chain_of_thoughts()
    if st.session_state.speaker:
        if st.button('🔊'):
            # Play an audio file when button is pressed
            TTS_STT.SpeakText(st.session_state.speakerText['intermediate'], savefile='audio_file', saveOnly=True)
            audio_file = open('audio_file.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            # st.rerun()

old_option = st.session_state.mode

option = option_menu(
    menu_title=None, 
    options=["CHAT", "REASON"], 
    icons=["💬", "🧠"], 
    orientation='horizontal',
    )

if st.button('Clear Memory'):
    current_session.delete_many({})
    st.rerun()

if st.button('Reset All'):
    for collection in db.list_collection_names():
        db.drop_collection(collection)
    st.rerun()


st.session_state.mode = option
if old_option != st.session_state.mode:
    st.rerun()

# st.write(st.session_state)