import os
import uuid
import requests
import logging
import json
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="Finance Insight Assistant",
    page_icon="💬",
    layout="centered",  # Centered layout, not full-width
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    }
)

# ===================== Hide Default Elements =====================
# Hide the menu (three dots), the footer, and the header (which includes the Deploy button)
hide_default_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    </style>
"""
st.markdown(hide_default_style, unsafe_allow_html=True)

# ===================== Load Environment Variables =====================
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://backend_api:8000")

# ===================== Functions =====================

def check_credentials(username, password, filepath="users.json"):
    """
    Verifies credentials by comparing them with those stored in the JSON file.
    It is assumed that the JSON has the structure:
      {
         "user1": "password1",
         "user2": "password2",
         ...
      }
    """
    try:
        with open(filepath, "r") as f:
            users = json.load(f)
    except Exception as e:
        st.error("Error loading the users file.")
        logger.error(f"Error reading {filepath}: {e}")
        return False
    return username in users and users[username] == password

def call_chat_api(messages, thread_id):
    try:
        payload = {
            "messages": messages,
            "config": {
                "configurable": {
                    "thread_id": thread_id
                }
            }
        }
        response = requests.post(f"{BACKEND_API_URL}/chat", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        return None

# ===================== Session State Initialization =====================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
    
# ===================== Sidebar with Disclaimer =====================
st.sidebar.markdown(
    """
    ### Disclaimer
    This bot provides financial information for educational purposes only.
    It does not constitute professional financial advice.
    Consult a certified financial advisor before making investment decisions.
    """
)

# ===================== Custom Navbar =====================
navbar = """
<style>
.custom-navbar {
  position: fixed;
  top: 10px;
  right: 10px;
  background-color: #333;
  padding: 8px 12px;
  border-radius: 5px;
  z-index: 1000;
  font-family: sans-serif;
  color: white;
}
</style>
<div class="custom-navbar">
  Finance Insight Assistant
</div>
"""
st.markdown(navbar, unsafe_allow_html=True)

# Add top margin so that the content is not hidden by the navbar
st.markdown("<div style='margin-top:70px;'></div>", unsafe_allow_html=True)

# ===================== ChatBot Application =====================
st.title("💬 Finance Insight Assistant")
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Type your question here...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Preparing response..."):
        response = call_chat_api(st.session_state["messages"], st.session_state["thread_id"])
    
    if response and "response" in response:
        formatted_response = response["response"]
        st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
        st.chat_message("assistant").write(formatted_response)
            
        # Optional: If there is plot data in the response, display it
        if response.get("has_plot"):
            try:
                fig = go.Figure(response["plot_data"])
                st.plotly_chart(fig)
            except Exception as e:
                logger.error(f"Error displaying graph: {e}")
                st.error(f"Error displaying graph: {str(e)}")
