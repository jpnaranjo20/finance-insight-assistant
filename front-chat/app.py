import os
import uuid
import requests
import logging
import json
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_local_storage import LocalStorage

LS_THREAD_ID_KEY = "fia_thread_id"

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
# hide_default_style = """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     [data-testid="stHeader"] {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_default_style, unsafe_allow_html=True)

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
        response = requests.post(f"{BACKEND_API_URL}/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        return None

# ===================== Session State Initialization =====================
# Browser-localStorage-backed persistence: chat survives page refresh without
# any server-side storage. Cleared via the New chat button or browser data wipe.
local_storage = LocalStorage()

if "hydrated" not in st.session_state:
    saved_thread_id = local_storage.getItem(LS_THREAD_ID_KEY)
    thread_id = saved_thread_id or str(uuid.uuid4())

    # Fetch history from the backend. Returns [] if the container restarted
    # (MemorySaver wiped), giving us the docker-compose-down-clears-memory behaviour.
    saved_messages = []
    if saved_thread_id:
        try:
            resp = requests.get(
                f"{BACKEND_API_URL}/history/{saved_thread_id}", timeout=5
            )
            if resp.ok:
                saved_messages = resp.json().get("messages", [])
        except Exception:
            pass

    st.session_state["messages"] = saved_messages
    st.session_state["thread_id"] = thread_id
    st.session_state["hydrated"] = True

def _persist_state():
    local_storage.setItem(
        LS_THREAD_ID_KEY,
        st.session_state["thread_id"],
        key="ls_set_thread_id",
    )

def _clear_persisted_state():
    local_storage.deleteItem(LS_THREAD_ID_KEY, key="ls_del_thread_id")

# ===================== Sidebar with Disclaimer =====================
st.sidebar.markdown(
    """
    ### Disclaimer
    This bot provides financial information for educational purposes only.
    It does not constitute professional financial advice.
    Consult a certified financial advisor before making investment decisions.
    """
)

# ===================== ChatBot Application =====================
st.title("💬 Finance Insight Assistant")

# Streamlit's markdown renderer interprets `$...$` as LaTeX math, which mangles
# dollar amounts in assistant responses. Escape them before display.
def _safe(text: str) -> str:
    return text.replace("$", "\\$") if isinstance(text, str) else text

def _split_sources(text: str):
    """Split the agent's strict single-line `Sources:` footer from the body."""
    if not isinstance(text, str) or "Sources:" not in text:
        return text, None
    body, _, sources = text.rpartition("Sources:")
    return body.rstrip(), sources.strip()

def _render_message(role: str, content: str):
    with st.chat_message(role):
        if role == "assistant":
            body, sources = _split_sources(content)
            st.write(_safe(body))
            if sources:
                with st.expander("📚 Sources"):
                    st.write(_safe(sources))
        else:
            st.write(_safe(content))

# Empty-state: example prompts to seed the conversation
EXAMPLE_PROMPTS = [
    "What were Apple's results in 2021?",
    "Current stock price of NVDA",
    "What was Adobe's revenue growth in 2020?",
    "Show me Microsoft's financial metrics"
]

# Resolve the prompt for this run BEFORE deciding whether to render the
# empty-state buttons — otherwise the buttons re-render alongside the new
# conversation and stale clicks get swallowed by the next rerun.
prompt = st.chat_input("Type your question here...")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})

if not st.session_state["messages"]:
    st.markdown("#### Try an example")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(ex, key=f"example_{i}", width='stretch'):
            st.session_state["pending_prompt"] = ex
            st.rerun()

for i, msg in enumerate(st.session_state["messages"]):
    if msg.get("plot_data"):
        try:
            st.plotly_chart(go.Figure(msg["plot_data"]), use_container_width=True, key=f"hist_chart_{i}")
        except Exception as e:
            logger.error(f"Error rendering chart from history: {e}")
    _render_message(msg["role"], msg["content"])

# Small toolbar above the input, only when there's a conversation to clear
if st.session_state["messages"]:
    _, clear_col = st.columns([5, 1])
    if clear_col.button("🆕 New chat", width='stretch'):
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state.pop("pending_prompt", None)
        _clear_persisted_state()
        st.rerun()

if prompt:
    with st.spinner("Preparing response..."):
        response = call_chat_api(st.session_state["messages"], st.session_state["thread_id"])

    if response and "response" in response:
        formatted_response = response["response"]
        assistant_msg = {"role": "assistant", "content": formatted_response}
        if response.get("has_plot"):
            assistant_msg["plot_data"] = response["plot_data"]
        st.session_state["messages"].append(assistant_msg)

        if response.get("has_plot"):
            try:
                fig = go.Figure(response["plot_data"])
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{len(st.session_state['messages'])}")
            except Exception as e:
                logger.error(f"Error displaying graph: {e}")
                st.error(f"Error displaying graph: {str(e)}")

        _render_message("assistant", formatted_response)
        _persist_state()

# ===================== Footer =====================
st.markdown("---")
st.caption(
    "Powered by GPT-4o-mini · LangGraph ReAct agent · ChromaDB · Yahoo Finance"
)
