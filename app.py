import base64
from html import escape
import os
from pathlib import Path
import re
import time
import uuid

import streamlit as st

from rag_pipeline import COLLECTION, DB_PATH, ask_question, build_vector_db, ensure_vector_db


st.set_page_config(
    page_title="Cosmetic Chemical Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body {
    overflow-y: auto !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _has_html_artifact(text: str) -> bool:
    if not isinstance(text, str):
        return True

    stripped = text.strip().lower()
    if stripped == "</div>":
        return True

    artifact_patterns = [
        r"^</?div[^>]*>$",
        r"<div[^>]*>",
        r"</div>",
        r"<span[^>]*>",
        r"</span>",
        r"<br\s*/?>",
        r"&nbsp;",
    ]
    return any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in artifact_patterns)


def _clear_corrupted_chats_if_needed() -> None:
    chats = st.session_state.get("chats")
    if not isinstance(chats, dict):
        st.session_state.chats = {}
        return

    corrupted_found = False
    for chat in chats.values():
        messages = chat.get("messages", []) if isinstance(chat, dict) else []
        for message in messages:
            if not isinstance(message, dict):
                corrupted_found = True
                break
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if _has_html_artifact(content):
                    corrupted_found = True
                    break
        if corrupted_found:
            break

    if corrupted_found:
        st.session_state.chats = {}
        st.session_state.pop("current_chat_id", None)


def _bootstrap_vector_db_once() -> None:
    if st.session_state.get("_db_bootstrapped"):
        return

    # Use the improved validation function that checks collection count
    ensure_vector_db(csv_path="data/chemicals.csv", db_path=DB_PATH, collection_name=COLLECTION)

    st.session_state._db_bootstrapped = True


def find_video_path() -> str | None:
    candidates = [
        os.path.join(os.getcwd(), "static", "chem_loop.mp4"),
        os.path.join(os.path.dirname(__file__), "static", "chem_loop.mp4"),
        os.path.join("static", "chem_loop.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def file_to_b64(path: str) -> str:
    with open(path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def new_chat() -> None:
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [
            {
                "role": "assistant",
                "content": "Hi! Ask me about cosmetic products, brands, companies, or chemicals.",
                "sources": [],
            }
        ],
    }


def set_chat_title_if_needed(chat_id: str) -> None:
    chat = st.session_state.chats[chat_id]
    if chat["title"] == "New chat":
        for message in chat["messages"]:
            if message["role"] == "user":
                text = message["content"].strip()
                chat["title"] = (text[:28] + "...") if len(text) > 28 else text
                break


if "chats" not in st.session_state:
    st.session_state.chats = {}

_clear_corrupted_chats_if_needed()
_bootstrap_vector_db_once()

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    new_chat()


def _safe_label(value: str) -> str:
    text = (value or "").strip()
    return escape(text) if text else "Unknown"


def render_assistant_message(message: dict) -> None:
    # Get and clean the answer text
    raw_answer = (message.get("content") or "").strip() or "No answer generated."
    
    # Remove any stray HTML tags that might have leaked in
    import re
    cleaned_answer = re.sub(r'</?div[^>]*>', '', raw_answer)
    cleaned_answer = re.sub(r'</?span[^>]*>', '', cleaned_answer)
    cleaned_answer = cleaned_answer.strip()
    
    sources = message.get("sources") or []

    # Use Streamlit's native chat message component
    with st.chat_message("assistant"):
        # Render the answer using st.markdown
        st.markdown(cleaned_answer)
        
        # Render sources if available
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                product = source.get("product", "Unknown")
                brand = source.get("brand", "Unknown")
                chemical = source.get("chemical", "Unknown")
                company = source.get("company", "Unknown")
                
                # Format as simple bullet points
                source_text = f"• **{product}** | Brand: {brand} | Chemical: {chemical} | Company: {company}"
                st.markdown(source_text)


video_path = find_video_path()
if video_path:
    video_b64 = file_to_b64(video_path)
    st.markdown(
        f"""
<style>
.stApp {{
    background: transparent !important;
}}

video {{
    position: fixed;
    top: 0;
    left: 0;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    z-index: -1;
}}

.overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 0;
}}
</style>

<video autoplay muted loop playsinline>
    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>

<div class="overlay"></div>
""",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## Chats")

    if st.button("+ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("---")

    chat_ids = list(st.session_state.chats.keys())[::-1]
    for chat_id in chat_ids:
        title = st.session_state.chats[chat_id]["title"]
        is_current = chat_id == st.session_state.current_chat_id
        label = f"> {title}" if is_current else title

        if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()


st.markdown(
    """
<div style="
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:white;
    margin-top:20px;
    margin-bottom:20px;
">
    Cosmetic Chemical Intelligence
</div>
""",
    unsafe_allow_html=True,
)


current_id = st.session_state.current_chat_id
current_chat = st.session_state.chats[current_id]

for message in current_chat["messages"]:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:14px 0;">
                <div style="
                    max-width:70%;
                    background:#1f242b;
                    padding:16px 20px;
                    border-radius:16px;
                    color:white;
                    font-size:18px;
                    font-weight:600;
                    box-shadow:0 4px 20px rgba(0,0,0,0.35);
                    white-space:pre-wrap;
                ">
                    {escape(message['content'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_assistant_message(message)


prompt = st.chat_input("Ask a question about cosmetics data...")

if prompt:
    current_chat["messages"].append({"role": "user", "content": prompt})
    set_chat_title_if_needed(current_id)

    with st.spinner("Generating answer..."):
        time.sleep(0.15)
        result = ask_question(prompt)
        answer_text = (result.get("answer") or "").strip() or "No answer generated."
        sources = result.get("sources") or []

    current_chat["messages"].append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
    st.rerun()
