import os
import base64
import uuid
import time
import streamlit as st

from rag_pipeline import ask_question


# ---------------- Page config ----------------
st.set_page_config(
    page_title="Cosmetic Chemical Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Allow Scrolling ----------------
st.markdown("""
<style>
html, body {
    overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Video helpers ----------------
def find_video_path():
    candidates = [
        os.path.join(os.getcwd(), "static", "chem_loop.mp4"),
        os.path.join(os.path.dirname(__file__), "static", "chem_loop.mp4"),
        os.path.join("static", "chem_loop.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------- Chat helpers ----------------
def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [
            {"role": "assistant", "content": "Hi! Ask me about chemicals in cosmetic products 😊"}
        ],
    }
    st.session_state.current_chemical = None


def set_chat_title_if_needed(chat_id: str):
    chat = st.session_state.chats[chat_id]
    if chat["title"] == "New chat":
        for m in chat["messages"]:
            if m["role"] == "user":
                t = m["content"].strip()
                chat["title"] = (t[:28] + "…") if len(t) > 28 else t
                break


# ---------------- Session state ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    new_chat()

if "current_chemical" not in st.session_state:
    st.session_state.current_chemical = None

# ---------------- Result formatting ----------------
def format_result(result: dict) -> str:
    mode = result.get("mode", "")
    header = result.get("answer", "").strip()
    items = result.get("items", []) or []

    output = header

    if mode == "chemical_to_products" and items:
        lines = []
        for it in items:
            lines.append(
                f"- {it.get('product','—')} — "
                f"{it.get('brand','—')} "
                f"({it.get('company','—')}) | "
                f"{it.get('primary_category','—')} / "
                f"{it.get('sub_category','—')}"
            )
        output += "\n\n" + "\n".join(lines)

    return output

# ---------------- Background video ----------------
video_path = find_video_path()
if video_path:
    b64 = file_to_b64(video_path)
    st.markdown(
        f"""
<video style="
    position:fixed;
    top:0;
    left:0;
    width:100vw;
    height:100vh;
    object-fit:cover;
    z-index:0;
" autoplay muted loop playsinline>
  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
</video>

<div style="
    position:fixed;
    inset:0;
    background:rgba(0,0,0,0.45);
    z-index:1;
"></div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<div style="position:relative; z-index:2;">', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 💬 Chats")

    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("---")

    chat_ids = list(st.session_state.chats.keys())[::-1]

    for cid in chat_ids:
        title = st.session_state.chats[cid]["title"]
        is_current = (cid == st.session_state.current_chat_id)
        label = f"➡️ {title}" if is_current else title

        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat_id = cid
            st.rerun()


# ---------------- Heading ----------------
st.markdown("""
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
""", unsafe_allow_html=True)


# ---------------- Chat Display ----------------
current_id = st.session_state.current_chat_id
current_chat = st.session_state.chats[current_id]

for m in current_chat["messages"]:

    if m["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:14px 0;">
                <div style="
                    max-width:65%;
                    background:#1f242b;
                    padding:18px 22px;
                    border-radius:18px;
                    color:white;
                    font-size:20px;
                    font-weight:600;
                    box-shadow:0 4px 20px rgba(0,0,0,0.4);
                ">
                    {m["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-start; margin:14px 0;">
                <div style="
                    max-width:65%;
                    background:#08202c;
                    padding:18px 22px;
                    border-radius:18px;
                    color:white;
                    font-size:20px;
                    line-height:1.6;
                    box-shadow:0 4px 20px rgba(0,0,0,0.4);
                ">
                    {m["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------- Chat Input ----------------
prompt = st.chat_input("Ask anything…")

if prompt:
    current_chat["messages"].append({"role": "user", "content": prompt})
    set_chat_title_if_needed(current_id)

    with st.spinner("Thinking..."):
        time.sleep(0.2)

        result = ask_question(
            prompt,
            current_chemical=st.session_state.current_chemical
        )

        if isinstance(result, dict) and result.get("chemical"):
            st.session_state.current_chemical = result["chemical"]

        answer = format_result(result)

    current_chat["messages"].append({"role": "assistant", "content": answer})
    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)