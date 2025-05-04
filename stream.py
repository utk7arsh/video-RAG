import streamlit as st
from PIL import Image
from utils import encode_image
from gradio_utils import get_default_rag_chain, split_video

# --- Setup ---
st.set_page_config(page_title="Video RAG Chat", layout="wide")
st.title("🎬 Multimodal RAG: Chat with Videos")

# --- State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "frame_path" not in st.session_state:
    st.session_state.frame_path = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_default_rag_chain()

# --- UI: Query Input ---
dropdown_list = [
    "What is the name of one of the astronauts?",
    "An astronaut's spacewalk",
    "What does the astronaut say?",
]
query = st.selectbox("Sample Queries", dropdown_list)
user_query = st.text_input("Or enter your own query:", value=query)

if st.button("Ask"):
    # --- RAG Pipeline ---
    rag_chain = st.session_state.rag_chain
    response = rag_chain.invoke(user_query)
    answer = response['final_text_output']
    meta = response['input_to_lvlm']['metadata']
    frame_path = response['input_to_lvlm']['image']
    transcript = meta['transcript']
    video_path = meta.get('video_path')
    timestamp = meta.get('mid_time_ms')

    # Optionally split video for context
    if video_path and timestamp:
        subvideo_path = split_video(video_path, timestamp)
        st.session_state.video_path = subvideo_path
    else:
        st.session_state.video_path = video_path

    st.session_state.frame_path = frame_path
    st.session_state.transcript = transcript
    st.session_state.history.append((user_query, answer))

# --- UI: Display Results ---
if st.session_state.frame_path:
    st.image(st.session_state.frame_path, caption="Extracted Frame", use_column_width=True)
if st.session_state.video_path:
    st.video(st.session_state.video_path)
if st.session_state.transcript:
    st.markdown(f"**Transcript:** {st.session_state.transcript}")

# --- Chat History ---
if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**RAG:** {a}")

if st.button("Clear History"):
    st.session_state.history = []
    st.session_state.frame_path = None
    st.session_state.video_path = None
    st.session_state.transcript = None