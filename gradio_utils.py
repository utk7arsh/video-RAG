"""
Gradio utilities for the Video RAG project.
This module provides the Gradio interface and related utilities for the video RAG application.
"""

import os
import io
import sys
import time
import base64
import dataclasses
from pathlib import Path
from enum import auto, Enum
from typing import List, Tuple, Any, Optional, Dict

import gradio as gr
import lancedb
from PIL import Image
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from moviepy.video.io.VideoFileClip import VideoFileClip

from utils import (
    openai_gpt4v_conv,
    load_json_file,
    encode_image,
    Conversation,
    lvlm_inference_with_conversation
)
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from mm_rag.MLM.lvlm import LVLM
from mm_rag.MLM.client import OpenAIGPT4VClient

# Constants
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
PROMPT_TEMPLATE = """The transcript associated with the image is '{transcript}'. {user_query}"""
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
DEFAULT_TABLE_NAME = "demo_tbl"

# Theme Configuration
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", 
        c50="#eff6ff", c500="#0054ae", c600="#00377c", c700="#00377c", 
        c800="#1e40af", c900="#1e3a8a", c950="#0a0c2b"
    ),
    secondary_hue=gr.themes.Color(
        c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", 
        c50="#eff6ff", c500="#0054ae", c600="#0054ae", c700="#0054ae", 
        c800="#1e40af", c900="#1e3a8a", c950="#1d3660"
    )
).set(
    body_background_fill_dark='*primary_950',
    body_text_color_dark='*neutral_300',
    border_color_accent='*primary_700',
    border_color_accent_dark='*neutral_800',
    block_background_fill_dark='*primary_950',
    block_border_width='2px',
    block_border_width_dark='2px',
    button_primary_background_fill_dark='*primary_500',
    button_primary_border_color_dark='*primary_500'
)

# CSS Configuration
CSS = '''
    @font-face {
        font-family: IntelOne;
        src: url("/file=./assets/intelone-bodytext-font-family-regular.ttf");
    }
    .gradio-container {background-color: #0a0c2b}
    table {
      border-collapse: collapse;
      border: none;
    }
'''

class SeparatorStyle(Enum):
    """Different separator styles for conversation formatting."""
    SINGLE = auto()

@dataclasses.dataclass
class GradioInstance:
    """
    A class that maintains conversation history and state for the Gradio interface.
    
    Attributes:
        system (str): System message for the conversation
        roles (List[str]): List of possible roles in the conversation
        messages (List[List[str]]): List of message pairs [role, content]
        offset (int): Offset for message history
        sep_style (SeparatorStyle): Style of message separation
        sep (str): Separator string between messages
        sep2 (Optional[str]): Secondary separator string
        version (str): Version identifier
        path_to_img (Optional[str]): Path to current image
        video_title (Optional[str]): Title of current video
        path_to_video (Optional[str]): Path to current video
        caption (Optional[str]): Current caption
        mm_rag_chain (Any): RAG chain instance
        skip_next (bool): Flag to skip next generation
    """
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"
    sep2: Optional[str] = None
    version: str = "Unknown"
    path_to_img: Optional[str] = None
    video_title: Optional[str] = None
    path_to_video: Optional[str] = None
    caption: Optional[str] = None
    mm_rag_chain: Any = None
    skip_next: bool = False

    def _template_caption(self) -> str:
        """Format caption with template if it exists."""
        return f"The caption associated with the image is '{self.caption}'." if self.caption else ""

    def get_prompt_for_rag(self) -> str:
        """Get prompt for RAG processing."""
        if len(self.messages) != 2:
            raise ValueError("Current conversation should have exactly 2 messages")
        if self.messages[1][1] is not None:
            raise ValueError("First response message should be None")
        return self.messages[0][1]

    def get_conversation_for_lvlm(self) -> Conversation:
        """Get conversation formatted for LVLM processing."""
        pg_conv = openai_gpt4v_conv.copy()
        if self.path_to_img:
            b64_img = encode_image(self.path_to_img)
            for i, (role, msg) in enumerate(self.messages[self.offset:]):
                if msg is None:
                    break
                if i == 0:
                    pg_conv.append_message(openai_gpt4v_conv.roles[0], [msg, b64_img])
                elif i == len(self.messages[self.offset:]) - 2:
                    pg_conv.append_message(role, [PROMPT_TEMPLATE.format(
                        transcript=self.caption, user_query=msg
                    )])
                else:
                    pg_conv.append_message(role, [msg])
        return pg_conv

    def append_message(self, role: str, message: str) -> None:
        """Append a new message to the conversation."""
        self.messages.append([role, message])

    def get_images(self, return_pil: bool = False) -> List[str]:
        """Get list of image paths."""
        return [self.path_to_img] if self.path_to_img else []

    def to_gradio_chatbot(self) -> List[List[str]]:
        """Convert conversation to Gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if isinstance(msg, tuple):
                    msg, image, _ = msg
                    img_str = self._process_image_for_chat(image)
                    ret.append([img_str + msg.replace('<image>', '').strip(), None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def _process_image_for_chat(self, image: Image.Image) -> str:
        """Process image for chat display."""
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 800, 400
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
            
        image = image.resize((W, H))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'

    def copy(self) -> 'GradioInstance':
        """Create a deep copy of the instance."""
        return GradioInstance(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            mm_rag_chain=self.mm_rag_chain,
        )

    def dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary format."""
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "path_to_img": self.path_to_img,
            "video_title": self.video_title,
            "path_to_video": self.path_to_video,
            "caption": self.caption,
        }

    def get_path_to_subvideos(self) -> Optional[str]:
        """Get path to subvideos based on current state."""
        if self.video_title and self.path_to_img:
            info = video_helper_map[self.video_title]
            vid_index = self.path_to_img.split('/')[-1].split('_')[-1].replace('.jpg', '')
            return os.path.join(info['path'], f"{info['prefix']}{vid_index}.mp4")
        return self.path_to_video

def split_video(
    video_path: str,
    timestamp_in_ms: int,
    output_video_path: str = "./shared_data/splitted_videos",
    output_video_name: str = "video_tmp.mp4",
    play_before_sec: int = 3,
    play_after_sec: int = 3
) -> str:
    """
    Split video at a specific timestamp.
    
    Args:
        video_path: Path to input video
        timestamp_in_ms: Timestamp in milliseconds
        output_video_path: Directory for output video
        output_video_name: Name of output video file
        play_before_sec: Seconds to include before timestamp
        play_after_sec: Seconds to include after timestamp
        
    Returns:
        Path to split video file
    """
    timestamp_in_sec = int(timestamp_in_ms / 1000)
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    output_video = os.path.join(output_video_path, output_video_name)
    
    with VideoFileClip(video_path) as video:
        duration = video.duration
        start_time = max(timestamp_in_sec - play_before_sec, 0)
        end_time = min(timestamp_in_sec + play_after_sec, duration)
        new = video.subclip(start_time, end_time)
        new.write_videofile(output_video, audio_codec='aac')
    
    return output_video

def get_default_rag_chain():
    """
    Create and configure the default RAG chain.
    
    Returns:
        Configured RAG chain instance
    """
    # Initialize vectorstore
    db = lancedb.connect(LANCEDB_HOST_FILE)
    embedder = BridgeTowerEmbeddings()
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE,
        embedding=embedder,
        table_name=DEFAULT_TABLE_NAME
    )
    retriever_module = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 1}
    )

    # Initialize LVLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        import streamlit as st
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAIGPT4VClient(api_key=api_key)
    lvlm_inference_module = LVLM(client=client)
    
    def prompt_processing(input: Dict[str, Any]) -> Dict[str, Any]:
        """Process prompt for RAG chain."""
        retrieved_results, user_query = input['retrieved_results'], input['user_query']
        retrieved_result = retrieved_results[0]
        metadata = retrieved_result.metadata['metadata']
        
        return {
            'prompt': PROMPT_TEMPLATE.format(
                transcript=metadata['transcript'],
                user_query=user_query
            ),
            'image': metadata['extracted_frame_path'],
            'metadata': metadata,
        }

    prompt_processing_module = RunnableLambda(prompt_processing)
    
    return (
        RunnableParallel({
            "retrieved_results": retriever_module,
            "user_query": RunnablePassthrough()
        })
        | prompt_processing_module
        | RunnableParallel({
            'final_text_output': lvlm_inference_module,
            'input_to_lvlm': RunnablePassthrough()
        })
    )

def get_gradio_instance(mm_rag_chain: Optional[Any] = None) -> GradioInstance:
    """
    Create a new GradioInstance with optional RAG chain.
    
    Args:
        mm_rag_chain: Optional RAG chain instance
        
    Returns:
        New GradioInstance
    """
    if mm_rag_chain is None:
        mm_rag_chain = get_default_rag_chain()
        
    return GradioInstance(
        system="",
        roles=openai_gpt4v_conv.roles,
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n",
        path_to_img=None,
        video_title=None,
        caption=None,
        mm_rag_chain=mm_rag_chain,
    )

def clear_history(state: GradioInstance, request: gr.Request) -> Tuple[GradioInstance, List[List[str]], str, None, gr.Button]:
    """Clear conversation history."""
    state = get_gradio_instance(state.mm_rag_chain)
    return (state, state.to_gradio_chatbot(), "", None, gr.Button(interactive=False))

def add_text(state: GradioInstance, text: str, request: gr.Request) -> Tuple[GradioInstance, List[List[str]], str, gr.Button]:
    """Add text to conversation."""
    if not text:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", gr.Button())

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", gr.Button(interactive=False))

def http_bot(state: GradioInstance, request: gr.Request) -> Tuple[GradioInstance, List[List[str]], Optional[str], gr.Button]:
    """Handle bot responses."""
    if state.skip_next:
        path_to_sub_videos = state.get_path_to_subvideos()
        yield (state, state.to_gradio_chatbot(), path_to_sub_videos, gr.Button())
        return

    if len(state.messages) == state.offset + 2:
        new_state = get_gradio_instance(state.mm_rag_chain)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    all_images = state.get_images()
    is_very_first_query = len(all_images) == 0
    
    prompt_or_conversation = (
        state.get_prompt_for_rag() if is_very_first_query
        else state.get_conversation_for_lvlm()
    )
    
    executor = state.mm_rag_chain if is_very_first_query else lvlm_inference_with_conversation
    
    state.messages[-1][-1] = "▌"
    path_to_sub_videos = state.get_path_to_subvideos()
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos, gr.Button(interactive=False))

    try:
        if is_very_first_query:
            response = executor.invoke(prompt_or_conversation)
            message = response['final_text_output']
            if 'metadata' in response['input_to_lvlm']:
                metadata = response['input_to_lvlm']['metadata']
                if state.path_to_img is None and 'image' in response['input_to_lvlm']:
                    state.path_to_img = response['input_to_lvlm']['image']
                if state.path_to_video is None and 'video_path' in metadata:
                    video_path = metadata['video_path']
                    mid_time_ms = metadata['mid_time_ms']
                    state.path_to_video = split_video(video_path, mid_time_ms)
                if state.caption is None and 'transcript' in metadata:
                    state.caption = metadata['transcript']
            else:
                raise ValueError("Response format is invalid")
        else:
            message = executor(prompt_or_conversation)
            
    except Exception as e:
        print(e)
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield (state, state.to_gradio_chatbot(), None, gr.Button(interactive=True))
        return

    state.messages[-1][-1] = message
    path_to_sub_videos = state.get_path_to_subvideos()
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos, gr.Button(interactive=True))

def get_demo(rag_chain: Optional[Any] = None) -> gr.Blocks:
    """
    Create and configure the Gradio demo interface.
    
    Args:
        rag_chain: Optional RAG chain instance
        
    Returns:
        Configured Gradio Blocks interface
    """
    if rag_chain is None:
        rag_chain = get_default_rag_chain()
        
    with gr.Blocks(theme=THEME, css=CSS) as demo:
        instance = get_gradio_instance(rag_chain)
        state = gr.State(instance)
        
        # Load dark theme by default
        demo.load(
            None,
            None,
            js="""
            () => {
                const params = new URLSearchParams(window.location.search);
                if (!params.has('__theme')) {
                    params.set('__theme', 'dark');
                    window.location.search = params.toString();
                }
            }"""
        )
        
        # Header
        gr.HTML(value='''
            <table style="bordercolor=#0a0c2b; border=0">
            <tr style="height:150px; border:0">
                <td style="border:0"><img src="/file=./assets/header.png"></td>
            </tr>
            </table>
        ''')
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=4):
                video = gr.Video(height=512, width=512, elem_id="video", interactive=False)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Multimodal RAG Chatbot",
                    height=512,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Dropdown(
                            [
                                "What is the name of one of the astronauts?",
                                "An astronaut's spacewalk",
                                "What does the astronaut say?",
                            ],
                            allow_custom_value=True,
                            label="Query",
                            info="Enter your query here or choose a sample from the dropdown list!"
                        )
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send",
                            variant="primary",
                            interactive=True
                        )
                with gr.Row(elem_id="buttons"):
                    clear_btn = gr.Button(
                        value="🗑️  Clear history",
                        interactive=False
                    )

        # Event handlers
        btn_list = [clear_btn]
        clear_btn.click(
            clear_history,
            [state],
            [state, chatbot, textbox, video] + btn_list
        )
        submit_btn.click(
            add_text,
            [state, textbox],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [state],
            [state, chatbot, video] + btn_list,
        )
        
    return demo

