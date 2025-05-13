"""
Utility functions for the Video RAG project.
This module provides helper functions for video processing, image handling, 
transcript management, and AI model interactions.
"""

import os
import base64
import json
import random
import textwrap
import dataclasses
from enum import auto, Enum
from io import StringIO, BytesIO
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from urllib.request import urlopen
from pathlib import Path

import cv2
import openai
import PIL
import requests
from PIL import Image
from tqdm import tqdm
from pytubefix import YouTube, Stream
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import MessageLikeRepresentation

# Type definitions
MultimodalModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]]

# Constants
DEFAULT_TEMPLATES = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]

class SeparatorStyle(Enum):
    """Different separator styles for conversation formatting."""
    SINGLE = auto()

@dataclasses.dataclass
class Conversation:
    """
    A class that maintains conversation history for multimodal interactions.
    
    Attributes:
        system (str): System message for the conversation
        roles (List[str]): List of possible roles in the conversation
        messages (List[List[str]]): List of message pairs [role, content]
        map_roles (Dict[str, str]): Mapping of role names
        version (str): Version identifier for the conversation format
        sep_style (SeparatorStyle): Style of message separation
        sep (str): Separator string between messages
    """
    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"

    def _get_prompt_role(self, role: str) -> str:
        """Get the mapped role name if it exists."""
        return self.map_roles.get(role, role)

    def _build_content_for_first_message_in_conversation(self, first_message: List[str]) -> List[Dict[str, Any]]:
        """
        Build content for the first message in a conversation.
        
        Args:
            first_message: List containing [prompt, base64_image]
            
        Returns:
            List of content dictionaries for the API
        """
        if len(first_message) != 2:
            raise TypeError("First message must include prompt and base64-encoded image")
        
        prompt, b64_image = first_message[0], first_message[1]
        
        if prompt is None:
            raise TypeError("API does not support None prompt")
        if b64_image is None:
            raise TypeError("API does not support text-only conversation")
        if not isBase64(b64_image):
            raise TypeError("Image must be base64 encoded")
            
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": b64_image}
            }
        ]

    def _build_content_for_follow_up_messages_in_conversation(self, follow_up_message: List[str]) -> str:
        """Build content for follow-up messages in a conversation."""
        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError("Follow-up message must not include an image")
        
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError("Follow-up message must include exactly one text message")

        return follow_up_message[0]

    def get_message(self) -> List[Dict[str, Any]]:
        """Convert conversation to API message format."""
        api_messages = []
        for i, (role, message_content) in enumerate(self.messages):
            content = (
                self._build_content_for_first_message_in_conversation(message_content)
                if i == 0
                else self._build_content_for_follow_up_messages_in_conversation(message_content)
            )
            
            api_messages.append({
                "role": role,
                "content": content,
            })
        return api_messages

    def serialize_messages(self) -> str:
        """Serialize conversation into a single string format."""
        if self.sep_style != SeparatorStyle.SINGLE:
            raise ValueError(f"Invalid style: {self.sep_style}")

        ret = f"{self.system}{self.sep}" if self.system else ""
        
        for i, (role, message) in enumerate(self.messages):
            role = self._get_prompt_role(role)
            if message:
                if isinstance(message, List):
                    message = message[0]
                ret += message if i == 0 else f"{role}: {message}"
                if i < len(self.messages) - 1:
                    ret += self.sep
            else:
                ret += f"{role}:"
                
        return ret

    def append_message(self, role: str, message: List[str]) -> None:
        """Append a new message to the conversation."""
        if not self.messages:
            if role != self.roles[0]:
                raise ValueError(f"First message must be from role {self.roles[0]}")
            if len(message) != 2:
                raise ValueError("First message must include prompt and image")
            if not isBase64(message[1]):
                raise ValueError("Image must be base64 encoded")
        else:
            if role not in self.roles:
                raise ValueError(f"Message must be from one of {self.roles}")
            if len(message) != 1:
                raise ValueError("Follow-up message must be text only")
                
        self.messages.append([role, message])

    def copy(self) -> 'Conversation':
        """Create a deep copy of the conversation."""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary format."""
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }

# Initialize default conversation template
openai_gpt4v_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="OpenAI GPT-4V Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={"user": "user", "assistant": "assistant"}
)

# Environment and API Functions
def load_env() -> None:
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    return api_key

# Video Processing Functions
def download_video(video_url: str, path: str = '/tmp/') -> str:
    """
    Download video from YouTube URL.
    
    Args:
        video_url: YouTube video URL
        path: Directory to save the video
        
    Returns:
        Path to downloaded video file
    """
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    existing_videos = glob.glob(os.path.join(path, '*.mp4'))
    if existing_videos:
        return existing_videos[0]

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    
    yt = YouTube(video_url, on_progress_callback=progress_callback)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
    if not stream:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, stream.default_filename)
    
    if not os.path.exists(filepath):
        print('Downloading video from YouTube...')
        pbar = tqdm(desc='Downloading video', total=stream.filesize, unit="bytes")
        stream.download(path)
        pbar.close()
    
    return filepath

def get_video_id_from_url(video_url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Video ID string
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            return urllib.parse.parse_qs(url.query)['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]
    
    return video_url

# Transcript Functions
def get_transcript_vtt(video_url: str, path: str = '/tmp') -> str:
    """
    Get video transcript in VTT format.
    
    Args:
        video_url: YouTube video URL
        path: Directory to save transcript
        
    Returns:
        Path to VTT file
    """
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path, 'captions.vtt')
    
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(webvtt_formatted)
    
    return filepath

def format_timestamp(seconds: float, always_include_hours: bool = False, 
                    fractional_separator: str = '.') -> str:
    """
    Format timestamp for VTT/SRT files.
    
    Args:
        seconds: Time in seconds
        always_include_hours: Whether to always show hours
        fractional_separator: Separator for milliseconds
        
    Returns:
        Formatted timestamp string
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractional_separator}{milliseconds:03d}"

# Image Processing Functions
def encode_image(image_path_or_PIL_img: Union[str, PIL.Image.Image]) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_path_or_PIL_img: Path to image or PIL Image object
        
    Returns:
        Base64 encoded image string
    """
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    with open(image_path_or_PIL_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def isBase64(sb: Union[str, bytes]) -> bool:
    """
    Check if string is base64 encoded.
    
    Args:
        sb: String or bytes to check
        
    Returns:
        True if string is base64 encoded
    """
    try:
        if isinstance(sb, str):
            sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False

# AI Model Functions
def lvlm_inference(prompt: str, image: str, max_tokens: int = 300, **kwargs) -> str:
    """
    Perform inference using the vision-language model.
    
    Args:
        prompt: Text prompt
        image: Path to image or base64 string
        max_tokens: Maximum tokens in response
        
    Returns:
        Model response text
    """
    if not image.startswith("data:image"):
        image_url = f"data:image/jpeg;base64,{encode_image(image)}"
    else:
        image_url = image

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
    
    openai.api_key = get_openai_api_key()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def lvlm_inference_with_conversation(conversation: Conversation, 
                                   max_tokens: int = 300, **kwargs) -> str:
    """
    Perform inference using conversation history.
    
    Args:
        conversation: Conversation object
        max_tokens: Maximum tokens in response
        
    Returns:
        Model response text
    """
    messages = conversation.get_message()
    user_msg = messages[0]["content"]
    
    prompt = next(item["text"] for item in user_msg if item["type"] == "text")
    image_url = next(item["image_url"]["url"] for item in user_msg if item["type"] == "image_url")
    
    return lvlm_inference(prompt, image_url, max_tokens=max_tokens)
