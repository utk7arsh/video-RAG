# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
from io import StringIO, BytesIO
import textwrap
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from enum import auto, Enum
import base64
import glob
from tqdm import tqdm
from pytubefix import YouTube, Stream
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
from predictionguard import PredictionGuard
import cv2
import json
import PIL
from PIL import Image
import dataclasses
import random
from datasets import load_dataset

from langchain_core.prompt_values import PromptValue
from langchain_core.messages import (
    MessageLikeRepresentation,
)

MultimodalModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]]

def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)

def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        return default
        
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_prediction_guard_api_key():
    load_env()
    PREDICTION_GUARD_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY", None)
    if PREDICTION_GUARD_API_KEY is None:
        PREDICTION_GUARD_API_KEY = input("Please enter your Prediction Guard API Key: ")
    return PREDICTION_GUARD_API_KEY
    
PREDICTION_GUARD_URL_ENDPOINT = os.getenv("DLAI_PREDICTION_GUARD_URL_ENDPOINT", "https://dl-itdc.predictionguard.com") ###"https://proxy-dl-itdc.predictionguard.com"

# prompt templates
templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]

# function helps to prepare list image-text pairs from the first [test_size] data of a Huggingface dataset
def prepare_dataset_for_umap_visualization(hf_dataset, class_name, templates=templates, test_size=1000):
    # load Huggingface dataset (download if needed)
    dataset = load_dataset(hf_dataset, trust_remote_code=True)
    # split dataset with specific test_size
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    # get the test dataset
    test_dataset = train_test_dataset['test']
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)].format(class_name),
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs
    

def download_video(video_url, path='/tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    
    yt = YouTube(video_url, on_progress_callback=progress_callback)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
    if stream is None:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, stream.default_filename)
    if not os.path.exists(filepath):   
        print('Downloading video from YouTube...')
        pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
        stream.download(path)
        pbar.close()
    return filepath

def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]

    return video_url
    
# if this has transcript then download
def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

    return filepath
    

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# a help function that helps to convert a specific time written as a string in format `webvtt` into a time in miliseconds
def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    # get the corresponding time as total seconds 
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds
    
def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)
    
# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

# encoding image at given path or PIL Image using base64
def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        # this is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# checking whether the given string is base64 or not
def isBase64(sb):
    try:
        if isinstance(sb, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
                sb_bytes = sb
        else:
                raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False

def encode_image_from_path_or_url(image_path_or_url):
    try:
        # try to open the url to check valid url
        f = urlopen(image_path_or_url)
        # if this is an url
        return base64.b64encode(requests.get(image_path_or_url).content).decode('utf-8')
    except:
        # this is a path to image
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# helper function to compute the joint embedding of a prompt and a base64-encoded image through PredictionGuard
def bt_embedding_from_prediction_guard(prompt, base64_image):
    # get PredictionGuard client
    client = _getPredictionGuardClient()
    message = {"text": prompt,}
    if base64_image is not None and base64_image != "":
        if not isBase64(base64_image): 
            raise TypeError("image input must be in base64 encoding!")
        message['image'] = base64_image
    response = client.embeddings.create(
        model="bridgetower-large-itm-mlm-itc",
        input=[message]
    )
    return response['data'][0]['embedding']

    
def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_retrieved_results(results):
    print(f'There is/are {len(results)} retrieved result(s)')
    print()
    for i, res in enumerate(results):
        print(f'The caption of the {str(i+1)}-th retrieved result is:\n"{results[i].page_content}"')
        print()
        display(Image.open(results[i].metadata['metadata']['extracted_frame_path']))
        print("------------------------------------------------------------")

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
