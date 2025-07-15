#!/usr/bin/env python3
"""
YouTube Summarizer - Streamlit Web Application
Complete implementation with personalized summaries and intelligent caching
"""

import streamlit as st
import os
import sys
import subprocess
import tempfile
import shutil
import time
import gc
import hashlib
from pathlib import Path
import logging
from dotenv import load_dotenv
import re
import json
from collections import Counter
import pandas as pd
from datetime import datetime
import base64

try:
    import yt_dlp
    import whisper
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Missing library: {e}")
    st.error("Install: pip install yt-dlp openai-whisper google-generativeai python-dotenv")
    st.stop()

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SummarizeAI - YouTube Video Summarizer",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: rgba(15, 23, 42, 0.95);
        padding: 1rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .logo {
        display: flex;
        align-items: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #8b5cf6;
    }
    
    .logo-icon {
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        border-radius: 8px;
        padding: 8px;
        margin-right: 12px;
        color: white;
        font-size: 1.2rem;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f8fafc 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0;
        line-height: 1.1;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Input Container */
    .input-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        backdrop-filter: blur(15px);
    }
    
    /* Role Selection Cards */
    .role-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .role-card {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .role-card:hover {
        border-color: #8b5cf6;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
    }
    
    .role-card.selected {
        border-color: #8b5cf6;
        background: rgba(139, 92, 246, 0.1);
    }
    
    .role-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .role-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .role-description {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        color: #f8fafc;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    .stSelectbox > div > div > select {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        color: #f8fafc;
    }
    
    /* Progress and Status */
    .status-container {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .status-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .status-icon {
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Summary Container */
    .summary-container {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Analytics Cards */
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .analytics-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .analytics-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #8b5cf6;
        display: block;
    }
    
    .analytics-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.7);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'cache_stats' not in st.session_state:
    st.session_state.cache_stats = None

class YouTubeSummarizer:
    """YouTube Summarizer with caching and personalized analysis"""
    
    def __init__(self):
        """Initialize the summarizer with APIs and models"""
        self.setup_apis()
        self.load_whisper()
        self.cache_dir = Path("youtube_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def setup_apis(self):
        """Initialize Gemini API"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.use_gemini = True
                logger.info("✅ Gemini API ready")
            except Exception as e:
                logger.warning(f"Gemini API failed: {e}")
                self.use_gemini = False
        else:
            self.use_gemini = False
            logger.warning("❌ No Gemini API key - using basic summaries")
    
    def load_whisper(self):
        """Load Whisper model"""
        try:
            logger.info("🔄 Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper loaded")
        except Exception as e:
            raise Exception(f"Failed to load Whisper: {e}")
    
    def get_video_id_from_url(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        return video_id_match.group(1) if video_id_match else None
    
    def get_cache_paths(self, video_id: str) -> dict:
        """Get cache file paths for a video"""
        return {
            'metadata': self.cache_dir / f"{video_id}_metadata.json",
            'audio': self.cache_dir / f"{video_id}_audio.wav",
            'transcript': self.cache_dir / f"{video_id}_transcript.json"
        }
    
    def save_to_cache(self, video_id: str, data_type: str, data) -> None:
        """Save data to cache"""
        try:
            cache_paths = self.get_cache_paths(video_id)
            
            if data_type == 'metadata':
                with open(cache_paths['metadata'], 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif data_type == 'transcript':
                with open(cache_paths['transcript'], 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def load_from_cache(self, video_id: str, data_type: str):
        """Load data from cache"""
        try:
            cache_paths = self.get_cache_paths(video_id)
            
            if data_type == 'metadata':
                if cache_paths['metadata'].exists():
                    with open(cache_paths['metadata'], 'r', encoding='utf-8') as f:
                        return json.load(f)
            elif data_type == 'transcript':
                if cache_paths['transcript'].exists():
                    with open(cache_paths['transcript'], 'r', encoding='utf-8') as f:
                        return json.load(f)
            elif data_type == 'audio':
                if cache_paths['audio'].exists():
                    return str(cache_paths['audio'])
                    
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def download_audio(self, url: str, video_id: str, progress_callback=None) -> str:
        """Download audio with caching and progress updates"""
        
        # Check cache first
        cached_audio = self.load_from_cache(video_id, 'audio')
        if cached_audio and os.path.exists(cached_audio):
            if progress_callback:
                progress_callback("📁 Using cached audio")
            logger.info(f"📁 Using cached audio: {cached_audio}")
            return cached_audio
        
        cache_paths = self.get_cache_paths(video_id)
        output_path = str(cache_paths['audio'].with_suffix(''))
        
        try:
            if progress_callback:
                progress_callback("📥 Downloading audio...")
            logger.info("📥 Downloading audio...")
            
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': output_path + '.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 180,
                'retries': 3,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': ['-ac', '1', '-ar', '16000']
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the downloaded file
            final_audio_file = str(cache_paths['audio'])
            possible_files = [output_path + ext for ext in ['.wav', '.mp3', '.m4a']]
            
            for test_file in possible_files:
                if os.path.exists(test_file):
                    if not test_file.endswith('.wav'):
                        # Convert to WAV
                        cmd = ['ffmpeg', '-i', test_file, '-ac', '1', '-ar', '16000', '-y', final_audio_file]
                        subprocess.run(cmd, capture_output=True, timeout=60)
                        os.remove(test_file)
                    else:
                        shutil.move(test_file, final_audio_file)
                    return final_audio_file
            
            raise Exception("No audio file was created")
            
        except Exception as e:
            raise Exception(f"Audio download failed: {str(e)[:200]}")
    
    def get_audio_duration(self, audio_file: str) -> float:
        """Get audio duration"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                if duration > 0:
                    logger.info(f"📏 Audio duration: {duration/60:.1f} minutes")
                    return duration
        except Exception as e:
            logger.warning(f"Duration detection failed: {e}")
        
        # Fallback to file size estimation
        try:
            file_size = os.path.getsize(audio_file)
            estimated_duration = file_size / 32000
            return max(60, estimated_duration)
        except:
            return 600  # 10 minutes default
    
    def split_audio(self, input_file: str, total_duration: float, progress_callback=None) -> list:
        """Split audio into chunks with progress updates"""
        
        # Determine chunk size based on duration
        if total_duration <= 300:
            chunk_duration = 60
        elif total_duration <= 900:
            chunk_duration = 120
        elif total_duration <= 1800:
            chunk_duration = 180
        else:
            chunk_duration = 240
        
        if progress_callback:
            progress_callback(f"🔪 Splitting audio into {chunk_duration/60:.1f}min chunks...")
        logger.info(f"🔪 Splitting audio into {chunk_duration/60:.1f}min chunks...")
        
        chunks = []
        temp_dir = os.path.dirname(input_file)
        
        for start_time in range(0, int(total_duration), chunk_duration):
            chunk_file = os.path.join(temp_dir, f"chunk_{len(chunks):04d}.wav")
            actual_duration = min(chunk_duration, total_duration - start_time)
            
            try:
                cmd = [
                    'ffmpeg', '-i', input_file,
                    '-ss', str(start_time),
                    '-t', str(actual_duration),
                    '-ac', '1', '-ar', '16000',
                    '-y', '-loglevel', 'error',
                    chunk_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=90)
                
                if result.returncode == 0 and os.path.exists(chunk_file):
                    file_size = os.path.getsize(chunk_file)
                    if file_size > 1000:
                        chunks.append({
                            'file': chunk_file,
                            'start_time': start_time,
                            'duration': actual_duration,
                            'size': file_size
                        })
                        if progress_callback:
                            progress_callback(f"   ✂️  Chunk {len(chunks)} ready")
                        logger.info(f"   ✂️  Chunk {len(chunks)} ready")
                        continue
                
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    
            except Exception as e:
                logger.warning(f"Failed to create chunk at {start_time}s: {e}")
                continue
        
        if not chunks:
            raise Exception("No audio chunks were created")
        
        if progress_callback:
            progress_callback(f"✅ Created {len(chunks)} audio chunks")
        logger.info(f"✅ Created {len(chunks)} audio chunks")
        return chunks
    
    def clean_transcript(self, text: str) -> str:
        """Clean transcript text and remove casual language"""
        if not text:
            return ""
        
        try:
            # Remove excessive repetitions
            text = re.sub(r'\b(\w+)(\s+\1\b){2,}', r'\1', text)
            
            # Remove casual filler words and expressions
            casual_patterns = [
                r'\b(?i:um|uh|er|ah|eh|oh|hmm|mmm|ugh)\b',
                r'\b(?i:you know|like|actually|basically|literally|obviously|definitely)\b',
                r'\b(?i:sort of|kind of|a bit|a little|pretty much)\b',
                r'\b(?i:yes,?\s*my\s*god,?\s*yes|oh\s*my\s*god|jesus|christ)\b',
                r'\b(?i:yeah|yep|nah|nope|okay|ok|alright|right)\s*[,.]?\s*',
                r'\b(?i:so\s*anyway|anyway|well\s*anyway)\b'
            ]
            
            for pattern in casual_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Remove annotations and sound effects
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(.*?\)', '', text)
            text = re.sub(r'<.*?>', '', text)
            
            # Clean up incomplete thoughts and trailing phrases
            text = re.sub(r'\s*\.\.\.\s*$', '.', text)
            text = re.sub(r'\s*,\s*$', '.', text)
            text = re.sub(r'\s*and\s*$', '.', text)
            text = re.sub(r'\s*but\s*$', '.', text)
            text = re.sub(r'\s*so\s*$', '.', text)
            
            # Fix spacing and punctuation
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([,.!?;:])', r'\1', text)
            text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1', text)
            
            # Capitalize properly and ensure professional tone
            text = text.strip()
            if text:
                sentences = re.split(r'[.!?]+', text)
                cleaned_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 3:
                        # Capitalize first letter
                        sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                        
                        # Replace casual phrases with professional alternatives
                        sentence = re.sub(r'\b(?i:gonna)\b', 'going to', sentence)
                        sentence = re.sub(r'\b(?i:wanna)\b', 'want to', sentence)
                        sentence = re.sub(r'\b(?i:gotta)\b', 'need to', sentence)
                        sentence = re.sub(r'\b(?i:kinda)\b', 'somewhat', sentence)
                        sentence = re.sub(r'\b(?i:sorta)\b', 'somewhat', sentence)
                        
                        cleaned_sentences.append(sentence)
                
                text = '. '.join(cleaned_sentences)
                if text and text[-1] not in '.!?':
                    text += '.'
            
            return text
            
        except Exception as e:
            logger.warning(f"Text cleaning error: {e}")
            return str(text).strip() if text else ""
    
    def transcribe_chunks(self, chunks: list, video_id: str, progress_callback=None) -> list:
        """Transcribe chunks with caching and progress updates"""
        
        # Check if transcript is cached
        cached_transcript = self.load_from_cache(video_id, 'transcript')
        if cached_transcript:
            if progress_callback:
                progress_callback("📁 Using cached transcript")
            logger.info("📁 Using cached transcript")
            return cached_transcript
        
        transcripts = []
        if progress_callback:
            progress_callback(f"🎤 Transcribing {len(chunks)} chunks...")
        logger.info(f"🎤 Transcribing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                if progress_callback:
                    progress_callback(f"   🎵 Processing chunk {i}/{len(chunks)}")
                logger.info(f"   🎵 Processing chunk {i}/{len(chunks)}")
                
                result = self.whisper_model.transcribe(
                    chunk['file'],
                    fp16=False,
                    language=None,
                    verbose=False,
                    temperature=0.0
                )
                
                text = result['text'].strip()
                
                if text and len(text) > 10:
                    text = self.clean_transcript(text)
                    
                    if len(text) > 10:
                        transcripts.append({
                            'start_time': chunk['start_time'],
                            'end_time': chunk['start_time'] + chunk['duration'],
                            'text': text,
                            'timestamp': f"{chunk['start_time']//60}:{chunk['start_time']%60:02d}"
                        })
                        if progress_callback:
                            progress_callback(f"      ✅ {len(text)} chars")
                        logger.info(f"      ✅ {len(text)} chars")
                
                # Cleanup chunk file
                try:
                    os.remove(chunk['file'])
                except:
                    pass
                
            except Exception as e:
                logger.warning(f"Chunk {i} transcription failed: {e}")
                continue
        
        # Cache the transcript
        if transcripts:
            self.save_to_cache(video_id, 'transcript', transcripts)
        
        if progress_callback:
            progress_callback(f"✅ Transcribed {len(transcripts)} segments")
        logger.info(f"✅ Transcribed {len(transcripts)} segments")
        return transcripts
    
    def extract_key_concepts(self, transcripts: list) -> dict:
        """Extract meaningful key concepts from transcripts"""
        
        full_text = " ".join([t['text'] for t in transcripts])
        
        # Comprehensive stop words including vague terms
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'will', 'an', 'be', 'or', 'by',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'not', 'no', 'yes', 'but',
            'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what', 'who', 'whom', 'whose', 'because', 'since', 'for',
            'now', 'here', 'there', 'today', 'tomorrow', 'yesterday', 'again', 'also', 'still', 'just', 'only', 'even',
            'very', 'really', 'quite', 'so', 'too', 'more', 'most', 'much', 'many', 'little', 'few', 'some', 'any', 'all',
            'going', 'get', 'got', 'getting', 'goes', 'went', 'come', 'came', 'coming', 'comes', 'put', 'puts', 'putting',
            'take', 'takes', 'taking', 'took', 'make', 'makes', 'making', 'made', 'give', 'gives', 'giving', 'gave',
            'see', 'sees', 'seeing', 'saw', 'look', 'looks', 'looking', 'looked', 'want', 'wants', 'wanted', 'wanting',
            'know', 'knows', 'knowing', 'knew', 'think', 'thinks', 'thinking', 'thought', 'say', 'says', 'saying', 'said',
            'tell', 'tells', 'telling', 'told', 'ask', 'asks', 'asking', 'asked', 'work', 'works', 'working', 'worked',
            'try', 'tries', 'trying', 'tried', 'use', 'uses', 'using', 'used', 'need', 'needs', 'needing', 'needed',
            'like', 'likes', 'liking', 'liked', 'well', 'good', 'better', 'best', 'bad', 'worse', 'worst',
            # Additional vague terms to filter out
            'with', 'without', 'within', 'out', 'different', 'same', 'other', 'another', 'each', 'every', 'both',
            'either', 'neither', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below',
            'around', 'about', 'into', 'onto', 'upon', 'across', 'under', 'over', 'from', 'off', 'back', 'away',
            'thing', 'things', 'something', 'anything', 'nothing', 'everything', 'someone', 'anyone', 'everyone',
            'way', 'ways', 'right', 'left', 'first', 'last', 'next', 'new', 'old', 'big', 'small', 'long', 'short'
        }
        
        # Enhanced business concept patterns with more specific context
        business_concept_patterns = [
            # AI Development & Productivity Tools
            r'\b(?i:(?:Claude\s*Code|Cursor\s*AI|GitHub\s*Copilot|AI\s*agent|code\s*assistant)\s*(?:for|in|with)?\s*(?:development|productivity|automation|workflow|efficiency)?)\b',
            r'\b(?i:(?:developer|development|coding)\s*(?:productivity|efficiency|automation|workflow|assistance|acceleration))\b',
            r'\b(?i:(?:AI-powered|AI-assisted|automated)\s*(?:development|coding|file\s*editing|code\s*review|software\s*development))\b',
            
            # Healthcare & Medical
            r'\b(?i:(?:medical|dental|healthcare|clinic|practice|patient|doctor|physician|appointment|consultation|insurance|coverage|HIPAA|compliance)\s*(?:automation|system|management|workflow|process|integration|software)?)\b',
            r'\b(?i:(?:patient|lead)\s*(?:qualification|generation|management|follow-up|conversion|interaction|booking|scheduling))\b',
            
            # AI & Automation
            r'\b(?i:AI\s*(?:agent|assistant|bot|call|phone|voice|automation|system|workflow|integration|powered|driven))\b',
            r'\b(?i:(?:automated|automatic)\s*(?:booking|scheduling|appointment|call|follow-up|lead|response|workflow|process))\b',
            r'\b(?i:(?:voice|phone|call)\s*(?:agent|automation|system|integration|workflow|processing|handling))\b',
            
            # Business & Sales
            r'\b(?i:(?:sales|marketing|business|customer|client)\s*(?:automation|process|funnel|pipeline|strategy|workflow|management|system))\b',
            r'\b(?i:(?:CRM|database|data)\s*(?:integration|management|storage|workflow|automation|system|processing))\b',
            r'\b(?i:(?:real-time|24/7|round-the-clock)\s*(?:automation|processing|system|workflow|operation|monitoring))\b',
            
            # Technical Integration & Development
            r'\b(?i:(?:no-code|low-code|workflow|API|webhook|database)\s*(?:automation|integration|deployment|framework|solution|platform))\b',
            r'\b(?i:(?:terminal\s*commands|file\s*editing|code\s*generation|agent\s*automation|development\s*workflow))\b'
        ]
        
        meaningful_concepts = []
        for pattern in business_concept_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                cleaned_match = ' '.join(match.split())  # Clean whitespace
                if len(cleaned_match) > 3:
                    meaningful_concepts.append(cleaned_match.lower())
        
        # Count concept frequency
        concept_counts = Counter(meaningful_concepts)
        domain_concepts = [(concept, count) for concept, count in concept_counts.most_common(8) if count >= 1]
        
        # Extract meaningful single words (higher threshold)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
        meaningful_words = [word for word in words if word not in stop_words]
        word_counts = Counter(meaningful_words)
        
        # Only keep high-frequency, meaningful single words
        significant_keywords = [(word, count) for word, count in word_counts.most_common(15) 
                              if count >= 3 and len(word) > 4]
        
        # Combine domain concepts with significant keywords
        combined_topics = domain_concepts + significant_keywords[:5]
        
        # Enhanced technical terms detection with business context
        technical_patterns = [
            # AI Development Tools & Platforms
            r'\b(?i:Claude\s*Code|Cursor\s*AI|GitHub\s*Copilot|AI\s*agent|code\s*assistant|development\s*agent)\b',
            r'\b(?i:VS\s*Code|terminal\s*commands|file\s*editing|code\s*review|PR\s*automation)\b',
            
            # Specific platforms and tools
            r'\b(?i:Twilio|Supabase|Airtable|Zapier|Shopify|Stripe|PayPal|Zoom|Slack|Discord|Notion|Retool|Bubble)\b',
            r'\b(?i:HubSpot|Salesforce|Mailchimp|ConvertKit|GoHighLevel|ActiveCampaign|Klaviyo)\b',
            r'\b(?i:OpenAI|ChatGPT|GPT|Claude|Anthropic|Whisper|Assembly|Deepgram|ElevenLabs|Vapi)\b',
            
            # Programming and technical terms
            r'\b(?i:API|SDK|REST|GraphQL|JWT|OAuth|JSON|XML|CSV|HTML|CSS|JavaScript|Python|React|Vue|Angular|Node\.?js|PHP)\b',
            r'\b(?i:Firebase|MongoDB|PostgreSQL|MySQL|Redis|AWS|Azure|Google\s*Cloud)\b',
            
            # Business and analytics terms
            r'\b(?i:CRM|ERP|CMS|LMS|ATS|POS|ROI|KPI|SaaS|B2B|B2C|HIPAA)\b',
            
            # Development workflow terms
            r'\b(?i:workflow\s*automation|productivity\s*tool|development\s*efficiency|code\s*generation|agent\s*automation)\b',
            
            # Healthcare specific
            r'\b(?i:EMR|EHR|HL7|FHIR|telehealth|telemedicine|patient\s*portal)\b'
        ]
        
        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 1:
                    technical_terms.append(match.strip())
        
        # Remove duplicates while preserving case and cleaning
        seen_terms = set()
        clean_technical_terms = []
        for term in technical_terms:
            term_lower = term.lower()
            if term_lower not in seen_terms and len(term) > 1:
                seen_terms.add(term_lower)
                clean_technical_terms.append(term)
        
        # Enhanced action words (implementation-focused)
        action_patterns = [
            r'\b(?i:implement|deploy|integrate|automate|configure|setup|build|create|develop|design|optimize)\b',
            r'\b(?i:schedule|booking|qualify|convert|track|monitor|analyze|process|manage|handle)\b',
            r'\b(?i:streamline|enhance|improve|scale|customize|connect|sync|generate|execute)\b'
        ]
        
        action_words = []
        for pattern in action_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            action_words.extend([match.lower() for match in matches])
        
        action_words = list(set(action_words))
        
        return {
            'keywords': combined_topics[:10],
            'domain_concepts': domain_concepts,
            'technical_terms': clean_technical_terms[:12],
            'action_words': action_words[:10],
            'total_words': len(words),
            'unique_words': len(set(words)),
            'complexity_score': len(set(words)) / len(words) if words else 0
        }
    
    def create_ai_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict, user_profile: dict, progress_callback=None) -> str:
        """Create AI-enhanced summary"""
        
        # Combine transcript for AI analysis
        full_transcript = "\n\n".join([f"[{t['timestamp']}] {t['text']}" for t in transcripts])
        
        try:
            if progress_callback:
                progress_callback(f"🤖 Creating AI summary for {user_profile['name']}...")
            logger.info(f"🤖 Creating AI summary for {user_profile['name']}...")
            
            # Create prompt based on user profile
            user_focus_map = {
                "student": "learning objectives, practical applications, study tips",
                "professional": "business value, ROI, implementation strategies",
                "developer": "technical implementation, tools, best practices",
                "entrepreneur": "business opportunities, growth strategies, revenue models",
                "researcher": "methodology, evidence, theoretical frameworks",
                "general": "clear explanations, practical insights, actionable takeaways"
            }
            
            focus_areas = user_focus_map.get(user_profile['type'], user_focus_map['general'])
            
            length_guide = {
                "quick": "Be concise and focus on key points only.",
                "standard": "Provide balanced detail with clear explanations.",
                "detailed": "Offer comprehensive analysis with examples and insights."
            }
            
            prompt = f"""
Create a personalized summary for a {user_profile['name']}.

**VIDEO:** {video_title}
**FOCUS:** {focus_areas}
**LENGTH:** {length_guide[user_profile['length']]}
{f"**SPECIAL FOCUS:** {user_profile['focus']}" if user_profile.get('focus') else ""}

**KEY CONCEPTS:**
- Main topics: {', '.join([kw[0] for kw in key_concepts['keywords'][:5]])}
- Technical terms: {', '.join(key_concepts['technical_terms'][:5])}
- Action words: {', '.join(key_concepts['action_words'][:5])}

**TRANSCRIPT:**
{full_transcript}

Create a structured summary with:
1. Compelling overview
2. Clear sections with emojis
3. Actionable insights for the {user_profile['name']}
4. Next steps or recommendations
"""
            
            response = self.gemini_model.generate_content(prompt)
            ai_summary = response.text
            
            # Create final summary
            final_summary = f"""# {user_profile['icon']} {video_title}

> **Personalized for**: {user_profile['name']} | **Length**: {user_profile['length'].title()}
{f"> **Focus**: {user_profile['focus']}" if user_profile.get('focus') else ""}

*🔗 [Watch Video]({video_url})*

{ai_summary}

---

## 📊 Video Analytics
- **Duration**: {duration_str}
- **Segments**: {len(transcripts)}
- **Word count**: {key_concepts['total_words']:,}
- **Main topics**: {', '.join([kw[0] for kw in key_concepts['keywords'][:5]])}
- **Technical terms**: {', '.join(key_concepts['technical_terms'][:5]) if key_concepts['technical_terms'] else 'None'}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

*🎯 AI-Enhanced summary by YouTube Summarizer*
"""
            
            logger.info("✅ AI summary created successfully!")
            return final_summary
            
        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            return self.create_basic_summary(transcripts, video_title, video_url, duration_str, key_concepts, user_profile, progress_callback)
    
    def create_basic_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict, user_profile: dict, progress_callback=None) -> str:
        """Create enhanced basic summary without AI"""
        
        icon = user_profile['icon']
        user_name = user_profile['name']
        
        # Extract meaningful topics (prioritize domain concepts)
        topics = []
        if key_concepts.get('domain_concepts'):
            topics.extend([concept for concept, count in key_concepts['domain_concepts'][:5]])
        
        # Add high-frequency keywords if we need more topics
        if len(topics) < 5 and key_concepts.get('keywords'):
            for keyword, count in key_concepts['keywords']:
                if isinstance(keyword, str) and len(keyword) > 4 and keyword not in topics:
                    topics.append(keyword)
                if len(topics) >= 5:
                    break
        
        # Fallback topics if still empty
        if not topics:
            topics = ['automation systems', 'workflow integration', 'business processes']
        
        summary = f"""# {icon} {video_title}

> **Personalized for**: {user_name} | **Length**: {user_profile['length'].title()}
{f"> **Focus**: {user_profile['focus']}" if user_profile.get('focus') else ""}

*🔗 [Watch Video]({video_url})*

## 🎯 Overview for {user_name}

**📋 Main Topics**: {', '.join(topics[:5])}

**🛠️ Technical Elements**: {', '.join(set(key_concepts['technical_terms'][:6])) if key_concepts['technical_terms'] else 'General automation and integration tools'}

**⚡ Key Actions**: {', '.join(key_concepts['action_words'][:6]) if key_concepts['action_words'] else 'Implementation and process optimization'}

## 📝 Content Breakdown

"""
        
        # Add enhanced segments based on user preference
        segment_count = {"quick": 3, "standard": 5, "detailed": 7}[user_profile['length']]
        segment_count = min(segment_count, len(transcripts))
        
        if segment_count > 0:
            step = max(1, len(transcripts) // segment_count)
            
            for i in range(0, len(transcripts), step):
                if i < len(transcripts):
                    transcript = transcripts[i]
                    
                    max_chars = {"quick": 200, "standard": 350, "detailed": 500}[user_profile['length']]
                    
                    text = transcript['text']
                    if len(text) > max_chars:
                        # Find good breaking point
                        truncated = text[:max_chars]
                        last_period = truncated.rfind('.')
                        last_exclamation = truncated.rfind('!')
                        last_question = truncated.rfind('?')
                        
                        break_point = max(last_period, last_exclamation, last_question)
                        if break_point > max_chars * 0.7:
                            text = text[:break_point + 1]
                        else:
                            # Break at last complete word
                            last_space = truncated.rfind(' ')
                            if last_space > max_chars * 0.8:
                                text = text[:last_space] + "..."
                            else:
                                text = truncated + "..."
                    
                    # Determine segment type based on content
                    segment_title = f"Key Segment ({transcript['timestamp']})"
                    text_lower = text.lower()
                    
                    if any(word in text_lower for word in ['introduction', 'intro', 'welcome', 'start', 'overview']):
                        segment_title = f"Introduction ({transcript['timestamp']})"
                    elif any(word in text_lower for word in ['conclusion', 'summary', 'wrap', 'end', 'final']):
                        segment_title = f"Conclusion ({transcript['timestamp']})"
                    elif any(word in text_lower for word in ['demo', 'demonstration', 'example', 'show', 'tutorial']):
                        segment_title = f"Demonstration ({transcript['timestamp']})"
                    elif any(word in text_lower for word in ['implement', 'setup', 'configure', 'build', 'create']):
                        segment_title = f"Implementation ({transcript['timestamp']})"
                    elif any(tech in text_lower for tech in [term.lower() for term in key_concepts['technical_terms'][:5]]):
                        segment_title = f"Technical Details ({transcript['timestamp']})"
                    
                    summary += f"### 🔸 {segment_title}\n\n{text}\n\n"
        
        # Add rest of summary based on user profile
        summary += f"""

---

## 📊 Video Analytics
- **Duration**: {duration_str}
- **Segments**: {len(transcripts)}
- **Word count**: {key_concepts['total_words']:,}
- **Main topics**: {', '.join([kw[0] for kw in key_concepts['keywords'][:5]])}
- **Technical terms**: {', '.join(key_concepts['technical_terms'][:5]) if key_concepts['technical_terms'] else 'None'}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

*🔧 Enhanced analysis (AI unavailable)*
*💡 Add GEMINI_API_KEY for AI-powered summaries*
*🎯 Intelligently analyzed for {icon} {user_name}*
"""
        
        return summary
    
    def create_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict, user_profile: dict, progress_callback=None) -> str:
        """Create personalized summary"""
        
        if not transcripts:
            return "# Error\n\nNo transcript available for summarization."
        
        if self.use_gemini:
            return self.create_ai_summary(transcripts, video_title, video_url, duration_str, key_concepts, user_profile, progress_callback)
        else:
            return self.create_basic_summary(transcripts, video_title, video_url, duration_str, key_concepts, user_profile, progress_callback)
    
    def process_video(self, url: str, user_profile: dict, progress_callback=None) -> dict:
        """Main processing function with progress updates"""
        
        if not url or not isinstance(url, str):
            return {'success': False, 'error': 'Invalid URL provided'}
        
        if not ('youtube.com' in url or 'youtu.be' in url):
            return {'success': False, 'error': 'Please provide a valid YouTube URL'}
        
        # Extract video ID
        video_id = self.get_video_id_from_url(url)
        if not video_id:
            return {'success': False, 'error': 'Could not extract video ID from URL'}
        
        logger.info(f"🎬 Processing video ID: {video_id}")
        
        start_time = time.time()
        temp_dir = None
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="yt_summarizer_")
            if progress_callback:
                progress_callback("🚀 Starting video analysis...")
            logger.info("🚀 Starting video analysis...")
            
            # 1. Extract video metadata
            if progress_callback:
                progress_callback("📋 Extracting metadata...")
            logger.info("📋 Extracting metadata...")
            
            cached_metadata = self.load_from_cache(video_id, 'metadata')
            if cached_metadata:
                if progress_callback:
                    progress_callback("📁 Using cached metadata")
                logger.info("📁 Using cached metadata")
                video_title = cached_metadata['title']
                duration = cached_metadata['duration']
                channel = cached_metadata['channel']
            else:
                try:
                    ydl_opts = {
                        'quiet': True, 
                        'no_warnings': True,
                        'socket_timeout': 30,
                        'retries': 3,
                        'extract_flat': False
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        
                        if not info:
                            raise Exception("Could not extract video information")
                        
                        video_title = str(info.get('title', 'Unknown Video')).strip()
                        duration = int(info.get('duration', 0))
                        channel = str(info.get('uploader', 'Unknown Channel')).strip()
                        
                        if not video_title or video_title == 'Unknown Video':
                            video_title = f"Video_{video_id}"
                        
                        if duration <= 0:
                            duration = 600  # Default 10 minutes
                        
                        # Cache metadata
                        metadata = {
                            'title': video_title,
                            'duration': duration,
                            'channel': channel,
                            'url': url
                        }
                        self.save_to_cache(video_id, 'metadata', metadata)
                        
                except Exception as e:
                    error_msg = f"Failed to extract video info: {str(e)[:200]}"
                    return {'success': False, 'error': error_msg}
            
            duration_str = f"{duration//60}:{duration%60:02d}"
            
            if progress_callback:
                progress_callback(f"✅ Video: {video_title[:50]}{'...' if len(video_title) > 50 else ''}")
            logger.info(f"✅ Video: {video_title[:50]}{'...' if len(video_title) > 50 else ''}")
            logger.info(f"📺 Channel: {channel}")
            logger.info(f"⏱️  Duration: {duration_str}")
            
            # Check video length
            if duration > 7200:  # 2 hours
                return {
                    'success': False, 
                    'error': f'Video too long ({duration//3600}h {(duration%3600)//60}m). Maximum: 2 hours'
                }
            
            # 2. Download and process audio
            try:
                audio_file = self.download_audio(url, video_id, progress_callback)
                
                if not audio_file or not os.path.exists(audio_file):
                    raise Exception("Audio download failed")
                    
            except Exception as e:
                return {'success': False, 'error': f'Audio processing failed: {str(e)[:300]}'}
            
            # 3. Split audio into chunks
            try:
                actual_duration = self.get_audio_duration(audio_file)
                chunks = self.split_audio(audio_file, actual_duration, progress_callback)
                
                if not chunks:
                    raise Exception("No audio chunks created")
                
            except Exception as e:
                return {'success': False, 'error': f'Audio splitting failed: {str(e)[:300]}'}
            
            # 4. Transcribe audio
            try:
                transcripts = self.transcribe_chunks(chunks, video_id, progress_callback)
                
                if not transcripts:
                    raise Exception("No transcripts generated")
                
            except Exception as e:
                return {'success': False, 'error': f'Transcription failed: {str(e)[:300]}'}
            
            # 5. Analyze and create summary
            try:
                if progress_callback:
                    progress_callback("🔍 Analyzing content...")
                key_concepts = self.extract_key_concepts(transcripts)
                summary_text = self.create_summary(transcripts, video_title, url, duration_str, key_concepts, user_profile, progress_callback)
                
                if not summary_text or len(summary_text) < 100:
                    raise Exception("Summary generation failed")
                
            except Exception as e:
                return {'success': False, 'error': f'Summary creation failed: {str(e)[:300]}'}
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(f"🎉 Analysis completed in {processing_time/60:.1f} minutes")
            logger.info(f"🎉 Analysis completed in {processing_time/60:.1f} minutes")
            
            return {
                'success': True,
                'summary': summary_text,
                'video_title': video_title,
                'channel': channel,
                'duration': duration_str,
                'segments': len(transcripts),
                'processing_time': f"{processing_time/60:.1f} minutes",
                'ai_enhanced': self.use_gemini,
                'user_profile': user_profile.copy(),
                'cached_data_used': bool(cached_metadata or self.load_from_cache(video_id, 'transcript')),
                'key_concepts': key_concepts
            }
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)[:300]}'
            }
            
        finally:
            # Cleanup temporary files (keep cache)
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Cleanup warning: {cleanup_error}")
            
            gc.collect()

def get_cache_info():
    """Get cache information"""
    cache_dir = Path("youtube_cache")
    if not cache_dir.exists():
        return {
            'exists': False,
            'files': 0,
            'size_mb': 0,
            'videos': 0
        }
    
    files = list(cache_dir.glob("*"))
    if not files:
        return {
            'exists': True,
            'files': 0,
            'size_mb': 0,
            'videos': 0
        }
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    # Group by video ID
    video_data = {}
    for file in files:
        if file.is_file():
            name_parts = file.stem.split('_')
            if len(name_parts) >= 2:
                video_id = name_parts[0]
                if video_id not in video_data:
                    video_data[video_id] = []
    
    return {
        'exists': True,
        'files': len(files),
        'size_mb': total_size / 1024 / 1024,
        'videos': len(video_data)
    }

def clear_cache():
    """Clear all cached files"""
    cache_dir = Path("youtube_cache")
    if cache_dir.exists():
        try:
            file_count = len(list(cache_dir.glob("*")))
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            return file_count
        except Exception as e:
            return None
    return 0

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <div class="logo">
                <span class="logo-icon">▶️</span>
                SummarizeAI
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                🌟 Dashboard
            </div>
        </div>
        <h1 class="main-title">Summarize Any YouTube Video<br>in Seconds — Powered by AI</h1>
        <p class="subtitle">Transform long videos into personalized summaries tailored to your role and interests. Save time, learn faster, and never miss key insights.</p>
    </div>
    """, unsafe_allow_html=True)

def render_role_selection():
    """Render role selection interface"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #f8fafc; font-size: 2rem; margin-bottom: 1rem;">Choose Your Role for Personalized Summaries</h2>
    </div>
    """, unsafe_allow_html=True)
    
    roles = [
        {
            'type': 'student',
            'name': 'Student',
            'icon': '🎓',
            'description': 'Learn and study effectively'
        },
        {
            'type': 'developer',
            'name': 'Developer',
            'icon': '💻',
            'description': 'Technical implementation focus'
        },
        {
            'type': 'entrepreneur',
            'name': 'Entrepreneur',
            'icon': '🚀',
            'description': 'Business opportunities & growth'
        },
        {
            'type': 'researcher',
            'name': 'Researcher',
            'icon': '🔬',
            'description': 'Academic & evidence-based'
        },
        {
            'type': 'professional',
            'name': 'Business Pro',
            'icon': '💼',
            'description': 'Strategic business insights'
        },
        {
            'type': 'general',
            'name': 'General',
            'icon': '🌟',
            'description': 'Clear & practical explanations'
        }
    ]
    
    # Create two rows of three columns each
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            role_idx = row * 3 + col_idx
            if role_idx < len(roles):
                role = roles[role_idx]
                with cols[col_idx]:
                    # Create a button-like card
                    selected = st.session_state.user_profile and st.session_state.user_profile['type'] == role['type']
                    
                    card_class = "role-card selected" if selected else "role-card"
                    
                    if st.button(
                        f"{role['icon']}\n{role['name']}\n{role['description']}",
                        key=f"role_{role['type']}",
                        use_container_width=True
                    ):
                        st.session_state.user_profile = {
                            'type': role['type'],
                            'name': role['name'],
                            'icon': role['icon'],
                            'description': role['description'],
                            'length': 'standard',
                            'focus': None
                        }
                        st.rerun()

def render_input_form():
    """Render the main input form"""
    if not st.session_state.user_profile:
        return
    
    st.markdown("""
    <div class="input-container">
    """, unsafe_allow_html=True)
    
    # Show selected profile
    profile = st.session_state.user_profile
    st.markdown(f"""
    <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; border: 1px solid rgba(139, 92, 246, 0.3);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 2rem; margin-right: 1rem;">{profile['icon']}</span>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #f8fafc;">{profile['name']}</div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">{profile['description']}</div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Summary Length</div>
                <div style="font-size: 1rem; color: #8b5cf6; font-weight: 600;">{profile['length'].title()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # URL Input
    st.markdown("### 📺 YouTube Video URL")
    url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
    
    # Additional options in expandable section
    with st.expander("⚙️ Customize Your Summary", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Summary Length**")
            length = st.selectbox(
                "Choose length",
                ["quick", "standard", "detailed"],
                index=["quick", "standard", "detailed"].index(profile['length']),
                format_func=lambda x: {
                    "quick": "⚡ Quick (2-3 min read)",
                    "standard": "📋 Standard (5-7 min read)", 
                    "detailed": "📖 Detailed (10+ min read)"
                }[x],
                label_visibility="collapsed"
            )
            
        with col2:
            st.markdown("**🎯 Special Focus (Optional)**")
            focus = st.text_input(
                "Any specific areas to focus on?",
                value=profile.get('focus', '') or '',
                placeholder="e.g., implementation details, cost analysis...",
                label_visibility="collapsed"
            )
    
    # Update profile if changed
    if length != profile['length'] or focus != profile.get('focus'):
        st.session_state.user_profile['length'] = length
        st.session_state.user_profile['focus'] = focus if focus.strip() else None
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Generate Summary",
            use_container_width=True,
            disabled=not url or st.session_state.processing
        ):
            if url:
                st.session_state.processing = True
                st.session_state.summary_result = None
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return url

def render_status_updates(status_text):
    """Render status updates"""
    st.markdown(f"""
    <div class="status-container">
        <div class="status-item">
            <span class="status-icon">⚡</span>
            <span>{status_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_analytics_cards(result):
    """Render analytics cards"""
    st.markdown("""
    <div class="analytics-grid">
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="analytics-card">
            <span class="analytics-value">{result['duration']}</span>
            <div class="analytics-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="analytics-card">
            <span class="analytics-value">{result['segments']}</span>
            <div class="analytics-label">Segments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="analytics-card">
            <span class="analytics-value">{'✨ AI' if result['ai_enhanced'] else '🔧 Basic'}</span>
            <div class="analytics-label">Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="analytics-card">
            <span class="analytics-value">{result['processing_time']}</span>
            <div class="analytics-label">Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_summary_result(result):
    """Render the summary result"""
    st.markdown("""
    <div class="summary-container">
    """, unsafe_allow_html=True)
    
    # Success message
    st.success(f"🎉 Summary generated successfully!")
    
    # Video info
    st.markdown(f"""
    ### 🎬 {result['video_title']}
    **Channel:** {result['channel']}  
    **Your Profile:** {result['user_profile']['icon']} {result['user_profile']['name']}
    """)
    
    # Analytics cards
    render_analytics_cards(result)
    
    # Summary content
    st.markdown("### 📋 Your Personalized Summary")
    st.markdown(result['summary'])
    
    # Download button
    summary_bytes = result['summary'].encode('utf-8')
    safe_title = re.sub(r'[^\w\s\-_]', '', result['video_title'])[:30]
    filename = f"summary_{result['user_profile']['type']}_{safe_title}_{int(time.time())}.md"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Summary",
            data=summary_bytes,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True
        )
    
    # Key concepts if available
    if 'key_concepts' in result:
        with st.expander("🔍 Content Analysis", expanded=False):
            kc = result['key_concepts']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if kc['keywords']:
                    st.markdown("**📋 Main Topics**")
                    for topic, count in kc['keywords'][:5]:
                        st.markdown(f"• {topic} ({count} mentions)")
                
                if kc['action_words']:
                    st.markdown("**⚡ Action Words**")
                    st.markdown(f"• {', '.join(kc['action_words'][:8])}")
            
            with col2:
                if kc['technical_terms']:
                    st.markdown("**🛠️ Technical Terms**")
                    for term in kc['technical_terms'][:8]:
                        st.markdown(f"• {term}")
                
                st.markdown("**📊 Text Statistics**")
                st.markdown(f"• Total words: {kc['total_words']:,}")
                st.markdown(f"• Unique words: {kc['unique_words']:,}")
                st.markdown(f"• Complexity: {kc['complexity_score']:.1%}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with cache info and controls"""
    with st.sidebar:
        st.markdown("### 🛠️ Tools & Settings")
        
        # Cache information
        cache_info = get_cache_info()
        st.markdown("#### 📁 Cache Status")
        
        if cache_info['exists']:
            st.markdown(f"""
            - **Files:** {cache_info['files']}
            - **Size:** {cache_info['size_mb']:.1f} MB
            - **Videos:** {cache_info['videos']}
            """)
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cleared = clear_cache()
                if cleared is not None:
                    st.success(f"✅ Cleared {cleared} files")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear cache")
        else:
            st.info("📁 No cache found")
        
        # API Status
        st.markdown("#### 🤖 API Status")
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            st.success("✅ Gemini AI Ready")
        else:
            st.warning("⚠️ Basic Mode Only")
            st.info("Add GEMINI_API_KEY for AI summaries")
        
        # Role info
        if st.session_state.user_profile:
            st.markdown("#### 👤 Current Profile")
            profile = st.session_state.user_profile
            st.markdown(f"""
            **{profile['icon']} {profile['name']}**  
            *{profile['description']}*
            
            **Length:** {profile['length'].title()}  
            {f"**Focus:** {profile['focus']}" if profile.get('focus') else ""}
            """)
            
            if st.button("🔄 Change Profile", use_container_width=True):
                st.session_state.user_profile = None
                st.rerun()
        
        # Help section
        st.markdown("#### ❓ Help & Tips")
        with st.expander("💡 Usage Tips"):
            st.markdown("""
            **For best results:**
            - Use public YouTube videos
            - Videos under 2 hours work best
            - Choose the right profile for your needs
            - Specify focus areas for targeted summaries
            
            **Profile Guide:**
            - 🎓 **Student**: Study guides, learning objectives
            - 💻 **Developer**: Technical implementation
            - 🚀 **Entrepreneur**: Business opportunities
            - 🔬 **Researcher**: Academic analysis
            - 💼 **Business Pro**: Strategic insights
            - 🌟 **General**: Clear explanations
            """)

def main():
    """Main Streamlit app"""
    
    # Initialize summarizer
    @st.cache_resource
    def load_summarizer():
        return YouTubeSummarizer()
    
    try:
        summarizer = load_summarizer()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.error("💡 Install required packages: pip install yt-dlp openai-whisper google-generativeai python-dotenv")
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Main content area
    if not st.session_state.user_profile:
        # Role selection
        render_role_selection()
    else:
        # Input form
        url = render_input_form()
        
        # Process video if requested
        if st.session_state.processing and url:
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            def update_status(message):
                status_container.markdown(f"""
                <div class="status-container">
                    <div class="status-item">
                        <span class="status-icon">⚡</span>
                        <span>{message}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                # Process with status updates
                result = summarizer.process_video(
                    url, 
                    st.session_state.user_profile,
                    progress_callback=update_status
                )
                
                progress_bar.progress(100)
                
                if result['success']:
                    st.session_state.summary_result = result
                    st.session_state.processing = False
                    status_container.empty()
                    progress_bar.empty()
                    st.rerun()
                else:
                    st.session_state.processing = False
                    st.error(f"❌ {result['error']}")
                    
            except Exception as e:
                st.session_state.processing = False
                st.error(f"❌ Unexpected error: {str(e)}")
        
        # Show results if available
        if st.session_state.summary_result:
            render_summary_result(st.session_state.summary_result)
        
        # Show recent activity or examples
        if not st.session_state.processing and not st.session_state.summary_result:
            st.markdown("""
            <div style="text-align: center; margin: 3rem 0; color: #94a3b8;">
                <h3>🎯 Ready to Transform Your Learning</h3>
                <p>Paste a YouTube URL above to get started with your personalized summary!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Example videos for different profiles
            st.markdown("### 💡 Example Videos to Try")
            
            examples = {
                'student': [
                    "Khan Academy tutorials",
                    "Educational lectures", 
                    "Study technique videos"
                ],
                'developer': [
                    "Programming tutorials",
                    "Tech conferences",
                    "Code review sessions"
                ],
                'entrepreneur': [
                    "Startup pitches",
                    "Business strategy talks",
                    "Marketing case studies"
                ],
                'researcher': [
                    "Academic presentations",
                    "Research methodology",
                    "Scientific discussions"
                ],
                'professional': [
                    "Industry insights",
                    "Leadership talks",
                    "Business analysis"
                ],
                'general': [
                    "TED talks",
                    "Documentary clips",
                    "How-to videos"
                ]
            }
            
            user_type = st.session_state.user_profile['type']
            if user_type in examples:
                st.markdown(f"**Perfect for {st.session_state.user_profile['icon']} {st.session_state.user_profile['name']}:**")
                for example in examples[user_type]:
                    st.markdown(f"• {example}")

if __name__ == "__main__":
    main()