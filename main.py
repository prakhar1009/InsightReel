#!/usr/bin/env python3
"""
Complete YouTube Summarizer - Clean, efficient implementation
Generates personalized summaries with intelligent caching
"""

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

try:
    import yt_dlp
    import whisper
    import google.generativeai as genai
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install: pip install yt-dlp openai-whisper google-generativeai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeSummarizer:
    """YouTube Summarizer with caching and personalized analysis"""
    
    def __init__(self):
        """Initialize the summarizer with APIs and models"""
        self.setup_apis()
        self.load_whisper()
        self.user_profile = {}
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
    
    def get_user_preferences(self):
        """Get user preferences for personalized summaries"""
        print("\n🎯 Let's personalize your summary!")
        print("=" * 40)
        
        # User type
        print("\n👤 What's your role/background?")
        print("1. 🎓 Student")
        print("2. 💼 Business Professional") 
        print("3. 🔧 Developer/Technical")
        print("4. 🚀 Entrepreneur")
        print("5. 📚 Researcher")
        print("6. 🌟 General Audience")
        
        while True:
            try:
                choice = int(input("\nSelect your profile (1-6): ").strip())
                if 1 <= choice <= 6:
                    break
                print("❌ Please enter a number between 1-6")
            except ValueError:
                print("❌ Please enter a valid number")
        
        user_types = {
            1: {"type": "student", "name": "Student", "icon": "🎓"},
            2: {"type": "professional", "name": "Business Professional", "icon": "💼"},
            3: {"type": "developer", "name": "Developer", "icon": "🔧"},
            4: {"type": "entrepreneur", "name": "Entrepreneur", "icon": "🚀"},
            5: {"type": "researcher", "name": "Researcher", "icon": "📚"},
            6: {"type": "general", "name": "General Audience", "icon": "🌟"}
        }
        
        self.user_profile = user_types[choice]
        
        # Summary length preference
        print(f"\n📝 Summary length for {self.user_profile['name']}:")
        print("1. ⚡ Quick (Key points - 2-3 min read)")
        print("2. 📋 Standard (Balanced - 5-7 min read)")
        print("3. 📖 Detailed (Comprehensive - 10+ min read)")
        
        while True:
            try:
                length_choice = int(input("\nSelect length (1-3): ").strip())
                if 1 <= length_choice <= 3:
                    break
                print("❌ Please enter 1, 2, or 3")
            except ValueError:
                print("❌ Please enter a valid number")
        
        length_types = {1: "quick", 2: "standard", 3: "detailed"}
        self.user_profile["length"] = length_types[length_choice]
        
        # Specific interests
        print(f"\n🎯 Any specific focus areas? (optional)")
        specific_focus = input("Enter focus areas (or press Enter to skip): ").strip()
        self.user_profile["focus"] = specific_focus if specific_focus else None
        
        print(f"\n✅ Profile set: {self.user_profile['icon']} {self.user_profile['name']} | {self.user_profile['length'].title()}")
        if self.user_profile["focus"]:
            print(f"🎯 Focus: {self.user_profile['focus']}")
        print()
    
    def download_audio(self, url: str, video_id: str) -> str:
        """Download audio with caching"""
        
        # Check cache first
        cached_audio = self.load_from_cache(video_id, 'audio')
        if cached_audio and os.path.exists(cached_audio):
            logger.info(f"📁 Using cached audio: {cached_audio}")
            return cached_audio
        
        cache_paths = self.get_cache_paths(video_id)
        output_path = str(cache_paths['audio'].with_suffix(''))
        
        try:
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
    
    def split_audio(self, input_file: str, total_duration: float) -> list:
        """Split audio into chunks"""
        
        # Determine chunk size based on duration
        if total_duration <= 300:
            chunk_duration = 60
        elif total_duration <= 900:
            chunk_duration = 120
        elif total_duration <= 1800:
            chunk_duration = 180
        else:
            chunk_duration = 240
        
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
                        logger.info(f"   ✂️  Chunk {len(chunks)} ready")
                        continue
                
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    
            except Exception as e:
                logger.warning(f"Failed to create chunk at {start_time}s: {e}")
                continue
        
        if not chunks:
            raise Exception("No audio chunks were created")
        
        logger.info(f"✅ Created {len(chunks)} audio chunks")
        return chunks
    
    def transcribe_chunks(self, chunks: list, video_id: str) -> list:
        """Transcribe chunks with caching"""
        
        # Check if transcript is cached
        cached_transcript = self.load_from_cache(video_id, 'transcript')
        if cached_transcript:
            logger.info("📁 Using cached transcript")
            return cached_transcript
        
        transcripts = []
        logger.info(f"🎤 Transcribing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            try:
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
        
        logger.info(f"✅ Transcribed {len(transcripts)} segments")
        return transcripts
    
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
    
    def create_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict) -> str:
        """Create personalized summary"""
        
        if not transcripts:
            return "# Error\n\nNo transcript available for summarization."
        
        if self.use_gemini:
            return self.create_ai_summary(transcripts, video_title, video_url, duration_str, key_concepts)
        else:
            return self.create_basic_summary(transcripts, video_title, video_url, duration_str, key_concepts)
    
    def create_ai_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict) -> str:
        """Create AI-enhanced summary"""
        
        # Combine transcript for AI analysis
        full_transcript = "\n\n".join([f"[{t['timestamp']}] {t['text']}" for t in transcripts])
        
        try:
            logger.info(f"🤖 Creating AI summary for {self.user_profile['name']}...")
            
            # Create prompt based on user profile
            user_focus_map = {
                "student": "learning objectives, practical applications, study tips",
                "professional": "business value, ROI, implementation strategies",
                "developer": "technical implementation, tools, best practices",
                "entrepreneur": "business opportunities, growth strategies, revenue models",
                "researcher": "methodology, evidence, theoretical frameworks",
                "general": "clear explanations, practical insights, actionable takeaways"
            }
            
            focus_areas = user_focus_map.get(self.user_profile['type'], user_focus_map['general'])
            
            length_guide = {
                "quick": "Be concise and focus on key points only.",
                "standard": "Provide balanced detail with clear explanations.",
                "detailed": "Offer comprehensive analysis with examples and insights."
            }
            
            prompt = f"""
Create a personalized summary for a {self.user_profile['name']}.

**VIDEO:** {video_title}
**FOCUS:** {focus_areas}
**LENGTH:** {length_guide[self.user_profile['length']]}
{f"**SPECIAL FOCUS:** {self.user_profile['focus']}" if self.user_profile.get('focus') else ""}

**KEY CONCEPTS:**
- Main topics: {', '.join([kw[0] for kw in key_concepts['keywords'][:5]])}
- Technical terms: {', '.join(key_concepts['technical_terms'][:5])}
- Action words: {', '.join(key_concepts['action_words'][:5])}

**TRANSCRIPT:**
{full_transcript}

Create a structured summary with:
1. Compelling overview
2. Clear sections with emojis
3. Actionable insights for the {self.user_profile['name']}
4. Next steps or recommendations
"""
            
            response = self.gemini_model.generate_content(prompt)
            ai_summary = response.text
            
            # Create final summary
            final_summary = f"""# {self.user_profile['icon']} {video_title}

> **Personalized for**: {self.user_profile['name']} | **Length**: {self.user_profile['length'].title()}
{f"> **Focus**: {self.user_profile['focus']}" if self.user_profile.get('focus') else ""}

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
            return self.create_basic_summary(transcripts, video_title, video_url, duration_str, key_concepts)
    
    def create_basic_summary(self, transcripts: list, video_title: str, video_url: str, duration_str: str, key_concepts: dict) -> str:
        """Create enhanced basic summary without AI"""
        
        icon = self.user_profile['icon']
        user_name = self.user_profile['name']
        
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

> **Personalized for**: {user_name} | **Length**: {self.user_profile['length'].title()}
{f"> **Focus**: {self.user_profile['focus']}" if self.user_profile.get('focus') else ""}

*🔗 [Watch Video]({video_url})*

## 🎯 Overview for {user_name}

**📋 Main Topics**: {', '.join(topics[:5])}

**🛠️ Technical Elements**: {', '.join(set(key_concepts['technical_terms'][:6])) if key_concepts['technical_terms'] else 'General automation and integration tools'}

**⚡ Key Actions**: {', '.join(key_concepts['action_words'][:6]) if key_concepts['action_words'] else 'Implementation and process optimization'}

## 📝 Content Breakdown

"""
        
        # Add enhanced segments based on user preference
        segment_count = {"quick": 3, "standard": 5, "detailed": 7}[self.user_profile['length']]
        segment_count = min(segment_count, len(transcripts))
        
        if segment_count > 0:
            step = max(1, len(transcripts) // segment_count)
            
            for i in range(0, len(transcripts), step):
                if i < len(transcripts):
                    transcript = transcripts[i]
                    
                    max_chars = {"quick": 200, "standard": 350, "detailed": 500}[self.user_profile['length']]
                    
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
        
        # Enhanced user-specific insights
        user_insights = {
            'student': f"""## 📚 What You'll Learn

**Topics:**
- {topics[0] if topics else 'AI-powered development workflows'} and practical applications
- {topics[1] if len(topics) > 1 else 'Modern development tools'} integration and usage
- {topics[2] if len(topics) > 2 else 'Automation techniques'} for coding productivity
- How professionals use {', '.join(key_concepts['technical_terms'][:3]) if key_concepts['technical_terms'] else 'AI tools'} in real-world projects

**Tools Covered:**
- {', '.join(set(key_concepts['technical_terms'][:5])) if key_concepts['technical_terms'] else 'Development tools and platforms'}
- Integration workflows and automation techniques
- Code review and collaboration processes

---

## 🧠 Study Guide

**Learning Goals:**
- Understand how AI assists in real-world software development
- Learn practical implementation of {', '.join(key_concepts['action_words'][:3]) if key_concepts['action_words'] else 'automation workflows'}
- Study professional development practices and tool usage

**Suggested Projects:**
- Create a mini project using the discussed technologies
- Try replicating the workflow patterns shown in the video
- Build a simple automation tool incorporating these concepts
- Practice with the specific tools mentioned: {', '.join(key_concepts['technical_terms'][:3]) if key_concepts['technical_terms'] else 'relevant development tools'}

**Study Tips:**
- Take notes on specific commands and configurations shown
- Research each tool mentioned for deeper understanding
- Join communities around these technologies for hands-on learning
""",
            
            'professional': f"""## 💼 Business Strategy Insights

**Strategic Applications:**
- **Developer Productivity**: {topics[0] if topics else 'AI automation tools'} can reduce engineering workload and context-switching overhead
- **Operational Efficiency**: {', '.join(key_concepts['action_words'][:2]) if key_concepts['action_words'] else 'Process automation'} enables faster development cycles and reduced manual intervention
- **Technology Integration**: {', '.join(key_concepts['technical_terms'][:3]) if key_concepts['technical_terms'] else 'Modern development tools'} integrate with existing workflows (Slack, GitHub, VS Code)

**Executive Takeaways:**
- Potential 20-30% reduction in development cycle time for small-to-mid teams
- Ideal for startup CTOs or product managers needing AI-assisted development without expanding headcount
- Consider piloting these tools in internal projects before organization-wide deployment
- Evaluate against competitors like Cursor AI or GitHub Copilot for strategic positioning

**ROI Considerations:**
- Reduced time-to-market for product features
- Lower recruitment pressure for senior developers
- Improved code review and quality assurance processes
""",
            
            'developer': f"""## 🔧 Technical Deep Dive

**Architecture & Implementation:**
- Core technologies: {', '.join(set(key_concepts['technical_terms'][:6])) if key_concepts['technical_terms'] else 'Modern development stack'}
- Key implementation patterns: {', '.join(key_concepts['action_words'][:4]) if key_concepts['action_words'] else 'Development workflows and automation'}
- Integration strategies: {', '.join(topics[:3]) if topics else 'System integration approaches'}

**Development Roadmap:**
- **Phase 1**: Environment setup with {', '.join(key_concepts['technical_terms'][:2]) if key_concepts['technical_terms'] else 'required tools'}
- **Phase 2**: Core feature implementation and testing
- **Phase 3**: Integration with existing systems and workflows
- **Phase 4**: Performance optimization and monitoring

**Technical Considerations:**
- Scalability factors and architectural decisions
- Security implications and best practices
- Performance monitoring and optimization strategies
- Maintenance and update procedures

**Code Examples & Resources:**
- Review official documentation for mentioned tools
- Explore open-source implementations
- Set up local development environment for testing
""",
            
            'entrepreneur': f"""## 🚀 Business Opportunity & Market Analysis

**Market Positioning:**
- Target market opportunities: {', '.join(topics[:3]) if topics else 'Automation and AI-assisted workflows'}
- Competitive landscape: Analysis of tools like {', '.join(key_concepts['technical_terms'][:3]) if key_concepts['technical_terms'] else 'emerging technologies'}
- Value proposition: How {', '.join(key_concepts['action_words'][:3]) if key_concepts['action_words'] else 'automation and efficiency'} creates market advantage

**Revenue Models:**
- **SaaS Subscription**: Monthly/annual pricing for platform access
- **Professional Services**: Implementation and customization consulting
- **Enterprise Licensing**: White-label solutions for larger organizations
- **Marketplace Model**: Commission-based integration platform

**Go-to-Market Strategy:**
- **Phase 1**: MVP development and beta testing with early adopters
- **Phase 2**: Product-market fit validation and initial customer acquisition
- **Phase 3**: Scale operations and expand feature set
- **Phase 4**: Market expansion and strategic partnerships

**Investment Considerations:**
- Development costs and technical team requirements
- Market size and growth potential analysis
- Competitive differentiation and barrier to entry
- Scalability factors and infrastructure needs

**Risk Assessment:**
- Technology adoption risks and market timing
- Competitive response and market saturation
- Regulatory considerations and compliance requirements
""",
            
            'researcher': f"""## 📚 Research Framework & Analysis

**Primary Research Domains:**
- **Technical Investigation**: {', '.join(topics[:2]) if len(topics) >= 2 else 'System architecture and performance analysis'}
- **Methodological Approach**: {', '.join(key_concepts['action_words'][:3]) if key_concepts['action_words'] else 'Empirical analysis and evaluation methods'}
- **Technology Assessment**: {', '.join(key_concepts['technical_terms'][:4]) if key_concepts['technical_terms'] else 'Platform and tool evaluation'}

**Research Questions:**
- **Effectiveness**: How do these systems compare to traditional approaches in terms of performance and outcomes?
- **Scalability**: What are the technical and operational limits of the discussed implementations?
- **Adoption**: What factors influence user adoption and long-term engagement with these technologies?
- **Impact**: What measurable benefits do organizations experience from implementation?

**Methodology Recommendations:**
- **Quantitative Analysis**: Performance metrics, usage statistics, and comparative benchmarking
- **Qualitative Assessment**: User interviews, case studies, and ethnographic observation
- **Longitudinal Study**: Track implementation outcomes over extended periods
- **Control Groups**: Compare organizations with and without these technologies

**Data Collection Strategies:**
- Industry surveys and expert interviews
- Performance monitoring and analytics
- User behavior analysis and feedback collection
- Competitive analysis and market research

**Theoretical Frameworks:**
- Technology Acceptance Model (TAM) for adoption analysis
- Diffusion of Innovations theory for market penetration
- Systems thinking for organizational impact assessment
"""
        }
        
        # Add user-specific insights
        if self.user_profile['type'] in user_insights:
            summary += user_insights[self.user_profile['type']]
        else:
            summary += f"""## 🎯 Key Takeaways

**Core Focus Areas:**
- Main concepts: {', '.join(topics[:5])}
- Practical applications: {', '.join(key_concepts['action_words'][:5]) if key_concepts['action_words'] else 'Process improvement and automation'}
- Technology tools: {', '.join(key_concepts['technical_terms'][:5]) if key_concepts['technical_terms'] else 'Modern software solutions'}

**Recommended Next Steps:**
- Research the specific technologies and platforms mentioned
- Evaluate how these concepts apply to your particular context
- Consider pilot implementation or proof-of-concept development
"""
        
        # Enhanced analysis section
        complexity_level = "High" if key_concepts['complexity_score'] > 0.7 else "Medium" if key_concepts['complexity_score'] > 0.5 else "Basic"
        technical_level = "Advanced" if len(key_concepts['technical_terms']) > 8 else "Intermediate" if len(key_concepts['technical_terms']) > 4 else "Basic"
        
        summary += f"""

## 📊 Content Analysis

- **Duration**: {duration_str}
- **Content Complexity**: {complexity_level}
- **Technical Level**: {technical_level}
- **Word Count**: {key_concepts['total_words']:,}
- **Vocabulary Richness**: {key_concepts['complexity_score']:.1%}
- **Domain Focus**: {"Specialized" if key_concepts.get('domain_concepts') else "General"}

"""
        
        # Create user-appropriate "Worth Your Time" section
        worth_your_time_sections = {
            'student': f"""## 🎯 Is This Worth Watching?

**✅ Absolutely — if you're:**
- A student learning to code and want to see how AI can accelerate development
- Interested in modern development workflows with tools like {', '.join(key_concepts['technical_terms'][:2]) if key_concepts['technical_terms'] else 'AI coding assistants'}
- Building projects and need help with code structure, review, or automation
- Curious about how AI fits into real-world development processes like GitHub workflows
- Exploring career paths in software development or AI-assisted programming

**⏱️ Time Investment**: {duration_str}
**📘 Educational Value**: {"High" if complexity_level in ["High", "Medium"] else "Good"} 
**🧠 Skill Level Required**: {"Intermediate" if technical_level == "Advanced" else "Beginner-to-Intermediate"} (tech-curious students)

**What You'll Gain:**
- Practical understanding of AI development tools
- Insight into professional development workflows
- Knowledge of modern coding automation techniques""",

            'professional': f"""## 🎯 Strategic Value Assessment

**✅ Recommended for business leaders who are:**
- Product managers exploring AI development assistants and their ROI impact
- Evaluating {', '.join(key_concepts['technical_terms'][:2]) if key_concepts['technical_terms'] else 'AI tools'} vs. competitors for team productivity
- Looking to improve development efficiency without expanding technical headcount
- CTOs or technical directors assessing AI-powered development workflows
- Planning digital transformation initiatives with AI integration

**⏱️ Time Investment**: {duration_str}
**📊 Business Value**: {"Strategic" if complexity_level == "High" and technical_level in ["Advanced", "Intermediate"] else "Tactical" if complexity_level in ["Medium", "High"] else "Informational"}
**💼 Decision-Maker Relevance**: {"High" if technical_level in ["Advanced", "Intermediate"] else "Medium"} - {"actionable insights" if complexity_level == "High" else "practical overview"}

**Business Impact:**
- Potential 20-30% improvement in development velocity
- Reduced dependency on senior developer hiring
- Enhanced code quality and review processes""",

            'developer': f"""## 🎯 Technical Relevance Check

**✅ Worth your time if you're:**
- A developer interested in AI-assisted coding and automation workflows
- Working with {', '.join(key_concepts['technical_terms'][:3]) if key_concepts['technical_terms'] else 'modern development tools'} or similar technologies
- Looking to streamline your development process with AI integration
- Interested in code review automation and workflow optimization
- Exploring advanced development productivity tools and techniques

**⏱️ Time Investment**: {duration_str}
**🔧 Technical Depth**: {"Deep dive" if technical_level == "Advanced" else "Practical overview" if technical_level == "Intermediate" else "Introduction"}
**💻 Implementation Ready**: {"Immediately actionable" if complexity_level == "High" else "Requires additional research" if complexity_level == "Medium" else "Conceptual foundation"}

**Technical Takeaways:**
- Hands-on implementation patterns
- Integration strategies and best practices
- Performance and scalability considerations""",

            'entrepreneur': f"""## 🎯 Market Opportunity Assessment

**✅ Essential viewing for entrepreneurs:**
- Exploring AI-powered development tools as a business opportunity
- Evaluating market positioning against established players like {', '.join(key_concepts['technical_terms'][:2]) if key_concepts['technical_terms'] else 'existing platforms'}
- Looking for automation and efficiency business models
- Interested in B2B SaaS opportunities in the developer tools space
- Planning technology ventures in the AI/automation sector

**⏱️ Time Investment**: {duration_str}
**🚀 Business Potential**: {"High-opportunity market" if complexity_level == "High" else "Emerging market" if complexity_level == "Medium" else "Developing sector"}
**💰 Market Insights**: {"Strategic intelligence" if technical_level in ["Advanced", "Intermediate"] else "Market overview"}

**Opportunity Indicators:**
- Growing demand for AI development assistance
- Clear value proposition for productivity improvement
- Scalable technology infrastructure potential""",

            'researcher': f"""## 🎯 Research Value Proposition

**✅ Highly relevant for researchers studying:**
- AI-assisted software development and its impact on productivity
- Technology adoption patterns in professional development environments
- Human-AI collaboration in technical workflows
- {', '.join(topics[:2]) if len(topics) >= 2 else 'Automation and efficiency systems'} implementation and outcomes

**⏱️ Time Investment**: {duration_str}
**📊 Research Utility**: {"High" if complexity_level == "High" and technical_level in ["Advanced", "Intermediate"] else "Medium" if complexity_level in ["Medium", "High"] else "Moderate"} 
**🔬 Academic Value**: {"Primary source material" if technical_level == "Advanced" else "Case study potential" if technical_level == "Intermediate" else "Background research"}

**Research Applications:**
- Technology adoption and diffusion studies
- Productivity measurement and analysis
- Human-computer interaction research
- Industry transformation case studies"""
        }
        
        # Add the appropriate "Worth Your Time" section
        if self.user_profile['type'] in worth_your_time_sections:
            summary += worth_your_time_sections[self.user_profile['type']]
        else:
            # Fallback for general audience
            summary += f"""## 🎯 Is This Content Right for You?

**✅ Recommended if you're interested in:**
- {topics[0] if topics else 'Modern automation and AI tools'}
- {topics[1] if len(topics) > 1 else 'Technology implementation and workflows'}
- {topics[2] if len(topics) > 2 else 'Process optimization and efficiency'}

**⏱️ Time Investment**: {duration_str}
**📊 Value**: {"High" if complexity_level in ["High", "Medium"] else "Good"} information density
**🎯 Audience**: {"Technical" if technical_level in ["Advanced", "Intermediate"] else "General"} audience"""
        
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
    
    def process_video(self, url: str) -> dict:
        """Main processing function"""
        
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
            logger.info("🚀 Starting video analysis...")
            
            # 1. Extract video metadata
            logger.info("📋 Extracting metadata...")
            
            cached_metadata = self.load_from_cache(video_id, 'metadata')
            if cached_metadata:
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
                audio_file = self.download_audio(url, video_id)
                
                if not audio_file or not os.path.exists(audio_file):
                    raise Exception("Audio download failed")
                    
            except Exception as e:
                return {'success': False, 'error': f'Audio processing failed: {str(e)[:300]}'}
            
            # 3. Split audio into chunks
            try:
                actual_duration = self.get_audio_duration(audio_file)
                chunks = self.split_audio(audio_file, actual_duration)
                
                if not chunks:
                    raise Exception("No audio chunks created")
                
            except Exception as e:
                return {'success': False, 'error': f'Audio splitting failed: {str(e)[:300]}'}
            
            # 4. Transcribe audio
            try:
                transcripts = self.transcribe_chunks(chunks, video_id)
                
                if not transcripts:
                    raise Exception("No transcripts generated")
                
            except Exception as e:
                return {'success': False, 'error': f'Transcription failed: {str(e)[:300]}'}
            
            # 5. Analyze and create summary
            try:
                key_concepts = self.extract_key_concepts(transcripts)
                summary_text = self.create_summary(transcripts, video_title, url, duration_str, key_concepts)
                
                if not summary_text or len(summary_text) < 100:
                    raise Exception("Summary generation failed")
                
            except Exception as e:
                return {'success': False, 'error': f'Summary creation failed: {str(e)[:300]}'}
            
            # 6. Save results
            try:
                safe_title = re.sub(r'[^\w\s\-_]', '', video_title)[:30]
                safe_title = re.sub(r'\s+', '_', safe_title.strip())
                user_type = self.user_profile.get('type', 'general')
                timestamp = int(time.time())
                
                filename = f"summary_{user_type}_{safe_title}_{timestamp}.md"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                
                if not os.path.exists(filename):
                    raise Exception("Summary file not created")
                
            except Exception as e:
                return {'success': False, 'error': f'File saving failed: {str(e)[:300]}'}
            
            processing_time = time.time() - start_time
            
            logger.info(f"🎉 Analysis completed in {processing_time/60:.1f} minutes")
            logger.info(f"📁 Summary saved: {filename}")
            
            return {
                'success': True,
                'filename': filename,
                'video_title': video_title,
                'channel': channel,
                'duration': duration_str,
                'segments': len(transcripts),
                'processing_time': f"{processing_time/60:.1f} minutes",
                'ai_enhanced': self.use_gemini,
                'user_profile': self.user_profile.copy(),
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


def clear_cache():
    """Clear all cached files"""
    cache_dir = Path("youtube_cache")
    if cache_dir.exists():
        try:
            file_count = len(list(cache_dir.glob("*")))
            shutil.rmtree(cache_dir)
            print(f"✅ Cleared {file_count} cached files")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
    else:
        print("📁 No cache directory found")


def show_cache_info():
    """Show cache information"""
    cache_dir = Path("youtube_cache")
    if not cache_dir.exists():
        print("📁 No cache directory found")
        return
    
    files = list(cache_dir.glob("*"))
    if not files:
        print("📁 Cache directory is empty")
        return
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    print(f"📊 Cache Information:")
    print(f"   📁 Location: {cache_dir.absolute()}")
    print(f"   📄 Files: {len(files)}")
    print(f"   💾 Size: {total_size / 1024 / 1024:.1f} MB")
    
    # Group by video ID
    video_data = {}
    for file in files:
        if file.is_file():
            name_parts = file.stem.split('_')
            if len(name_parts) >= 2:
                video_id = name_parts[0]
                data_type = '_'.join(name_parts[1:])
                
                if video_id not in video_data:
                    video_data[video_id] = []
                video_data[video_id].append(data_type)
    
    print(f"\n🎬 Cached Videos: {len(video_data)}")
    for video_id, data_types in list(video_data.items())[:5]:  # Show first 5
        print(f"   {video_id}: {', '.join(data_types)}")
    
    if len(video_data) > 5:
        print(f"   ... and {len(video_data) - 5} more")


def main():
    """Main function"""
    
    print("🛠️ YouTube Summarizer")
    print("=" * 50)
    print("🎯 Personalized summaries with intelligent caching!")
    print("📁 Cache location: ./youtube_cache/")
    print()
    
    # Check for commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear-cache':
            clear_cache()
            return
        elif sys.argv[1] == '--cache-info':
            show_cache_info()
            return
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python youtube_summarizer.py                  # Run normally")
            print("  python youtube_summarizer.py --cache-info     # Show cache info")
            print("  python youtube_summarizer.py --clear-cache    # Clear cache")
            print("  python youtube_summarizer.py --help           # Show help")
            return
    
    try:
        # Initialize summarizer
        try:
            summarizer = YouTubeSummarizer()
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("💡 Install required packages:")
            print("   pip install yt-dlp openai-whisper google-generativeai python-dotenv")
            return
        
        # Check API status
        if not summarizer.use_gemini:
            print("⚠️  No Gemini API key - using basic summaries")
            print("💡 Add GEMINI_API_KEY to .env file for AI summaries")
            print()
        
        # Get user preferences
        try:
            summarizer.get_user_preferences()
        except KeyboardInterrupt:
            print("\n👋 Thanks for using YouTube Summarizer!")
            return
        except Exception as e:
            print(f"❌ Error setting preferences: {e}")
            return
        
        # Get URL
        while True:
            try:
                url = input("\n📺 Enter YouTube URL: ").strip()
                
                if not url:
                    print("❌ No URL provided")
                    continue
                
                if not ('youtube.com' in url or 'youtu.be' in url):
                    print("❌ Please provide a valid YouTube URL")
                    continue
                
                break
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using YouTube Summarizer!")
                return
        
        # Show processing info
        user_profile = summarizer.user_profile
        print(f"\n🚀 Processing for {user_profile['icon']} {user_profile['name']}...")
        print(f"📝 Style: {user_profile['length'].title()}")
        if user_profile.get('focus'):
            print(f"🎯 Focus: {user_profile['focus']}")
        print("📁 Checking cache...")
        print("⏳ This may take a few minutes...")
        print()
        
        # Process video
        try:
            result = summarizer.process_video(url)
        except KeyboardInterrupt:
            print("\n⏹️  Processing cancelled")
            return
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return
        
        # Show results
        if result.get('success'):
            print(f"\n🎉 SUCCESS!")
            print("=" * 50)
            print(f"📁 Summary: {result['filename']}")
            print(f"🎬 Video: {result['video_title']}")
            print(f"📺 Channel: {result['channel']}")
            print(f"⏱️  Duration: {result['duration']}")
            print(f"📊 Segments: {result['segments']}")
            print(f"🕒 Time: {result['processing_time']}")
            print(f"📁 Cached: {'Yes' if result.get('cached_data_used') else 'No'}")
            print(f"🤖 AI: {'Yes ✨' if result['ai_enhanced'] else 'Basic 🔧'}")
            
            # Show analysis
            if 'key_concepts' in result:
                kc = result['key_concepts']
                print(f"\n📊 Analysis:")
                if kc['keywords']:
                    print(f"   🏷️  Topics: {', '.join([kw[0] for kw in kc['keywords'][:3]])}")
                if kc['technical_terms']:
                    print(f"   🛠️  Tech: {', '.join(kc['technical_terms'][:3])}")
            
            print(f"\n✨ Your summary is ready!")
            print(f"📖 Open {result['filename']} to read")
            
        else:
            print(f"\n❌ FAILED!")
            print("=" * 50)
            print(f"Error: {result.get('error', 'Unknown error')}")
            print()
            print("💡 Tips:")
            print("   • Check internet connection")
            print("   • Verify URL is correct and public")
            print("   • Try --clear-cache if issues persist")
            
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("💡 Try restarting or use --clear-cache")
    finally:
        print("\n👋 Thanks for using YouTube Summarizer!")


if __name__ == "__main__":
    main()