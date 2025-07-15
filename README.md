# 🎥 InsightReel — From Video to Insight, Instantly

**InsightReel** is an AI-powered YouTube summarizer that transforms long videos into concise, actionable insights personalized for your learning or professional goals.

No more scrubbing through long videos. Whether you're a student, developer, entrepreneur, or curious learner — InsightReel saves you time and delivers what matters.

---

## 🚀 Features

- 🤖 **AI-Enhanced Summaries** — Powered by Gemini or OpenAI models
- 🧑‍💼 **Personalized Focus** — For students, developers, professionals, entrepreneurs, and more
- 🧠 **Key Concepts Extraction** — Keywords, technical terms, action verbs
- ✨ **Top 3 Takeaways** — Immediate insights upfront
- 📋 **Structured Output** — Emojis, markdown, action steps, and more
- 📊 **Analytics Section** — Word count, duration, segments, and generation time
- 💬 **Quote of the Day** (optional) — Inspirational or insightful moments

- **Smart Processing**:
  - Automatic video transcription using Whisper
  - AI-enhanced summaries with Google Gemini
  - Intelligent content analysis
  - Key concept extraction
  - Technical term detection

- **Advanced Caching**:
  - Efficient video processing
  - Cached transcripts and metadata
  - Reduced processing time

## 🚀 Getting Started

### Prerequisites

```bash
# Required Python packages
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the App

```bash
streamlit run app.py
```
```
python run main.py
```

## 📋 Usage

1. Select your user profile (Student, Developer, etc.)
2. Paste a YouTube URL
3. Customize summary length and focus (optional)
4. Click "Generate Summary"
5. View and download your personalized summary

## 🛠️ Technical Details

### Core Components

- **Frontend**: Streamlit with custom CSS
- **Video Processing**: yt-dlp
- **Transcription**: OpenAI Whisper
- **AI Summary**: Google Gemini
- **Caching System**: Local file-based cache

### Key Features

- Automatic video transcription
- Smart content analysis
- Role-based personalization
- Markdown summary export
- Progress tracking
- Cache management

## 📊 Summary Output

- Video metadata
- Duration and segments
- Key concepts and topics
- Technical terms
- Action items
- Personalized insights

## ⚙️ Configuration Options

- Summary length:
  - ⚡ Quick (2-3 min read)
  - 📋 Standard (5-7 min read)
  - 📖 Detailed (10+ min read)
- Custom focus areas
- Role-specific adaptations

## 🔒 Limitations

- Maximum video length: 2 hours
- Requires stable internet connection
- API key needed for AI features

## 🤝 Contributing

Feel free to:
- Open issues
- Submit PRs
- Suggest improvements
- Report bugs

## Additional System Requirements:

FFmpeg needs to be installed on your system:

  -Windows: choco install ffmpeg
  -Mac: brew install ffmpeg
  -Linux: sudo apt install ffmpeg

## 🙏 Acknowledgments

- OpenAI Whisper for transcription
- Google Gemini for AI summaries
- Streamlit for the web interface
- yt-dlp for video processing

## 🔮 Future Improvements

- [ ] Multiple language support
- [ ] More user roles
- [ ] Advanced analytics
- [ ] API endpoint
- [ ] Batch processing
- [ ] Custom themes