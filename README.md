# AI Engineering Knowledge System

Turn short technical videos (reels, shorts, tutorials) into a structured, searchable knowledge library.

Upload a video or paste a link. The system extracts the audio, transcribes it, and uses an LLM to pull out a structured record: the problem, root cause, solution, trade-offs, when to use it and when not to, tools mentioned, and topics. Every record goes through a human review step before it is searchable. You can then search the library or ask it questions in plain English.

## Features

- **Video ingestion**: upload `.mp4`, `.mov`, `.webm` or `.mkv` files (up to 200 MB), or import from a URL with `yt-dlp`
- **Audio extraction**: FFmpeg converts video to 16 kHz mono WAV
- **Transcription**: local speech-to-text with `faster-whisper` (the `base` model, running on CPU). Non-English speech is translated to English.
- **Knowledge extraction**: Google Gemini returns a strict JSON record per video. It is told not to invent details, so missing fields stay empty.
- **Human review**: new items are `pending_review` until approved or rejected, and you can correct fields while reviewing
- **Search**: keyword search, topic filter, combined filter, and natural-language Q&A over approved items, with filenames cited
- **Topic taxonomy**: items are tagged from a fixed list (LLM Engineering, RAG, Agents, Production / Infra, Data / Search, Security)
- **Web UI**: `knowledge_ui_pro.html` provides Library, Search, Ask, and Import pages

## How it works

```
Video / URL
    |
    v
FFmpeg  ->  Whisper  ->  Gemini (structured JSON)  ->  Review  ->  Approved library
 (audio)   (transcript)      (knowledge record)                    (search / ask)
```

## Tech stack

Python 3.12, FastAPI, Uvicorn, faster-whisper, FFmpeg, yt-dlp, Google Gemini (`google-genai`), and a vanilla HTML/JS front end. Data is stored as files on disk, with no database yet.

## Project structure

```
AI_Engineering_Knowledge_System/
├── app/
│   └── main.py              # FastAPI app: all endpoints and the pipeline
├── data/
│   ├── uploads/             # original videos (git-ignored)
│   ├── audio/               # extracted WAV files
│   ├── transcripts/         # .txt transcripts
│   └── knowledge/           # .json knowledge records
├── knowledge_ui_pro.html    # front end
├── test_env.py              # checks that the API key loads
├── test_gemini.py           # checks the Gemini connection
├── requirements.txt
└── .env                     # GEMINI_API_KEY (never commit this)
```

## Getting started

### Prerequisites

- Python 3.12
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/<your-username>/AI_Engineering_Knowledge_System.git
cd AI_Engineering_Knowledge_System

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Verify your setup:

```bash
python test_env.py
python test_gemini.py
```

### Run

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- UI: open `knowledge_ui_pro.html` in your browser while the server is running

The first run downloads the Whisper model, so it takes a little longer.

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload a video |
| POST | `/extract-audio/{filename}` | Extract audio from an uploaded video |
| POST | `/transcribe/{filename}` | Transcribe an audio file |
| POST | `/extract-knowledge/{filename}` | Extract a knowledge record from a transcript |
| POST | `/process-video/{filename}` | Run the full pipeline on an uploaded video |
| POST | `/upload-and-process` | Upload and process in one call (auto-approves) |
| POST | `/upload-from-url?url=...` | Download from a URL and process (auto-approves) |
| GET | `/knowledge/pending-review` | Items awaiting review |
| PATCH | `/review-knowledge/{filename}` | Approve or reject, with optional corrections |
| GET | `/knowledge/approved` | All approved items |
| GET | `/knowledge/search?q=...` | Keyword search |
| GET | `/knowledge/by-topic/{topic}` | Filter by topic |
| GET | `/knowledge/filter?topic=...&q=...` | Combine topic and keyword |
| GET | `/knowledge/ask?question=...` | Natural-language Q&A over approved items |

### Example knowledge record

```json
{
  "problem": "Finding free online training resources to learn AI and ML engineering.",
  "root_cause": "",
  "solution": ["Take Nvidia's free self-paced foundational AI/ML courses."],
  "tradeoffs": [],
  "when_to_use": ["When looking to learn AI/ML engineering at no cost"],
  "when_not_to_use": [],
  "tools_mentioned": ["Nvidia deep learning AI courses"],
  "topics": [],
  "confidence": 0.9,
  "verification_status": "unverified",
  "source_transcript_file": "ml.txt",
  "review_status": "approved"
}
```

## Limitations and roadmap

Current limitations:

- Data lives in JSON and media files, which is fine for a personal library but not for scale
- The `/knowledge/ask` endpoint sends all approved items to the model, so it won't scale to a large library
- CORS is open to all origins, so it is intended for local use only
- Only the audio track is used, so on-screen text is not captured

Planned:

- [ ] On-screen text extraction (OCR) from video frames
- [ ] Automated knowledge verification against sources
- [ ] SQLite / PostgreSQL storage
- [ ] Embedding-based retrieval for Q&A
- [ ] Document generation from approved knowledge
- [ ] Automated tests

## Notes

- Keep `.env` out of version control (it is already in `.gitignore`)
- Uploaded filenames are replaced with random UUIDs and path traversal is blocked
- Restrict CORS before deploying anywhere public


