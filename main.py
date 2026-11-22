
from pathlib import Path
from uuid import uuid4

import json
from google import genai 

from fastapi import FastAPI, File, HTTPException, UploadFile

import subprocess

from faster_whisper import WhisperModel

from pydantic import BaseModel

app = FastAPI(
    title="AI Engineering Knowledge System",
    description="An AI-powered system for extracting and organizing engineering knowledge.",
    version="0.1.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from any origin (fine for local personal use)
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#audio
AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
model = WhisperModel("base", device="cpu", compute_type="int8")

#transcript
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

#genai
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

gemini_client = genai.Client()  # reads GEMINI_API_KEY from the environment automatically

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


@app.get("/")
def home():
    return {
        "message": "Welcome to the AI Engineering Knowledge System!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "AI Engineering Knowledge System",
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload MP4, MOV, WEBM or MKV.",
        )

    # Generate a safe, unique filename rather than using the
    # original filename supplied by the user.
    saved_filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / saved_filename
    file_size = 0

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum size is 200 MB.",
                    )

                output.write(chunk)

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    return {
        "message": "Video uploaded successfully",
        "filename": saved_filename,
        "size_bytes": file_size,
    }

# AUDIO_DIR


@app.post("/extract-audio/{filename}")
def extract_audio(filename: str):
    # Prevent paths or unexpected filenames from being supplied.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    input_path = UPLOAD_DIR / filename

    if not input_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file not found.",
        )

    output_filename = f"{Path(filename).stem}.wav"
    output_path = AUDIO_DIR / output_filename

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="FFmpeg could not extract audio.",
            )

    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=504,
            detail="Audio extraction took too long.",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg was not found. Check your PATH.",
        )

    return {
        "message": "Audio extracted successfully",
        "audio_filename": output_filename,
        "audio_path": str(output_path),
    }

#transcript
@app.post("/transcribe/{filename}")
def transcribe_audio(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    audio_path = AUDIO_DIR / filename

    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    segments, info = model.transcribe(str(audio_path))
    segments, info = model.transcribe(str(audio_path), task="translate")
    transcript = " ".join(segment.text.strip() for segment in segments)

    transcript_path = TRANSCRIPT_DIR / f"{Path(filename).stem}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    return {
        "message": "Transcription successful",
        "language": info.language,
        "transcript": transcript,
    }

#genai
EXTRACTION_SYSTEM_PROMPT = """You are an engineering knowledge extraction system.
You will be given a raw transcript from a short technical video/reel.
Extract a structured knowledge record in STRICT JSON with exactly these fields:

{
  "problem": string,
  "root_cause": string,
  "solution": [string],
  "tradeoffs": [string],
  "when_to_use": [string],
  "when_not_to_use": [string],
  "tools_mentioned": [string],
  "topics": [string],
  "confidence": number between 0 and 1,
  "verification_status": "unverified"
}

For "topics", choose zero or more topics ONLY from this list that genuinely apply to the content.
Do not invent new topic names. Use multiple topics if the content spans more than one.

- LLM Engineering: Rate Limits, Retries / Backoff, Model Routing, Structured Output, Tool Calling, Context Management, Caching
- RAG: Chunking, Retrieval, Reranking, Query Rewriting, Evaluation, Groundedness
- Agents: Planning, Tool Use, Memory, Failure Recovery, Human-in-the-loop
- Production / Infra: Queues, Concurrency, Observability, Logging, Deployment, Cost Control
- Data / Search: SQL, Vector DB, Hybrid Search, Metadata Filtering, Indexing
- Security: Secrets, Prompt Injection, Access Control, Data Isolation

If the content is not technical/engineering at all, return an empty "topics" list rather than forcing a match.

Rules:
- If the transcript does not contain enough information for a field, use an empty string or empty list. Do not invent details.
- confidence reflects how clearly the transcript supports this extraction, not how good the advice is.
- verification_status must always be "unverified" at this stage.
- Return ONLY the JSON object. No markdown, no explanation, no code fences.
"""


import time

@app.post("/extract-knowledge/{filename}")
def extract_knowledge(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    transcript_path = TRANSCRIPT_DIR / filename

    if not transcript_path.is_file():
        raise HTTPException(status_code=404, detail="Transcript file not found.")

    transcript_text = transcript_path.read_text(encoding="utf-8")

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=f"{EXTRACTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript_text}",
            )
            break  # success, exit retry loop
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(5)  # wait 5 seconds before retrying
                continue
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Model still unavailable after {max_retries} attempts: {last_error}",
        )

    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        knowledge = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. Raw output: {raw_text}",
        )

    knowledge["source_transcript_file"] = filename
    knowledge["review_status"] = "pending_review"

    knowledge_path = KNOWLEDGE_DIR / f"{Path(filename).stem}.json"
    knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")

    return {
        "message": "Knowledge extraction successful",
        "knowledge": knowledge,
        "knowledge_path": str(knowledge_path),
    }

class ReviewUpdate(BaseModel):
    review_status: str  # "approved" or "rejected"
    corrected_knowledge: dict | None = None  # optional edited fields

@app.patch("/review-knowledge/{filename}")
def review_knowledge(filename: str, update: ReviewUpdate):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if update.review_status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="review_status must be 'approved' or 'rejected'.",
        )

    knowledge_path = KNOWLEDGE_DIR / filename

    if not knowledge_path.is_file():
        raise HTTPException(status_code=404, detail="Knowledge file not found.")

    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))

    # If the user corrected any fields during review, apply them
    if update.corrected_knowledge:
        knowledge.update(update.corrected_knowledge)

    knowledge["review_status"] = update.review_status

    knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")

    return {
        "message": f"Knowledge item marked as {update.review_status}",
        "knowledge": knowledge,
    }

@app.get("/knowledge/pending-review")
def list_pending_review():
    pending = []

    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") == "pending_review":
            pending.append({"filename": file.name, "knowledge": data})

    return {"count": len(pending), "items": pending}


#Chain

@app.post("/process-video/{filename}")
def process_video(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    input_path = UPLOAD_DIR / filename
    if not input_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found.")

    # Step 1: Extract audio
    audio_filename = f"{Path(filename).stem}.wav"
    audio_path = AUDIO_DIR / audio_filename

    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="FFmpeg could not extract audio.")

    # Step 2: Transcribe
    segments, info = model.transcribe(str(audio_path))
    segments, info = model.transcribe(str(audio_path), task="translate")
    transcript = " ".join(segment.text.strip() for segment in segments)

    transcript_path = TRANSCRIPT_DIR / f"{Path(filename).stem}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    # Step 3: Extract knowledge (with retry on 503)
    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=f"{EXTRACTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}",
            )
            break
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(5)
                continue
            raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Model still unavailable after {max_retries} attempts: {last_error}",
        )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        knowledge = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. Raw output: {raw_text}",
        )

    knowledge["source_transcript_file"] = transcript_path.name
    knowledge["review_status"] = "pending_review"

    knowledge_path = KNOWLEDGE_DIR / f"{Path(filename).stem}.json"
    knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")

    return {
        "message": "Video processed successfully",
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "knowledge_path": str(knowledge_path),
        "knowledge": knowledge,
    }

#list approved knowledge

@app.get("/knowledge/approved")
def list_approved_knowledge():
    approved = []
    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") == "approved":
            approved.append({"filename": file.name, "knowledge": data})
    return {"count": len(approved), "items": approved}

#classic keyword search

@app.get("/knowledge/search")
def search_knowledge(q: str):
    q_lower = q.lower()
    results = []

    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") != "approved":
            continue

        searchable_text = " ".join([
            data.get("problem", ""),
            data.get("root_cause", ""),
            " ".join(data.get("solution", [])),
            " ".join(data.get("tools_mentioned", [])),
        ]).lower()

        if q_lower in searchable_text:
            results.append({"filename": file.name, "knowledge": data})

    return {"count": len(results), "results": results}

#list approved knowledge

@app.get("/knowledge/approved")
def list_approved_knowledge():
    approved = []
    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") == "approved":
            approved.append({"filename": file.name, "knowledge": data})
    return {"count": len(approved), "items": approved}



#keyword search

@app.get("/knowledge/search")
def search_knowledge(q: str):
    q_lower = q.lower()
    results = []

    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") != "approved":
            continue

        searchable_text = " ".join([
            data.get("problem", ""),
            data.get("root_cause", ""),
            " ".join(data.get("solution", [])),
            " ".join(data.get("tradeoffs", [])),
            " ".join(data.get("tools_mentioned", [])),
        ]).lower()

        if q_lower in searchable_text:
            results.append({"filename": file.name, "knowledge": data})

    return {"count": len(results), "results": results}


#topic

@app.get("/knowledge/by-topic/{topic}")
def knowledge_by_topic(topic: str):
    matches = []
    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        item_topics = [t.lower() for t in data.get("topics", [])]
        if data.get("review_status") == "approved" and topic.lower() in item_topics:
            matches.append({"filename": file.name, "knowledge": data})
    return {"count": len(matches), "items": matches}

#topic + keyword


@app.get("/knowledge/filter")
def filter_knowledge(topic: str | None = None, q: str | None = None):
    results = []
    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") != "approved":
            continue

        if topic:
            item_topics = [t.lower() for t in data.get("topics", [])]
            if topic.lower() not in item_topics:
                continue

        if q:
            searchable_text = " ".join([
                data.get("problem", ""),
                data.get("root_cause", ""),
                " ".join(data.get("solution", [])),
                " ".join(data.get("tools_mentioned", [])),
            ]).lower()
            if q.lower() not in searchable_text:
                continue

        results.append({"filename": file.name, "knowledge": data})

    return {"count": len(results), "results": results}

#natural language search

@app.get("/knowledge/ask")
def ask_knowledge(question: str):
    approved_items = []
    for file in KNOWLEDGE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("review_status") == "approved":
            approved_items.append({"filename": file.name, **data})

    if not approved_items:
        return {"answer": "No approved knowledge items yet.", "sources": []}

    context = json.dumps(approved_items, indent=2)

    prompt = f"""You are an engineering knowledge assistant.
Given this list of saved knowledge items (JSON) and a user's question,
identify which items are relevant and answer the question using ONLY
information from these items. If nothing is relevant, say so clearly.
Cite the filename(s) you used.

Knowledge items:
{context}

Question: {question}
"""

    response = gemini_client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
    )

    return {"question": question, "answer": response.text.strip()}







@app.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload MP4, MOV, WEBM or MKV.",
        )

    saved_filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / saved_filename
    file_size = 0

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum size is 200 MB.",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    # Now run the full pipeline on the file we just saved
    result = process_video(saved_filename)



    # Auto-approve since this came through the UI's one-click flow
    knowledge_path = Path(result["knowledge_path"])
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
    knowledge["review_status"] = "approved"
    knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")
    result["knowledge"] = knowledge
    return {
        "message": "Video uploaded and processed successfully",
        "original_filename": file.filename,
        "saved_filename": saved_filename,
        **result,
    }


#upload from youtube

import yt_dlp


@app.post("/upload-from-url")
def upload_from_url(url: str):
    video_id = uuid4().hex
    destination_template = str(UPLOAD_DIR / f"{video_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": destination_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "max_filesize": MAX_FILE_SIZE,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download video: {e}")

    saved_filename = f"{video_id}.mp4"
    destination = UPLOAD_DIR / saved_filename

    if not destination.is_file() or destination.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="Download completed but the expected video file was not found or is empty.",
        )

    result = process_video(saved_filename)

    knowledge_path = Path(result["knowledge_path"])
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
    knowledge["review_status"] = "approved"
    knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")
    result["knowledge"] = knowledge

    return {
        "message": "Video downloaded and processed successfully",
        "source_url": url,
        "saved_filename": saved_filename,
        **result,
    }