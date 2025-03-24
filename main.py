import asyncio
import os
import tempfile
import uuid
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Any
from threading import Semaphore
from fastapi import File, HTTPException, BackgroundTasks, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydub import AudioSegment
import torch
from apscheduler.schedulers.background import BackgroundScheduler
import whisper

class TranscriptionAPISettings(BaseSettings):
    tmp_dir: str = 'tmp'
    cors_origins: str = '*'
    cors_allow_credentials: bool = True
    cors_allow_methods: str = '*'
    cors_allow_headers: str = '*'
    whisper_model: str = 'large-v3'
    device: str = 'cuda'
    compute_type: str = 'float32'
    batch_size: int = 4
    language_code: str = 'ru'
    hf_api_key: str = ''
    max_file_size_mb: int = 4096
    file_loading_chunk_size_mb: int = 1024

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'allow'

settings = TranscriptionAPISettings()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(','),
    allow_headers=settings.cors_allow_headers.split(','),
)

trancription_tasks = {}
trancription_tasks_queue = Queue()
whisperx_models = None  # Переименуем для совместимости

def load_whisperx_models(device: str = "cpu") -> None:
    global whisperx_models
    whisperx_models = whisper.load_model(settings.whisper_model, device=device)

def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion error: {str(e)}") from e

class WhisperModelManager:
    def __enter__(self):
        if whisperx_models:
            whisperx_models.to(settings.device)
            print("Model moved to GPU.")
        return whisperx_models

    def __exit__(self, exc_type, exc_value, traceback):
        if whisperx_models:
            whisperx_models.to("cpu")
            print("Model moved to CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared.")

def transcribe_audio(audio_file_path: str) -> dict:
    original_path = audio_file_path
    wav_path = None
    try:
        if not audio_file_path.lower().endswith('.wav'):
            print(f"Converting {audio_file_path} to WAV...")
            wav_path = convert_to_wav(audio_file_path)
            audio_file_path = wav_path

        result = whisperx_models.transcribe(
            audio_file_path,
            language=settings.language_code if settings.language_code != "auto" else None
        )

        formatted_result = {
            "language": result["language"],
            "segments": [
                {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
                for segment in result["segments"]
            ]
        }
        return formatted_result

    finally:
        for path in [audio_file_path, original_path, wav_path]:
            if path and os.path.exists(path) and path != original_path:
                try:
                    os.remove(path)
                    print(f"Deleted temp file {path}")
                except PermissionError:
                    print(f"Permission error: Could not delete temp file {path}.")
                except Exception as e:
                    print(f"Error deleting temp file {path}: {str(e)}")

max_concurrent_tasks = 3
concurrent_tasks_semaphore = Semaphore(max_concurrent_tasks)

def transcription_worker() -> None:
    print("Transcription worker started.")
    while True:
        task_id, tmp_path = trancription_tasks_queue.get()
        print(f"Processing task {task_id} with file {tmp_path}")

        with concurrent_tasks_semaphore:
            try:
                with WhisperModelManager():
                    result = transcribe_audio(tmp_path)
                trancription_tasks[task_id].update({"status": "completed", "result": result})
                print(f"Task {task_id} completed.")
            except Exception as e:
                trancription_tasks[task_id].update({"status": "failed", "result": str(e)})
                print(f"Task {task_id} failed: {str(e)}")
            finally:
                trancription_tasks_queue.task_done()

scheduler = BackgroundScheduler()

def cleanup_tmp_dir() -> None:
    if trancription_tasks_queue.empty():
        for file_name in os.listdir(settings.tmp_dir):
            file_path = os.path.join(settings.tmp_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted leftover temp file: {file_path}")
            except Exception as e:
                print(f"Error deleting temp file {file_path}: {str(e)}")

def cleanup_completed_tasks() -> None:
    now = datetime.utcnow()
    completed_tasks = [
        task_id for task_id, task in trancription_tasks.items()
        if task["status"] in {"completed", "failed"} and
        (now - task["creation_time"]).total_seconds() > 1200
    ]
    for task_id in completed_tasks:
        del trancription_tasks[task_id]
        print(f"Task {task_id} removed from memory.")

async def cleanup_task(task_id: str) -> None:
    await asyncio.sleep(20 * 60)
    trancription_tasks.pop(task_id, None)

@app.on_event("startup")
async def startup_event() -> None:
    os.makedirs(settings.tmp_dir, exist_ok=True)
    load_whisperx_models(device="cpu")
    Thread(target=transcription_worker, daemon=True).start()
    scheduler.add_job(cleanup_tmp_dir, "interval", minutes=10)
    scheduler.add_job(cleanup_completed_tasks, "interval", minutes=20)
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event() -> None:
    scheduler.shutdown()

@app.post("/transcribe/")
async def create_upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    task_id = str(uuid.uuid4())
    if not os.path.exists(settings.tmp_dir):
        os.makedirs(settings.tmp_dir, exist_ok=True)

    try:
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=settings.tmp_dir,
            suffix=file_ext
        ) as tmp_file:
            tmp_path = tmp_file.name

            trancription_tasks[task_id] = {
                "status": "loading",
                "creation_time": datetime.utcnow(),
                "result": None
            }

            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )

    trancription_tasks[task_id].update({"status": "processing"})
    trancription_tasks_queue.put((task_id, tmp_path))

    background_tasks.add_task(cleanup_task, task_id)

    return {
        "task_id": task_id,
        "creation_time": trancription_tasks[task_id]["creation_time"].isoformat(),
        "status": trancription_tasks[task_id]["status"]
    }

@app.get("/transcribe/status/{task_id}")
async def get_task_status(task_id: str) -> dict:
    task = trancription_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "creation_time": task["creation_time"],
        "status": task["status"]
    }

@app.get("/transcribe/result/{task_id}")
async def get_task_result(task_id: str) -> dict:
    task = trancription_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] in {"loading", "processing"}:
        raise HTTPException(status_code=404, detail="Task not completed")

    return {
        "task_id": task_id,
        "creation_time": task["creation_time"],
        "status": task["status"],
        "result": task["result"]
    }
