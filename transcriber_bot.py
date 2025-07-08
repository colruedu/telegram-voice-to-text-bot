# transcriber_bot.py

import os
import json
import tempfile
import subprocess
import signal
import logging
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import whisper
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

import config

# ──────────── Dependency Check ────────────
def check_dependencies():
    """Check if required external dependencies are available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        logger.info("FFmpeg and ffprobe dependencies verified")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg/ffprobe not found. Please install FFmpeg.")
        sys.exit(1)

# ──────────── Logging Configuration ────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Capture Python warnings into the logging system
logging.captureWarnings(True)

file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Filter out warnings (from the "py.warnings" logger) so they go only to the file
class NoWarningsFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("py.warnings")

console_handler.addFilter(NoWarningsFilter())
logger.addHandler(console_handler)

# ──────────── Whisper Model Loading ────────────
# Load a Whisper model once at startup. You can change "small" to "base", "medium", etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)
logger.info(f"Loaded Whisper model 'small' on {device}")

# ──────────── Rate-Limit Lock ────────────
transcribe_lock = asyncio.Lock()

# ──────────── Users JSON Helpers ────────────
USERS_PATH = Path(config.USERS_FILE)

def load_users() -> Dict[str, Any]:
    if not USERS_PATH.exists():
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load users.json: {e}")
        return {}

def save_users(data: Dict[str, Any]) -> None:
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save users.json: {e}")

def update_user_stats(user_id: int, user_data: dict, duration: float) -> None:
    users = load_users()
    user_id_str = str(user_id)
    
    if user_id_str not in users:
        users[user_id_str] = {
            "username": user_data.get("username", ""),
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
            "total_transcriptions": 0,
            "total_audio_minutes": 0.0,
        }
    
    # Ensure fields exist for existing users
    if "total_transcriptions" not in users[user_id_str]:
        users[user_id_str]["total_transcriptions"] = 0
    if "total_audio_minutes" not in users[user_id_str]:
        users[user_id_str]["total_audio_minutes"] = 0.0
    
    users[user_id_str]["total_transcriptions"] += 1
    users[user_id_str]["total_audio_minutes"] += duration / 60
    
    save_users(users)

# ──────────── FFmpeg Helpers ────────────
def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Convert any input audio (OGG/Opus, MP3, M4A, etc.) to 16 kHz mono WAV via ffmpeg.
    Returns True on success, False on failure.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return False

def get_audio_duration(path: str) -> float:
    """
    Return duration in seconds of an audio file using ffprobe.
    If ffprobe fails, return 0.0.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"ffprobe failed: {e}")
        return 0.0

def split_into_chunks(wav_path: str, chunk_dir: str, chunk_length: int) -> None:
    """
    Split wav_path into consecutive chunks of 'chunk_length' seconds.
    Outputs files named chunk_000.wav, chunk_001.wav, etc. in chunk_dir.
    """
    try:
        os.makedirs(chunk_dir, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-f",
                "segment",
                "-segment_time",
                str(chunk_length),
                "-c",
                "copy",
                os.path.join(chunk_dir, "chunk_%03d.wav"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg splitting failed: {e}")

# ──────────── /start Command Handler ────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        "Hello! Send any voice message (up to 4 minutes) and I'll return the transcription."
    )

    users = load_users()
    user_id_str = str(user.id)
    if user_id_str not in users:
        users[user_id_str] = {
            "username": user.username or "",
            "first_name": user.first_name or "",
            "last_name": user.last_name or "",
            "total_transcriptions": 0,
            "total_audio_minutes": 0.0,
        }
        save_users(users)
        logger.info(f"Added new user: {user_id_str} → {users[user_id_str]}")

# ──────────── Transcribe Helper ────────────
async def transcribe_chunk(wav_path: str) -> str:
    """
    Send one WAV chunk to the local Whisper model.
    Returns the transcript text (or empty string on failure).
    """
    try:
        # model.transcribe returns a dict with keys "text" and others
        if config.TRANSCRIPTION_LANGUAGE in ["auto", "none", None]:
            result = model.transcribe(wav_path)  # Auto-detect language
        else:
            result = model.transcribe(wav_path, language=config.TRANSCRIPTION_LANGUAGE)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription error for chunk {wav_path}: {e}")
        return ""

# ──────────── Voice/Audio Message Handler ────────────
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles both Telegram voice notes (OGG/Opus) and attached audio files.
    Downloads, converts to WAV, checks duration, splits into chunks if > MAX_AUDIO_DURATION,
    transcribes each chunk locally, and returns combined transcript.
    """
    async with transcribe_lock:
        message = update.message
        user = update.effective_user
        chat_id = message.chat.id

        # Indicate "typing" while processing
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        # Determine file_id and original extension
        if message.voice:
            file_id = message.voice.file_id
            orig_ext = ".ogg"
            file_size = message.voice.file_size
        elif message.audio:
            file_id = message.audio.file_id
            orig_ext = Path(message.audio.file_name or "").suffix or ".mp3"
            file_size = message.audio.file_size
        else:
            return  # Not a supported audio type

        # Check file size limit
        if file_size and file_size > config.MAX_FILE_SIZE:
            await message.reply_text(f"❗️File too large. Maximum {config.MAX_FILE_SIZE // 1_000_000}MB allowed.")
            return

        try:
            # 1) Download to a temporary directory
            tdir = tempfile.TemporaryDirectory()
            orig_path = os.path.join(tdir.name, f"input{orig_ext}")
            new_file = await context.bot.get_file(file_id)
            await new_file.download_to_drive(orig_path)
            logger.info(f"Downloaded audio to {orig_path} (user={user.id})")

            # 2) Convert any format → WAV
            wav_path = os.path.join(tdir.name, "converted.wav")
            if not convert_to_wav(orig_path, wav_path):
                await message.reply_text("❗️Audio conversion error. Please try again later.")
                return

            # 3) Check duration
            duration = get_audio_duration(wav_path)
            if duration <= 0:
                await message.reply_text("❗️Audio processing failed. Please try again.")
                return

            transcripts = []
            if duration <= config.MAX_AUDIO_DURATION:
                # Single chunk
                transcripts.append(await transcribe_chunk(wav_path))
            else:
                # Split into multiple chunks
                chunk_dir = os.path.join(tdir.name, "chunks")
                split_into_chunks(wav_path, chunk_dir, config.MAX_AUDIO_DURATION)
                for fname in sorted(os.listdir(chunk_dir)):
                    chunk_path = os.path.join(chunk_dir, fname)
                    if os.path.isfile(chunk_path):
                        transcripts.append(await transcribe_chunk(chunk_path))

            # Combine all transcripts
            full_text = "\n\n".join(filter(None, transcripts)).strip()
            if not full_text:
                await message.reply_text("❗️Could not extract text from audio.")
            else:
                await message.reply_text(full_text)

            # Update user stats
            user_data = {
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
            }
            update_user_stats(user.id, user_data, duration)
            logger.info(f"User {user.id} transcribed {duration:.1f}s audio")

        except Exception as e:
            logger.error(f"Exception in handle_voice: {e}")
            await message.reply_text("❗️Internal error occurred. Please try again later.")
        finally:
            if 'tdir' in locals():
                tdir.cleanup()

# ──────────── Graceful Shutdown ────────────
def shutdown_signal_handler(app):
    logger.info("Received termination signal. Shutting down…")
    asyncio.create_task(app.stop())

# ──────────── Main Entrypoint ────────────
def main() -> None:
    # Check dependencies first
    check_dependencies()
    
    application = (
        ApplicationBuilder()
        .token(config.TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    voice_filter = filters.VOICE | filters.AUDIO
    application.add_handler(MessageHandler(voice_filter, handle_voice))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: shutdown_signal_handler(application))

    logger.info("Bot started. Polling…")
    application.run_polling()

if __name__ == "__main__":
    main()
