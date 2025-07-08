# AI Voice Transcription Bot

A Telegram bot that converts voice messages and audio files to text using OpenAI's Whisper model locally.

## What it does
- Transcribes voice messages and audio files instantly
- Supports multiple audio formats (OGG, MP3, M4A, WAV)
- Handles files up to 50MB and 4 minutes duration
- Automatically splits longer audio into chunks
- Works completely offline (no API costs)

## How to use
Send the bot a voice message or audio file. It will return the transcribed text.

## Setup for developers
You need FFmpeg installed and a Telegram bot token.

1. Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)
2. `pip install -r requirements.txt`
3. Replace placeholder token in config.py with real token
4. `python3 transcriber_bot.py`

**Getting bot token:**
- Telegram: Message @BotFather to create bot token

**System requirements:**
- Python 3.8+
- FFmpeg and ffprobe installed
- 2GB+ RAM (for Whisper model)
- CUDA GPU optional (faster processing)

**Cost:** Completely free - runs locally with no API calls.

## Features
- Local Whisper AI processing (small model)
- Configurable language detection (auto/ru/en in config.py)
- File size limits (50MB max)
- Audio chunking for long files
- User statistics tracking
- Comprehensive logging
- Graceful error handling
