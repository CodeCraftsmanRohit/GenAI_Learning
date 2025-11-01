import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import asyncio
import tempfile
import uuid
import subprocess
import sys

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # match env variable name
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
async_client=AsyncOpenAI()

def speak_text(text: str, engine):
    """Speak the given text with pyttsx3 engine (blocking)."""
    engine.say(text)
    engine.runAndWait()


async def tts(speech : str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gemini-2.5-flash-preview-tts",
        voice="coral",
        instructions="Always speak in cheerfull manner with full of delight and happy ",
        input=speech,
        response_format="pcm"
    )as response:
        await LocalAudioPlayer().play(response)

def speak_gtts(text: str, lang: str = "hi"):
    """
    Create an MP3 via gTTS and play it.
    Tries playsound first; otherwise uses OS default player.
    Deletes the temp file after playing.
    """
    # create unique temp filename
    tmp_dir = tempfile.gettempdir()
    filename = os.path.join(tmp_dir, f"gtts_{uuid.uuid4().hex}.mp3")
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
    except Exception as e:
        # Could be no internet or gTTS error
        raise RuntimeError(f"gTTS save failed: {e}")

    # Try playsound if available
    try:
        from playsound import playsound
        playsound(filename)
    except Exception:
        # fallback to OS-level player
        try:
            if sys.platform.startswith("win"):
                # start is shell built-in; use subprocess with shell=True
                subprocess.run(["start", "", filename], shell=True, check=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", filename], check=True)
            else:  # assume linux/unix
                subprocess.run(["xdg-open", filename], check=True)
            # Note: OS player may be asynchronous and return immediately.
            # We'll wait a bit to allow playback; approximate by sleeping length.
            # If you want reliable sync playback, install `playsound`.
            time.sleep(max(1, len(text) * 0.05 + 0.5))
        except Exception as oe:
            # Could not open file; raise to let caller fallback to pyttsx3
            raise RuntimeError(f"Playback failed: {oe}")
    finally:
        # try to remove file (ignore errors)
        try:
            os.remove(filename)
        except Exception:
            pass


def main():

    engine = pyttsx3.init()
    # Optional: tweak voice rate / volume
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.95))  # slightly slower
    volume = engine.getProperty("volume")
    engine.setProperty("volume", volume)  # 0.0 to 1.0

    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2
        SYSTEM_PROMPT = f"""
            You're an expert voice agent. You are given the transcript of what
            user has said using voice.

            You need to output as if you are an voice agent and whatever you speak
            will be converted back to audio using AI and played back to user.
            """

        messages=[{"role": "system", "content": SYSTEM_PROMPT}]

        while True:

            print("🎙️ Speak Something...")
            audio = r.listen(source)

            print("🧠 Processing Audio... (Google STT)")
            try:
                stt = r.recognize_google(audio)
                print("🗣️ You said:", stt)
                messages.append({"role": "user", "content": stt})

            except sr.UnknownValueError:
                print("❌ Could not understand the audio.")
            except sr.RequestError as e:
                print(f"⚠️ Google Speech Recognition error: {e}")



            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                reasoning_effort="high",
                messages=messages


            )
            print("🔥AI Response:",response.choices[0].message.content)
            # asyncio.run(tts(speech=response.choices[0].message.content))

            try:
                speak_gtts(response.choices[0].message.content, lang="hi")
            except Exception as e:
             print("⚠️ gTTS playback failed, falling back to pyttsx3:", e)
             speak_text(response.choices[0].message.content, engine)


            time.sleep(0.2)

if __name__ == "__main__":
    main()
