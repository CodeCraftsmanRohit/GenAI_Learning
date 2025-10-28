from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import time

# Optional: voice libs
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are an AI Persona Assistant named Rohit Kumar.
You represent Rohit Kumar — a 25-year-old tech enthusiast and Principal Engineer.
Your main tech stack is JavaScript and Python, and you’re currently learning and experimenting with Generative AI.
You reply with confidence, friendliness, and technical clarity — like a smart, approachable engineer who loves to share knowledge.
Keep responses concise but insightful, and sound naturally conversational.
"""

# Initialize message history with system prompt
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Initialize TTS engine (pyttsx3) if available
def init_tts():
    if pyttsx3 is None:
        return None
    engine = pyttsx3.init()
    # Optional: tweak voice properties
    try:
        engine.setProperty("rate", 170)  # speaking rate
        engine.setProperty("volume", 1.0)
    except Exception:
        pass
    return engine

def speak_text(engine, text):
    if engine is None:
        print("🔈 (TTS not available) ->", text)
        return
    engine.say(text)
    engine.runAndWait()

# Initialize speech recognizer if available
def listen_once(timeout=6, phrase_time_limit=15):
    """
    Returns recognized text or None.
    Uses SpeechRecognition with the default Google Web Speech API.
    """
    if sr is None:
        print("🎙️ SpeechRecognition not installed. Falling back to text input.")
        return None

    r = sr.Recognizer()
    mic = None
    try:
        mic = sr.Microphone()
    except Exception as e:
        print("No microphone found or cannot access it:", e)
        return None

    with mic as source:
        print("🎧 Listening... (speak now)")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout).")
            return None

    try:
        # Using Google Web Speech API (online). Replace with r.recognize_vosk for offline VOSK if set up.
        text = r.recognize_google(audio)
        print("🗣️ You (recognized):", text)
        return text
    except sr.UnknownValueError:
        print("❓ Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("⚠️ Speech recognition service error:", e)
        return None

def get_user_input(mode, tts_engine):
    """
    mode: 'text' or 'voice'
    Returns user_input string or None (on exit request)
    """
    if mode == "text":
        try:
            return input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            return "bye"
    else:
        # voice mode
        spoken = listen_once()
        if spoken is None:
            # fallback to text prompt after failed attempt
            print("Type your message instead (or press Enter to retry voice):")
            try:
                txt = input("🧑 You (text fallback): ").strip()
                return txt if txt else None
            except (KeyboardInterrupt, EOFError):
                return "bye"
        return spoken

def main():
    print("🤖 Rohit Kumar Assistant — type 'bye' or 'exit' to end the chat")
    print("Choose input mode: [1] Text (default)  [2] Voice")
    mode_choice = input("Enter 1 or 2: ").strip()
    mode = "voice" if mode_choice == "2" else "text"

    tts_engine = init_tts()
    if mode == "voice" and sr is None:
        print("⚠️ speech_recognition not installed — falling back to text mode.")
        mode = "text"
    if tts_engine is None:
        print("⚠️ pyttsx3 (TTS) not available — replies will be printed but not spoken.")

    # Optional greeting
    greeting = "Hey! I'm Rohit Kumar. How can I help you today?"
    print("🤖 Rohit:", greeting)
    if tts_engine:
        speak_text(tts_engine, greeting)

    while True:
        user_input = get_user_input(mode, tts_engine)
        # If voice fallback returns None (failed recognition + empty fallback), retry loop
        if user_input is None:
            continue

        if user_input.lower().strip() in ["bye", "exit", "quit"]:
            farewell = "Catch you later! Keep building cool stuff. 🚀"
            print("🤖 Rohit:", farewell)
            if tts_engine:
                speak_text(tts_engine, farewell)
            break

        # append to conversation history
        message_history.append({"role": "user", "content": user_input})

        try:
            # call Gemini model
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=message_history
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"Sorry, I hit an error calling the model: {e}"
            print("⚠️", reply)

        # print and speak the reply
        print(f"🤖 Rohit: {reply}")
        if tts_engine:
            # speak in a non-blocking way? pyttsx3.runAndWait() blocks but fine for CLI
            speak_text(tts_engine, reply)

        # append assistant reply to history
        message_history.append({"role": "assistant", "content": reply})

        # small pause to avoid microphone bleed into TTS playback on some systems
        if mode == "voice":
            time.sleep(0.2)

if __name__ == "__main__":
    main()
