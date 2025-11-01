from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from pydantic import BaseModel, Field
from typing import Optional

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import time
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import asyncio
import tempfile
import uuid
import subprocess
import sys

# Initialize clients
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

async_client = AsyncOpenAI()

# System prompt for chain-of-thought reasoning
SYSTEM_PROMPT = """
You're an expert AI Assistant in resolving user queries using chain of thought.

You work on **START**, **PLAN**, and **OUPUT** steps.

You need to first **PLAN** what needs to be done. The **PLAN** can be multiple steps.

Once you think enough **PLAN** has been done, finally you can give an **OUTPUT**.

You can also call a **tool** if required from the list of available tools.

for every tool call wait for the observe step which is the output from the called tool.

Rules:
- Strictly Follow the given **JSON output format**
- Only run one step at a time.
- The sequence of steps is **START** (where user gives an input), **PLAN** (That can be multiple times) :

Output **JSON Format**:
{ "step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string" ,"tool":"string","input":"string"}

Available Tools:
-get_weather(city:str):Takes city name as an input string and returns the weather info about the city.
-run_command(cmd : str): Takes a system linux command as string and executes the command on user's system and returns the output from that command

Example 1:
 START: What is the weather of Delhi?
 PLAN: {"step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi"}
 PLAN: {"step": "PLAN", "content": "Lets see if we have any available tool from the list of available tools"}
 PLAN: {"step": "PLAN", "content": "Great, we have get_weather tool available for this query."}
 PLAN:{"step": "PLAN", "content": "I need to call get_weather tool for delhi as input for city"}
 PLAN: {"step": "TOOL", "tool": "get_weather", "input": "delhi"}
 PLAN {"step": "OBSERVE", "tool": "get_weather", "output": "The temp of delhi is cloudy with 20 deg. cesius"}
 PLAN: {"step": "PLAN", "content": "Great, I got the weather info about delhi"}
 OUTPUT: {"step": "OUTPUT", "content": "The cuurent weather in delhi is 20 C with some cloudy sky"}
"""

class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="The optional string content for the step")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")

# Tool functions
def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"

def run_command(cmd: str):
    try:
        result = os.popen(cmd).read()
        return result
    except Exception as e:
        return f"Error running command: {str(e)}"

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

# TTS functions
def speak_text(text: str, engine):
    """Speak the given text with pyttsx3 engine (blocking)."""
    engine.say(text)
    engine.runAndWait()

async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gemini-2.5-flash-preview-tts",
        voice="coral",
        instructions="Always speak in cheerfull manner with full of delight and happy ",
        input=speech,
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

def speak_gtts(text: str, lang: str = "en"):
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

def process_user_query(user_query, message_history):
    """Process user query using chain-of-thought reasoning"""
    message_history.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.parse(
            model="gemini-2.5-flash",
            response_format=MyOutputFormat,
            messages=message_history
        )

        raw_result = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": raw_result})
        parsed_result = response.choices[0].message.parsed

        if parsed_result.step == "START":
            print("🔥", parsed_result.content)
            continue

        if parsed_result.step == "TOOL":
            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input
            print(f"🔨{tool_to_call}({tool_input})")

            tool_response = available_tools[tool_to_call](tool_input)
            print(f"🔨{tool_to_call}({tool_input})={tool_response}")

            message_history.append({"role": "developer", "content": json.dumps(
                {"step": "OBSERVE", "tool": tool_to_call, "input": tool_input, "output": tool_response}
            )})
            continue

        if parsed_result.step == "PLAN":
            print("🧠", parsed_result.content)
            continue

        if parsed_result.step == "OUTPUT":
            print("✨", parsed_result.content)
            return parsed_result.content

def main():
    # Initialize voice components
    engine = pyttsx3.init()
    # Optional: tweak voice rate / volume
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.95))  # slightly slower
    volume = engine.getProperty("volume")
    engine.setProperty("volume", volume)  # 0.0 to 1.0

    r = sr.Recognizer()

    # Initialize message history for chain-of-thought
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        print("🎙️ Voice Assistant with Chain-of-Thought Reasoning is ready!")
        print("Speak your query...")

        while True:
            try:
                print("\n🎙️ Listening...")
                audio = r.listen(source, timeout=10)

                print("🧠 Processing Audio... (Google STT)")
                stt = r.recognize_google(audio)
                print("🗣️ You said:", stt)

                # Process the query using chain-of-thought reasoning
                print("🤔 Processing with chain-of-thought reasoning...")
                final_output = process_user_query(stt, message_history)

                # Speak the final output
                print("🎯 Final Answer:", final_output)

                try:
                    speak_gtts(final_output, lang="en")
                except Exception as e:
                    print("⚠️ gTTS playback failed, falling back to pyttsx3:", e)
                    speak_text(final_output, engine)

            except sr.WaitTimeoutError:
                print("⏰ No speech detected, listening again...")
                continue
            except sr.UnknownValueError:
                print("❌ Could not understand the audio.")
                error_msg = "Sorry, I couldn't understand what you said. Please try again."
                speak_text(error_msg, engine)
            except sr.RequestError as e:
                print(f"⚠️ Google Speech Recognition error: {e}")
                error_msg = "There was an error with the speech recognition service."
                speak_text(error_msg, engine)
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                error_msg = "Sorry, something went wrong. Please try again."
                speak_text(error_msg, engine)

            time.sleep(0.5)

if __name__ == "__main__":
    main()