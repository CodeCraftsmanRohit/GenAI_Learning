from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os

import json


import requests
from pydantic import BaseModel,Field
from typing import Optional

client=OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") , # match env variable name

    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)



SYSTEM_PROMPT="""

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

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT },
]


def get_weather(city :str):

    url=f"https://wttr.in/{city.lower()}?format=%C+%t"

    response=requests.get(url)



    if response.status_code==200:

        return f"The weather in {city} is {response.text}"

    return "Something went wrong"

def run_command(cmd :str):
    result=os.system(cmd)
    return result


available_tools={

    "get_weather":get_weather,
    "run_command":run_command

}


def main():

    user_query=input("👍 ")

    message_history.append({"role":"user","content":user_query})

    while True:
        response=client.chat.completions.parse(

            model="gemini-2.5-flash",

            response_format=MyOutputFormat,

            messages=message_history

        )

        raw_result=response.choices[0].message.content


        message_history.append({"role":"assistant","content":raw_result})

        parsed_result=response.choices[0].message.parsed



        if parsed_result.step=="START":

            print("🔥",parsed_result.content)

            continue



        if parsed_result.step=="TOOL":

            tool_yo_call=parsed_result.tool

            tool_input=parsed_result.input

            print(f"🔨{tool_yo_call}({tool_input})")



            tool_response=available_tools[tool_yo_call](tool_input)

            print(f"🔨{tool_yo_call}({tool_input})={tool_response}")

            message_history.append({"role":"developer","content":json.dumps(

                {"step":"OBSERVE","tool":tool_yo_call,"input":tool_input,"output":tool_response}

            )})

            continue







        if parsed_result.step == "PLAN":

            print("🧠", parsed_result.content)

            continue



        if parsed_result.step == "OUTPUT":

            print("✨", parsed_result.content)

            break

if __name__ == "__main__":
    main()