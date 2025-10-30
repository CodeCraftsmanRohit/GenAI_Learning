from openai import OpenAI

from dotenv import load_dotenv

import os

import json

load_dotenv()

import requests



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





 Example 2: Weather Query

 START: What is the weather of Delhi?

 PLANNING PLAN: $\{$"step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi in"$\}$

 PLAN: $\{$"step": "PLAN", "content": "Lets see if we have any available tool from the list of avai"$\}$PLAN: $\{$"step": "PLAN", "content": "Great, we have get_weather tool available for this query."$\}$PLAN: $\{$"step": "PLAN", "content": "I need to call get_weather tool for delhi as input for city"$\}$TOOL: $\{$"step": "TOOL", "tool": "get_weather", "input": "delhi"$\}$PLAN (OBSERVE): $\{$"step": "OBSERVE", "tool": "get_weather", "output": "The temp of delhi is cloudy with 20"$\}$PLAN: $\{$"step": "PLAN", "content": "Great, I got the weather info about delhi"$\}$FINAL OUTPUTOUTPUT: $\{$"step": "OUTPUT", "content": "The cuurent weather in delhi is 20 C with some cloudy sky"$\}$



Do NOT return any additional commentary or metadata outside the array.





"""







def get_weather(city :str):

    url=f"https://wttr.in/{city.lower()}?format=%C+%t"

    response=requests.get(url)



    if response.status_code==200:

        return f"The weather in {city} is {response.text}"

    return "Something went wrong"



available_tools={

    "get_weather":get_weather

}





def main():

    user_query=input("> ")

    response= client.chat.completions.create(

         model="gemini-2.5-flash",

         messages=[

             {"role":"user","content":user_query}

         ]

    )

    print(f"{response.choices[0].message.content}")







message_history=[

    {"role":"system","content": SYSTEM_PROMPT}

]



user_query=input("👍 ")

message_history.append({"role":"user","content":user_query})



while True:

    response=client.chat.completions.create(

        model="gemini-2.5-flash",

        response_format={"type":"json_object"},

        messages=message_history

    )

    raw_result=response.choices[0].message.content

    message_history.append({"role":"assistant","content":raw_result})

    parsed_result=json.loads(raw_result)



    if parsed_result.get("step")=="START":

        print("🔥",parsed_result.get("content"))

        continue



    if parsed_result.get("step")=="TOOL":

        tool_yo_call=parsed_result.get("tool")

        tool_input=parsed_result.get("input")

        print(f"🔨{tool_yo_call}({tool_input})")



        tool_response=available_tools[tool_yo_call](tool_input)

        print(f"🔨{tool_yo_call}({tool_input})={tool_response}")

        message_history.append({"role":"developer","content":json.dumps(

            {"step":"OBSERVE","tool":tool_yo_call,"input":tool_input,"output":tool_response}

        )})

        continue







    if parsed_result.get("step") == "PLAN":

        print("🧠", parsed_result.get("content"))

        continue



    if parsed_result.get("step") == "OUTPUT":

        print("✨", parsed_result.get("content"))

        continue