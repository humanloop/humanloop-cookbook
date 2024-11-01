"""This script demonstrates using the SDK decorators to instrument a simple conversational agent with function calling."""

import json
import os
import random
import pprint

from dotenv import load_dotenv
from humanloop import Humanloop
from openai import OpenAI
import humanloop
from main import hl_client

if __name__ == "__main__":
    print(humanloop.__version__)
    print(hl_client)
    # print(hl_client.api_key)
    print(hl_client.prompts.list())
    hl_client.prompts.upsert(
        model="gpt-4o",
        path="temp/Decorators Tutorial/Agent Prompt",
        template="hi {{name}}",   
    )
