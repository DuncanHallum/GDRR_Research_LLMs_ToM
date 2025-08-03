import os
from dotenv import load_dotenv
import numpy as np
import json
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path
import openai

BASE_PATH = Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

from model.main import generate_init_beliefs, recognise_character, update_belief, EMOTIONS, STATES

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def gen_save_message(emotion, file_path):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""
                                        Generate 10 messages from someone asking for advice regarding a workplace 
                                        problem involving one other person. The person writing the message should be 
                                        feeling {emotion} for each.
                                        Output only the messages to be saved to a .txt file, each seperated with a new line, with no markdown formatting.
            """}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    message =  response.choices[0].message.content
    with open(file_path/Path("user_"+emotion+".txt"), "w") as fp:
        fp.write(message)

if __name__ == "__main__":
    MESSAGES_PATH = BASE_PATH/"testing"/"files"/"messages"
    for emotion in EMOTIONS:
        gen_save_message(emotion, MESSAGES_PATH)
    
