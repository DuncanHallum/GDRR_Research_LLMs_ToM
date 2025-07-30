import os
from dotenv import load_dotenv
import numpy as np
import json
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.append(str(BASE_PATH))

from model.main import generate_init_beliefs, recognise_character, update_belief, EMOTIONS, STATES

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Generate belief distributions
def gen_dists(observations: list):
    all_dists = [] # 2D list, each sublist containing 3 distributions as lists
    for observation in observations:
        print("processing")
        character = recognise_character(observation)
        init_beliefs = generate_init_beliefs(STATES)
        system_belief_of_user = json.loads(update_belief(observation,
                                    f"the probability that the user is that state.",
                                    init_beliefs[0]
                                    ))
        system_belief_of_character = json.loads(update_belief(observation,
                                    f"the probability that {character} is in that state.",
                                    init_beliefs[1]
                                    ))
        system_belief_of_user_belief_of_character = json.loads(update_belief(observation,
                                    f"the probability of the user believing that {character} is in that state. The current belief distribution is.",
                                    init_beliefs[1]
                                    ))
          
        all_dists.append([system_belief_of_user.values(), system_belief_of_character.values(), system_belief_of_user_belief_of_character.values()])
    return all_dists

def save_dists_to_file(path, dists: list):
    for i in range(len(dists)):
        df = pd.DataFrame({
            "Emotion": EMOTIONS,
            "Prob_System_belief_User": dists[i][0],
            "Prob_System_belief_Character": dists[i][1],
            "Prob_System_belief_User_belief_Character": dists[i][2]
        })
        df.to_csv(path/("message_"+str(i+1)+"_belief_dist.csv"), index=False)
    
if __name__ == "__main__":
    MESSAGES_PATH = BASE_PATH/"testing"/"files"/"messages"
    DISTRIBUTIONS_PATH = BASE_PATH/"testing"/"files"/"distributions"
    for file in MESSAGES_PATH.iterdir():
        with open(file, encoding="utf-8") as f:
            messages = f.read().split("\n")
        dists = gen_dists(messages)
        save_dists_to_file(DISTRIBUTIONS_PATH/file.stem, dists) #DISTRIBUTIONS_PATH/file (without .txt)
    
    
