import os
import openai
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

#Generate belief distribution, where the other entities belief distribution is system's belief of what user think other entity/character's mental state is
def gen_dist_user_belief_about_character(observations: list):
    all_beliefs = [] # 2D list, each sublist containing 2 distributions (each the 2 beliefs) as lists
    for observation in observations:
        print("processing")
        character = recognise_character(observation)
        init_beliefs = generate_init_beliefs(STATES)
        user_belief = json.loads(update_belief(observation,
                                    f"the probability that the user is that state. The current belief distribution is {init_beliefs[0]}.",
                                    ))
        character_belief = json.loads(update_belief(observation,
                                    f"the probability of the user thinking that {character} is in that state. The current belief distribution is {init_beliefs[1]}."
                                    ))
        all_beliefs.append([user_belief.values(), character_belief.values()])

#Generate belief distribution, where the other entities belief distribution is system's belief of other entitiy/character's mental state
def gen_dist_system_belief_about_character(observations: list):
    all_dists = [] # 2D list, each sublist containing 2 distributions (each the 2 beliefs) as lists for one message/observation
    for observation in observations:
        print("processing")
        character = recognise_character(observation)
        init_beliefs = generate_init_beliefs(STATES)
        user_belief = json.loads(update_belief(observation,
                                    f"the probability that the user is that state. The current belief distribution is {init_beliefs[0]}.",
                                    ))
        character_belief = json.loads(update_belief(observation,
                                    f"the probability that {character} is in that state. The current belief distribution is {init_beliefs[1]}."
                                    ))
        all_dists.append([user_belief.values(), character_belief.values()])
    return all_dists

def save_dists_to_file(path, dists: list):
    for i in range(len(dists)):
        df = pd.DataFrame({
            "Emotion": EMOTIONS,
            "Prob_User": dists[i][0],
            "Prob_Character": dists[i][1]
        })
        df.to_csv(path/("message_"+str(i+1)+"_belief_dist.csv"), index=False)
    
if __name__ == "__main__":
    with open(BASE_PATH/r"testing/files/user_inputs/inputs_explicit_emotions.txt", encoding="utf-8") as f_explicit:
        messages_emotions_explicit = f_explicit.read().split("\n")
    with open(BASE_PATH/r"testing/files/user_inputs/inputs_implicit_emotions.txt", encoding="utf-8") as f_implicit:
        messages_emotions_implicit = f_implicit.read().split("\n")
    
    print("CASE 1")
    # Case 1: Distributions where other character's dist is system's belief of their mental state
    #dists = gen_dist_system_belief_about_character(messages_emotions_explicit)
    #save_dists_to_file(BASE_PATH/r"testing/files/case_1/explicit_emotions", dists)
    #dists = gen_dist_system_belief_about_character(messages_emotions_implicit)
    #save_dists_to_file(BASE_PATH/r"testing/files/case_1/implicit_emotions", dists)
    
    print("CASE 2")
    # Case 2: Distributions where other character's dist is system's belief of user's belief of their mental state
    dists = gen_dist_user_belief_about_character(messages_emotions_explicit)
    save_dists_to_file(BASE_PATH/r"testing/files/case_2/explicit_emotions", dists)
    print("CASE 2: pt 2")
    dists = gen_dist_user_belief_about_character(messages_emotions_implicit)
    save_dists_to_file(BASE_PATH/r"testing/files/case_2/implicit_emotions", dists)
