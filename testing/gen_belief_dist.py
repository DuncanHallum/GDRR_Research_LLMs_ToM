import os
import openai
from dotenv import load_dotenv
import numpy as np
import json
from pathlib import Path
from model.main import generate_init_beliefs, recognise_character, update_belief, EMOTIONS, STATES
import pandas as pd

BASE_PATH = Path(__file__).parent.parent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#Generate belief distribution, where the other entities belief distribution is system's belief of what user think other entity/character's mental state is
def gen_dist_user_belief_about_character(observations: list, user):
    all_beliefs = [] # 2D list, each sublist containing 2 distributions (each the 2 beliefs) as lists
    for observation in observations:
        character = recognise_character(observation)
        init_beliefs = generate_init_beliefs()
        user_belief = update_belief(observation,
                                    f"the probability that the user is that state. The current belief distribution is {init_beliefs[0]}.",
                                    )
        character_belief = update_belief(observation,
                                    f"the probability of the user thinking that {character} is in that state. The current belief distribution is {init_beliefs[1]}."
                                    )
        all_beliefs.append([user_belief.values(), character_belief.values()])

#Generate belief distribution, where the other entities belief distribution is system's belief of other entitiy/character's mental state
def gen_dist_system_belief_about_character(observations: list):
    all_beliefs = [] # 2D list, each sublist containing 2 distributions (each the 2 beliefs) as lists for one message/observation
    for observation in observations:
        character = recognise_character(observation)
        init_beliefs = generate_init_beliefs()
        user_belief = update_belief(observation,
                                    f"the probability that the user is that state. The current belief distribution is {init_beliefs[0]}.",
                                    )
        character_belief = update_belief(observation,
                                    f"the probability that {character} is in that state. The current belief distribution is {init_beliefs[1]}."
                                    )
        all_beliefs.append([user_belief.values(), character_belief.values()])

def save_dists_to_file(path, dists: list, messages):
    for i in range(dists):
        df = pd.DataFrame({
            "Emotion": EMOTIONS,
            "Prob_User": dists[i][0],
            "Prob_Character": dists[i][1]
        })
        df.to_csv(path+"\message_"+(i+1)+"_belief_dist.csv", index=False)
    
if __name__ == "__main__":
    messages_emotions_explicit = []
    gen_dist_user_belief_about_character(messages_emotions_explicit)
    messages_emotions_implicit = []