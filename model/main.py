import os
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
print(BASE_PATH)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 28 emotions of Go Emotions data set
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# State is 2 lists: user's emotion and user's believed emotion of another character
STATES = [EMOTIONS, EMOTIONS]

# Generates 2 uniform distribution over states
def generate_init_beliefs(states): 
    belief = [{},{}]
    prob = 1/len(states[0]) # uniform 
    for state in states[0]:
        belief[0][state] = prob
        belief[0][state] = prob
    return belief

def recognise_character(text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a named entity recognotion model. Identify the name of the person that the user is talking about, give only the name of the character and nothing else. This could be 'Coworker' or 'Boss' for example. If unsure, return 'Coworker'."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

def update_belief(observation, context, prev_action=None):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""
                                        You are part of a cognitive model, reasoning about peoples mental states in a 
                                        workplace environment following a POMDP structure. Your job is to adjust the current belief distribution following a Bayesian approach.
                                        
                                        Given the user's input as an observation, the current belief, and the previous action, update the belief distribution to include this new context as a JSON object mapping each of these mental states {EMOTIONS} to {context}
                                        Your previous action or message to the user was {prev_action} if available.

                                        Output only the dictionary mapping each state to a probability.
             """},
            {"role": "user", "content": observation}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_action(belief, observation, character):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""
                                        You are a cognitive model, used to give advice on workplace problems and relations with a POMDP structure.
                                        Each belief distribution maps a state to a probability.
                                        The current belief distribution of the user's mental state is {belief[0]}.
                                        The current belief distribution of which mental state the user thinks that {character} is in is {belief[1]}.

                                        Given this information, generate an action or actions from the following user message, in the form of giving helpful advice, asking a question to
                                        better your understanding without being too invasive, or give a sympathetic or encouraging response.
             """},
            {"role": "user", "content": observation}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

#Plot bar graphs for beliefs side by side
def plot_beliefs(beliefs, character):
    data1 = list(beliefs[0].values()) # probabilities for User's state
    data2 = list(beliefs[1].values()) # probabilities for other character's state 
    x = np.arange(len(EMOTIONS))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, data1, width, label="User's state")
    bars2 = ax.bar(x + width/2, data2, width, label=f"User's belief about {character}'s state")

    ax.set_xlabel('Mental States')
    ax.set_ylabel('Probability')
    ax.set_title('Belief Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS)
    ax.legend()

    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.show()


if __name__ == "__main__":
    beliefs = generate_init_beliefs(STATES) # uniform distribution
    action = None
    character = "default"
    observation = ""
    while True:
        observation = input("User: ")
        if observation != "end":
            if character == "default": #only gets character name for the 1st observations
                character = recognise_character(observation)
                print(character)
            context_users_state = f"the probability that the user is in that state. The current belief distribution is {beliefs[0]}."
            belief_user_state = json.loads(update_belief(observation, context_users_state, action)) #belief distribution of user's own mental state

            context_character_state = f"the probability of the user thinking that {character} is in that state. The current belief distribution is {beliefs[1]}."
            belief_character_state = json.loads(update_belief(observation, context_character_state, action)) #belief distribution of the mental state which the user believes the other character is in
        
            beliefs = [belief_user_state, belief_character_state]

            plot_beliefs(beliefs, character)

            action = generate_action(beliefs, observation, character)
            print(action)
        else:
            break

