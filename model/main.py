import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


#openai.apikey = os.environ.get("OPENAI_API_KEY")

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

# Generates uniform distribution over states
def generate_init_belief(states): 
    belief = [{},{}]
    prob = 1/len(states[0]) # uniform 
    for state in states[0]:
        belief[0][state] = prob
        belief[0][state] = prob
    return belief

def recognise_character(text):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Identify the name of the character that the user is talking about, give only the name of the character"},
            {"role": "user", "content": text}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

def update_belief(belief, observation, character, prev_action=None):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""
                                        You are a cognitive model, reasoning about peoples mental states to give 
                                        advice for navigating a workplace environment following a POMDP structure.
                                        
                                        The current belief distribution is {belief}, where the first dictionary represents
                                        the probability distribution of the user's mental states, and the second distribution 
                                        represents that of the user's belief about {character}'s mental state.
                                        Your previous action or message to the user was {prev_action} if available.
                                        
                                        Given the user's input as an observation, the current belief, and the previous action, generate a new 
                                        belief distribution as an array of 2 dictionaries as before.
             """},
            {"role": "user", "content": observation}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

def generate_action(belief, observation, character):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""
                                        You are a cognitive model, used to give advice on workplace problems and relations with a POMDP structure.
                                        The current belief distribution is {belief}, where the first dictionary represents
                                        the probability distribution of the user's mental states, and the second distribution
                                        represents that of the user's belief about {character}'s mental state.

                                        Given this information, generate an action from the follwing user message, in the form of giving helpful advice, asking a question to
                                        better your understand without being too invasive, or give a sympathetic response.
             """},
            {"role": "user", "content": observation}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    action = None
    user_input = ""
    while user_input != "end":
        observation = input("User: ")
        belief = generate_init_belief(STATES)
        print(belief)
        character = recognise_character(observation)
        print(character)
        belief = update_belief(belief, observation, character, action)
        print(belief)
        action = generate_action(observation, character)
        print(action)

