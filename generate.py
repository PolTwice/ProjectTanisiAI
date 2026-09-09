import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))

# PURPOSE: 
def generate_story(topic: str, vocab_words: list[str]) -> str:
    # Open rules file.
    with open("./story_rules.txt", "r") as f:
        system_rules = f.read()

    # Get topic and vocab from arguments
    user_request = f"Topic: {topic}\nRequired Vocabulary: {', '.join(vocab_words)}"

    # Send and wait for response to come in. 
    # chat: Tells the client: "We are targeting chat-tuned models that accept an array of role-based messages (messages=[...]), rather than a single raw string.
    # .completions: Tells the client: "We want the model to generate the next response (to 'complete' the conversation thread)."
    # create sends the request over the wire and give back the result
    response = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": user_request}
        ]
    )

    # response: Metadata about the request
    # choices: Get the first generated response. ATM theres only one response
    # [0] get the first element  of choices which is a list that only has 1 item
    # message: represents the generated message. also has the role and the content of the message
    # content: Gives the raw content of the response. 
    # json.loads parses it into a python dictionary. Probably want it raw. 
    return parse_story(response.choices[0].message.content)

def parse_story(raw_output: str) -> dict:
    #try take input and turn it into a dictionary
    try:
        story_data = json.loads(raw_output)

    # if theres a problem with turning the json into a dictionary
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print("Raw output received from model:")
        print(raw_output)
        return False
    
    return story_data

def reconstruct_json_string(ai_dict) -> str:
    json_string = json.dumps(ai_dict, indent=2, ensure_ascii=False)

def reconstruct_json_file(ai_dict) -> bool:
    with open("story_manifest.json", "w", encoding="utf-8") as f:
        #ai_dict is the dictionary input. 
        #f is the file where things are going
        #indent is how 
        json.dump(ai_dict, f, indent=2, ensure_ascii=False)

#################### Testing ########################

topic = "A bear goes camping"
vocab = ["fire", "log", "tent"]

with open("./exampleJson.txt", "r") as g:
    testJson = g.read()

#aioutput = generate_story(topic, vocab)

print(parse_story(testJson))