import os
import glob
import json
from pathlib import Path
import ijson

def parse_clean_battle_full(file_path):
    clean_battle_prompts = []
    
    with open(file_path, 'rb') as file:
        # Create a parser
        parser = ijson.parse(file)

        found_prompt = False
        for prefix, event, value in parser:
            if prefix == "item.conversation_a.item.role" and value == "user":
                found_prompt = True
            elif prefix == "item.conversation_a.item.content" and found_prompt:
                clean_battle_prompts.append(value)
                found_prompt = False
            
    # return clean_battle_prompts
    return clean_battle_prompts             
    


# # Usage
# file_path = 'your_json_file.json'
# clean_battle_prompts = parse_clean_battle(file_path)

# # Print the result
# print(clean_battle_prompts)


def parse_clean_battle_simple(file_path):
    clean_battle_prompts = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        battles = json.load(file)
        for battle in battles:
                conv = battle["conversation_a"]
                for resp in conv:
                     if resp["role"] == "user":
                          clean_battle_prompts.append(resp["content"])
    
    return clean_battle_prompts

def parse_mmlu_full(file_path):
    mmlu_prompts = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line by comma and get the first element
            prompt = line.split(',')[0]
            
            # Remove any surrounding quotes if present
            prompt = prompt.strip('"')
            
            mmlu_prompts.append(prompt)
    
    return mmlu_prompts

def process_all_mmlu_data_files(file_path):
    mmlu_data_folder = Path(file_path)
    mmlu_prompts = []
    for filename in os.listdir(mmlu_data_folder):
        file_path = mmlu_data_folder / filename
        if file_path.is_file():
            mmlu_prompts.extend(parse_mmlu_full(file_path))
    return mmlu_prompts

# Usage
clean_battle_file_path = 'embedding-similarity/data/clean_battle_data/mini_conv_battle.json'
mmlu_data_file_path = "embedding-similarity/data/mmlu_data/val"
simple_clean_battle_prompts = parse_clean_battle_simple(clean_battle_file_path)
full_clean_battle_prompts = parse_clean_battle_full(clean_battle_file_path)

# Print the result
# print(process_all_mmlu_data_files())
