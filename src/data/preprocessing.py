import json
import pandas as pd
from pathlib import Path
import re

base_dir = Path(__file__).parent.parent.parent

metadata_path = base_dir / 'data' / 'metadata.json'
with metadata_path.open ('r', encoding='utf-8') as jf:
    metadata = json.load(jf)

def get_score(rating):
    return (rating - 3.5) / 2.5

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)  
    return text.strip()

def create_df(folder_path):

    folder_path = Path(folder_path)
    
    data = {'text': [], 'score': []}

    for file_path in folder_path.iterdir():
        file_id = file_path.stem
        if file_id in metadata:
            with file_path.open ('r', encoding='utf-8') as file:
                data['text'].append(preprocess_text(file.read()))
                data['score'].append(get_score(metadata[file_id]['rating']))
                
    return pd.DataFrame(data)
                     