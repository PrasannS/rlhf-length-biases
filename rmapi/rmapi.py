from flask import Flask, request, jsonify
from transformers import pipeline


# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import set_seed

from rlhfutils.rl_utils import (
    ScriptArguments,
    load_models,
)

import threading

lock = threading.Lock()

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

# NOTE handle loading everything in, since hyperparams are same for every setting more or less
config, tokenizer, reward_model = load_models(script_args, "rm")

# TODO assume that RM strings are passed in with the right format

app = Flask(__name__)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 4, "truncation": True}

@app.route('/predict', methods=['POST'])
def predict():
    # for thread safety
    with lock:
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])

        # Check if input_texts is a list
        if not isinstance(input_texts, list):
            return jsonify({"error": "Input must be a list of strings."}), 400

        results = reward_model(input_texts, **sent_kwargs)
        scores = [output[0]["score"] for output in results]
        
        return jsonify(scores)
    
if __name__ == '__main__':
    app.run(debug=False)
