from datasets import Dataset
import json
import random

def preprocess_function(examples):
    return {
        "cot_only": examples["chain_of_thought"],
        "full_format": (
            f"{examples['question']}\n"
            f"{examples['chain_of_thought']}"
        ), 
    }

def load_data(test_size=100, input_type="full_format"):
    print("Loading dataset...")
    processed_data = []
    
    # Replace with your own dataset
    with open("data/vae_train.jsonl") as f:
        data = [json.loads(line) for line in f]
    
    for sample in data:
        processed_data.append({
            "question": sample['input'],
            "chain_of_thought": sample["output"],
        })
    
    hf_dataset = Dataset.from_list(processed_data)
    split_dataset = hf_dataset.train_test_split(test_size=test_size)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset = train_dataset.map(preprocess_function, num_proc=112)
    eval_dataset = eval_dataset.map(preprocess_function, num_proc=112)
    
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)
    print("Dataset loaded successfully...")

    # # Inference examples.
    with open("data/vae_val.jsonl") as f:
        train_10_inference_examples = [json.loads(line) for line in f]
        random.seed(42)
        train_10_inference_examples = random.sample(train_10_inference_examples, 200)

    if input_type == "full_format":
        train_10_inference_examples = [i['input']+ "\n" + i['output'] for i in train_10_inference_examples]

    lines = train_10_inference_examples

    return train_dataset, eval_dataset, lines