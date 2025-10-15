from torch.utils.data import Dataset
import torch
import json
from torch.nn.utils.rnn import pad_sequence


class ThoughtDataCollator:
    def __init__(self, pad_token_id=0):
        """
        Args:
            pad_token_id (int): The token ID used for padding (e.g., tokenizer.pad_token_id).
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        Expects each item of the batch to be a dictionary with at least the following key:
          - "input_ids_q": a 1D tensor of token IDs.
        Other keys (e.g. "input", "reasoning_text", "gt_solutions") are preserved as lists.
        """
        # Extract the field with variable-length sequences.
        input_ids_q_list = [ex["input_ids_q"] for ex in batch]
        # Pad the sequences to the maximum length found in the batch.
        input_ids_q = pad_sequence(input_ids_q_list, batch_first=True, padding_value=self.pad_token_id)

        # Extract the field with variable-length sequences.
        output_ids_list = [ex["output_ids"] for ex in batch]
        # Pad the sequences to the maximum length found in the batch.
        output_ids = pad_sequence(output_ids_list, batch_first=True, padding_value=self.pad_token_id)

        # For the rest of the fields, preserve them as lists.
        input_list = [ex["input"] for ex in batch]
        reasoning_text_list = [ex["reasoning_text"] for ex in batch]
        gt_solutions_list = [ex["gt_solutions"] for ex in batch]


        return {
            "input_ids_q": input_ids_q,
            "output_ids": output_ids,
            "input": input_list,
            "reasoning_text": reasoning_text_list,
            "gt_solutions": gt_solutions_list,
        }

class ThoughtDataset(Dataset):
    """
    Returns a single example each time (for illustration),
    with the question, reasoning_chain, and answer fields.
    Then preprocesses them (tokenization etc.) on the fly
    in __getitem__.
    """
    def __init__(self, text_tokenizer, json_file, compress_rate=3, train=True, size=None):
        """
        Args:
            text_tokenizer: HF tokenizer for text LLaMA
            json_file: path to your JSONL file
            compress_rate: not used here, but presumably for some future logic
            train: whether this dataset is for training or not
            size: if given, only load up to 'size' lines from the file
        """
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.data = []
        self.compress_rate = compress_rate
        self.train = train

        # ------------------------------
        # 1) Read all non-empty lines
        # ------------------------------
        with open(json_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]  # discard empty lines

        # If size is not specified, use the total number of non-empty lines
        if size is None:
            size = len(lines)

        # Optionally limit how many lines you actually process
        lines = lines[:size]

        # ------------------------------
        # 2) Parse lines and build self.data
        # ------------------------------
        for line in lines:
            dic = json.loads(line)
            dic["input_prompt"] = dic["input"] + "\n"
            dic["reasoning_text"] = dic["input"] + "\n" + dic["output"]
            self.data.append(dic)

        # ------------------------------
        # 3) Final size and logging
        # ------------------------------
        self.size = len(self.data)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns a single example of:
          question, reasoning_chain, answer
        Then tokenizes the text + chunk
        and returns a dictionary for training.
        """

        ex = self.data[idx]

        text_enc = self.text_tokenizer(
            ex["input_prompt"],
            return_tensors="pt"
        )
        input_ids_q = text_enc["input_ids"][0]        # shape [T_text]

        output_text = self.text_tokenizer(
            ex["reasoning_text"],
            return_tensors="pt"
        )
        output_ids = output_text["input_ids"][0]  

        gt_solutions = None
        if not self.train:
            gt_solutions = ex["solutions"]

        return {
            "input_ids_q": input_ids_q,
            "input": ex["input"],
            "reasoning_text": ex["reasoning_text"],
            "output_ids": output_ids,
            "gt_solutions": gt_solutions,
        }

