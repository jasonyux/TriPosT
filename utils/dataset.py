import torch
import random
import jsonlines
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from typing import List, Dict
from transformers import LlamaTokenizer, LlamaTokenizerFast
from utils.utils import find_sub_list


class LogicalDeductionFeedback(Dataset):
    def __init__(self, 
            data_path,
            tokenizer,
            shuffle=True,
            data_start_portion=0.0,
            data_end_portion=0.9,
            input_max_length=1024,
            target_max_length=200):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        with jsonlines.open(data_path) as reader:
            self.data = list(reader)
        # get data start and end idx
        data_start_idx = int(len(self.data) * data_start_portion)
        data_end_idx = int(len(self.data) * data_end_portion)
        self.data = self.data[data_start_idx:data_end_idx]

        self.formatted_data = self.format_data()
        if self.shuffle:
            random.shuffle(self.formatted_data)
        return

    def format_data(self):
        formatted_data = []
        for d in tqdm(self.data, desc="Encoding data"):
            question = d['question']
            formatted_choices = d['formatted_choices']
            corrupted_answer = d['meta_data']['corrupted_steps']
            explanation_prompt = d['explanation_prompt']
            feedback = d['feedback']

            # format input text
            input_text = f"""
            Q: {question}
            Options:
            {formatted_choices}
            Answer: Let's think step by step.
            {corrupted_answer}
            Feedback:
            """
            input_text = "\n".join([t.strip() for t in input_text.split("\n")]).strip()

            # format output text
            output_text = explanation_prompt + feedback
            output_text = output_text.strip()

            # tokenize
            model_inputs = self.tokenizer(
                text=input_text,
                max_length=self.input_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            model_targets = self.tokenizer(
                text=output_text,
                max_length=self.target_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            # change pad to -100
            model_targets[model_targets == self.tokenizer.pad_token_id] = -100
            model_data = {
                "input_ids": model_inputs["input_ids"][0],
                "attention_mask": model_inputs["attention_mask"][0],
                "labels": model_targets[0],
            }
            formatted_data.append(model_data)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.formatted_data[index]
    

class WordSortingFeedback(Dataset):
    def __init__(self, 
            data_path,
            tokenizer,
            shuffle=True,
            data_start_portion=0.0,
            data_end_portion=0.85,
            input_max_length=1024,
            target_max_length=256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        with jsonlines.open(data_path) as reader:
            self.data = list(reader)
        # get data start and end idx
        data_start_idx = int(len(self.data) * data_start_portion)
        data_end_idx = int(len(self.data) * data_end_portion)
        self.data = self.data[data_start_idx:data_end_idx]

        self.formatted_data = self.format_data()
        if self.shuffle:
            random.shuffle(self.formatted_data)
        return

    def format_data(self):
        formatted_data = []
        for d in tqdm(self.data, desc="Encoding data"):
            question = d['question']
            corrupted_answer = d['meta_data']['corrupted_steps']
            explanation_prompt = d['explanation_prompt']
            feedback = d['feedback']

            # format input text
            input_text = f"""
            Q: {question}
            Answer: Let's think step by step.
            {corrupted_answer}
            Feedback:
            """
            input_text = "\n".join([t.strip() for t in input_text.split("\n")]).strip()

            # format output text
            output_text = explanation_prompt + feedback
            output_text = output_text.strip()

            # tokenize
            model_inputs = self.tokenizer(
                text=input_text,
                max_length=self.input_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            model_targets = self.tokenizer(
                text=output_text,
                max_length=self.target_max_length, padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            # change pad to -100
            model_targets[model_targets == self.tokenizer.pad_token_id] = -100
            model_data = {
                "input_ids": model_inputs["input_ids"][0],
                "attention_mask": model_inputs["attention_mask"][0],
                "labels": model_targets[0],
            }
            formatted_data.append(model_data)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.formatted_data[index]


class SelfImproveDataset(torch.utils.data.Dataset):
    def __init__(self, 
            raw_data: List[Dict],
            tokenizer,
            end_data_idx=None,
            min_input_length=256,
            max_input_length=2048,
            mask_before_att_start_text=False,
            self_improve_section_weight=1.0,
            shuffle=False):
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.end_data_idx = end_data_idx
        self.tokenizer = tokenizer
        self.mask_before_att_start_text = mask_before_att_start_text
        self.self_improve_section_weight = self_improve_section_weight

        self._possible_sequence_length = self._get_sequence_length()
        self.data = self.encode_data(raw_data)
        if shuffle:
            # usually the training data files are ALREADY shuffled
            # in the case of few shot experiments, we want to explicitly shuffle the data
            random.seed(42)
            random.shuffle(self.data)
        return

    def _get_sequence_length(self):
        max_diff_lengths = 5
        if self.max_input_length == self.min_input_length:
            return [self.min_input_length]
        if self.max_input_length - self.min_input_length < 128:
            return [self.min_input_length, self.max_input_length]
        return np.linspace(self.min_input_length, self.max_input_length, num=max_diff_lengths, dtype=int).tolist()
    
    def _fixed_last_padding_id(self, input_ids, attention_mask):
        # ignore everything before non_pad_token_start
        non_pad_token_start = 0
        for i, i_id in enumerate(input_ids):
            if i_id != self.tokenizer.pad_token_id:
                non_pad_token_start = i
                break
        # ignore everything after non_pad_token_end + 1
        first_eos_after_non_pad_token = len(input_ids) - 1
        for i, i_id in enumerate(input_ids):
            if i < non_pad_token_start:
                continue
            if i_id == self.tokenizer.eos_token_id:
                first_eos_after_non_pad_token = i
                break
        attention_mask[:non_pad_token_start] = False
        attention_mask[first_eos_after_non_pad_token+1:] = False
        return attention_mask
    
    def _find_closest_sequence_length(self, text_to_encode, length):
        # find the closest sequence length
        if length > self.max_input_length:  # trajectories are too long
            return None, None
        
        sequence_length = self._possible_sequence_length[0]
        for possible_sequence_length in self._possible_sequence_length:
            if possible_sequence_length >= length:
                sequence_length = possible_sequence_length
                break
        # pad the input_ids and attention_mask
        padded_data = self.tokenizer(
            text_to_encode,
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = padded_data["input_ids"][0]
        # att_mask = padded_data["attention_mask"][0]

        att_mask = torch.ones(len(input_ids), dtype=torch.bool)
        # ensure eos_token_id is after non_padded_ids
        att_mask = self._fixed_last_padding_id(input_ids, att_mask)
        return input_ids, att_mask

    def encode_data(self, raw_data):
        # encode data to input_ids, attention_mask, and labels
        encoded_data = []
        for data in tqdm(raw_data[:self.end_data_idx], desc="Encoding data"):
            # first check how long it is
            full_input_ids = self.tokenizer.encode(data["text"])
            padded_full_input_ids, padded_attention_mask = self._find_closest_sequence_length(data["text"], len(full_input_ids))
            
            if padded_full_input_ids is None:
                # if cannot encode entire episode, then skip
                continue

            # update the full_attention_mask according to the start of the text
            padded_loss_mask = torch.ones_like(padded_attention_mask, dtype=torch.float)
            padded_weights = torch.ones_like(padded_attention_mask, dtype=torch.float)
            encode_start_text = data["text_att_start_text"]
            encoded_start_text_ids = self.tokenizer.encode(encode_start_text)
            if "\n" in encode_start_text and isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
                encoded_start_text_ids = encoded_start_text_ids[2:]  # this is encoded specially
            else:
                encoded_start_text_ids = encoded_start_text_ids[1:]

            start_position = find_sub_list(encoded_start_text_ids, padded_full_input_ids.tolist())[0]
            if self.mask_before_att_start_text:
                padded_loss_mask[:start_position] = 0.0
            if 'Feedback' in encode_start_text:
                padded_weights[start_position:] = self.self_improve_section_weight
            
            encoded_data.append({
                "input_ids": padded_full_input_ids,
                "attention_mask": padded_attention_mask,
                "loss_mask": padded_loss_mask,
                "weights": padded_weights,
                "labels": padded_full_input_ids
            })
        return encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]