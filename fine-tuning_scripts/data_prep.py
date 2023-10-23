from transformers import AutoTokenizer
from prompter import Prompter
from itertools import chain
from functools import partial
from datasets import load_dataset

tokenizer = None
prompter = Prompter()

def format_goat(sample):
    return f"<s>[INST] {sample['instruction']} [/INST] {sample['output']}</s>"


def generate_and_tokenize_prompt(data_point):
        full_prompt = format_goat(data_point)
        tokenized_full_prompt = tokenizer(full_prompt)
        return tokenized_full_prompt

# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    batch_chunk_length = 0
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length
    else:
        batch_chunk_length = batch_total_length
    
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def data_prep(dataset_id, rev="master"):
    dataset = load_dataset(dataset_id, split="train", revision=rev)

    model_id = "meta-llama/Llama-2-7b-hf"

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = dataset.map(
            generate_and_tokenize_prompt, remove_columns=list(dataset.features)
        ).map(
            partial(chunk, chunk_length=2048), batched=True,
        )

    return train_data