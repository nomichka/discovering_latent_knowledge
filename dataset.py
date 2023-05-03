
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F

# make sure to install promptsource, transformers, and datasets!
# from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset


############# Data #############
class ContrastInContextDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from promptsource and a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(self, raw_dataset, tokenizer, context_num=10, corrupt_prob=0,
                 model_type="encoder_decoder", use_decoder=False, device="cuda", text_key="content"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # for formatting the answers
        self.model_type = model_type
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert self.model_type != "encoder"

        # prompt
        # prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        # self.prompt = all_prompts[prompt_name_list[prompt_idx]]
        self.prompt = None

        # context
        self.context_num = context_num
        self.corrupt_prob = corrupt_prob

        self.text_key = text_key

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt):
        """
        Tokenize a given natural language prompt (from after applying self.prompt to an example)
        
        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), or
        (2) feed the question the encoder and the answer to the decoder, creating contrast pairs using the decoder hidden states
        
        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt
        
        # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        if self.model_type == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer)
        elif self.model_type == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer)
        else:
            input_ids = self.get_decoder_input_ids(question, answer)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids


    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer 
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models.
        This is the same as get_encoder_input_ids except that we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_encoder_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use the encoder hidden states or the decoder hidden states.
        """
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer(answer, truncation=True, padding="max_length", return_tensors="pt")
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
        
        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids

    def get_decoder_input_ids_in_context(self, questions, answers):
        combined_input = [question + " " + answer for question, answer in zip(questions, answers)] + [self.tokenizer.eos_token]
        combined_inputs = (" ").join(combined_input)
        input_ids = self.tokenizer(combined_inputs, truncation=True, padding="max_length", return_tensors="pt")

        return combined_input, input_ids

    def encode_in_context(self, prompt, context_prompts):
        # get question and answer from prompt
        question, answer = prompt
        context_questions = [context_prompt[0] for context_prompt in context_prompts].append(question)
        context_answers = [context_prompt[1] for context_prompt in context_prompts].append(answer)
        if self.model_type == "decoder":
            combined_input, input_ids = self.get_decoder_input_ids_in_context(context_questions, context_answers)
        else:
            print("unsupported!")
        
        # # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        # if self.model_type == "encoder_decoder":
        #     input_ids = self.get_encoder_decoder_input_ids(question, answer)
        # elif self.model_type == "encoder":
        #     input_ids = self.get_encoder_input_ids(question, answer)
        # else:
        #     input_ids = self.get_decoder_input_ids(question, answer)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        print(input_ids["input_ids"].shape)
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return combined_input, input_ids


    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]
        # text, true_answer = data["content"], data["label"]
        text, true_answer = data[self.text_key], data["label"]

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        neg_prompt = f"Review: {text}. Sentiment: negative"
        pos_prompt = f"Review: {text}. Sentiment: positive"

        if self.context_num > 0:
            # get the random context
            context_inds = np.random.uniform(low=0, high=len(self.raw_dataset), size=self.context_num).astype(np.int32)
            context_data = self.raw_dataset.select(context_inds)
            context_texts, context_true_answers = context_data["content"], context_data["label"]

            # # corrupt the context with certain probability self.corrupt_prob
            # flip_mask = np.array(np.random.choice(a=[0, 1], size=self.context_num, p=[1 - self.corrupt_prob, self.corrupt_prob]), dtype=np.int32)
            # context_true_answers = abs(context_true_answers - flip_mask)

            context_answers = ["positive" if context_true_answer == 1 else "negative" for context_true_answer in context_true_answers]

            # # get the possible labels
            # # (for simplicity assume the binary case for contrast pairs)
            # label_list = self.prompt.get_answer_choices_list(data)
            # assert len(label_list) == 2, print("Make sure there are only two possible answers! Actual number of answers:", label_list)
            
            # construct a list of tuples of (prompt, output)
            context_prompts = [f"Review: {context_text}. Sentiment: {context_answer}" for context_text, context_answer in zip(context_texts, context_answers)]
            
            combined_input = ("\n").join(context_prompts)
            
            neg_combined_input = combined_input + "\n" + neg_prompt + self.tokenizer.eos_token
            pos_combined_input = combined_input + "\n" + pos_prompt + self.tokenizer.eos_token
        else:
            neg_combined_input = neg_prompt + self.tokenizer.eos_token
            pos_combined_input = pos_prompt + self.tokenizer.eos_token


        neg_ids = self.tokenizer(neg_combined_input, truncation=True, padding="max_length", return_tensors="pt")
        pos_ids = self.tokenizer(pos_combined_input, truncation=True, padding="max_length", return_tensors="pt")

        print("length of tokens", len(self.tokenizer.encode(neg_combined_input, truncation=False)))


        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and self.model_type == "encoder_decoder":
            assert (neg_ids["decoder_input_ids"] - pos_ids["decoder_input_ids"]).sum() != 0, print("The decoder_input_ids for the contrast pairs are the same!", neg_ids, pos_ids)
        else:
            assert (neg_ids["input_ids"] - pos_ids["input_ids"]).sum() != 0, print("The input_ids for the contrast pairs are the same!", neg_ids, pos_ids)


        # return the tokenized inputs, the text prompts, and the true label
        return neg_ids, pos_ids, neg_prompt, pos_prompt, true_answer

    
def get_dataloader(dataset_name, split, tokenizer, batch_size=16, num_examples=1000, context_num=10, corrupt_prob=0.0,
                   model_type="encoder_decoder", use_decoder=False, device="cuda", pin_memory=True, num_workers=1):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    np.random.seed(0)

    # load the raw dataset
    raw_dataset = load_dataset(dataset_name)[split]

    # # load all the prompts for that dataset
    # all_prompts = DatasetTemplates(dataset_name)
    if dataset_name == "amazon_polarity":
        text_key = "content"
    elif dataset_name == "sst2":
        text_key = "sentence"
    # create the ConstrastDataset
    contrast_dataset = ContrastInContextDataset(raw_dataset, tokenizer, context_num=context_num, corrupt_prob=corrupt_prob,
                                       model_type=model_type, use_decoder=use_decoder, device=device, text_key=text_key)
    
    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(len(contrast_dataset))

    # # remove examples that would be truncated (since this messes up contrast pairs)
    # prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
    # prompt = all_prompts[prompt_name_list[prompt_idx]]
    keep_idxs = random_idxs
    # keep_idxs = []
    # for idx in random_idxs:
    #     question, answer = prompt.apply(raw_dataset[int(idx)])
    #     input_text = question + " " + answer
    #     if len(tokenizer.encode(input_text, truncation=False)) < tokenizer.model_max_length - 2:  # include small margin to be conservative
    #         keep_idxs.append(idx)
    #         if len(keep_idxs) >= num_examples:
    #             break

    # Fix the number of idxs
    if len(keep_idxs) > num_examples:
        keep_idxs = keep_idxs[: num_examples]

    # # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader