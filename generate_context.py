import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from dataset import get_dataloader

if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()
    # main(args)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("Loading dataloader")
    dataloader = get_dataloader("amazon_polarity", "test", tokenizer, 0, batch_size=20, 
                                num_examples=100, model_type="decoder", use_decoder=True, device="cuda")
    