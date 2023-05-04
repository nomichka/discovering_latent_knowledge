# Explore Discovering Latent Knowledge Without Supervision 

This repository is a fork of the original codebase https://github.com/collin-burns/discovering_latent_knowledge. For a detailed README.md file, please go to the their repository.

The full paper can be found in directory `paper`.

## 1. Use logical conjunctions to find the “truth” direction of the classifier

This part is done by Naomi Bashkansky. Go through `jupyter_notebook/conjunction.ipynb` for an overview.

## 2. Towards better eliciting latent knowledge on autoregressive models. 
This part is done by Chuyue Tang. Some bugs in the original codebase are fixed and codes are modified for running experiments in this project.

### 1) Using CCS
First, use `generate.py` for (1) creating contrast pairs, and (2) generating hidden states from a model. 
```
python generate.py --model_name gpt-j --dataset_name amazon_polarity  --num_examples 500 --all_layers 
```
To generate multi-shot context, you can specify `context_num`.
```
python generate.py --model_name gpt-j --dataset_name amazon_polarity --num_examples 500 --context_num 10 --all_layers
```
You can also get data with and without context at the same time by setting `context_both`(which is later used by `visualize.py` to generate PCA graph comparing two settings). 
```
python generate.py --model_name gpt-j --dataset_name amazon_polarity --num_examples 100 --context_num 10 --all_layers --context_both
python visualize.py --model_name gpt-j --dataset_name amazon_polarity --num_examples 100 --context_num 10 --all_layers --context_both
```
Due to the time limit, we do not guarantee that running these commands with other model or dataset settings will work out well.

After generating hidden states, you can use `evaluate.py` for running our main method, CCS, on those hidden states. **We highly recommend you to run this file using the same arguments as you run `generate.py`.** 

In addition to evaluating the performance of CCS, `evaluate.py` also verifies that logistic regression (LR) accuracy is reasonable. 
We also add a PCA baseline (which is referred to as TPC in the paper).
You will also get a plot with x-axis as layers and y-axis as accuracy for CCS, LR, and PCA.

### 2) Using VINC
Another part of this project is based on the under development codebase https://github.com/EleutherAI/elk.git.
First, install requirements based on their codebase. Then run
```
cd elk
python elk elicit EleutherAI/gpt-j-6b amazon_polarity --int8 True --max_examples 250 250 --num_variants 1 --num_shots 10 --corrupt_prob 0.0
```
Here, `num_variants` refers to how many different paraphrase prompts you want to use. By default it uses all available prompt formats from [promptsource](https://github.com/bigscience-workshop/promptsource); `num_shots` refers to how many context examples you want to include before the query statement; `corrupt_prob` is the probability of how each context example's label is flipped.

### Requirements

This code base was tested on Python 3.7.5 and PyTorch 1.12. It also uses the [datasets](https://pypi.org/project/datasets/) and [promptsource](https://github.com/bigscience-workshop/promptsource) packages for loading and formatting datasets. 



