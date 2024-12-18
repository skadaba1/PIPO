import random 
import numpy as np
import torch
from tqdm import tqdm
from core.losses.mlm import replace_with_mask, generate, get_sequence_probs

# Define global variables, you must train a model and fit first
model = None
tokenizer = None
device = None
params = None
linear_model = None

def parse_seq(seq):
  return seq.split('/')[1]

def snp(seq, max_length=50):
  decide_if_snp = random.choice([True, False])
  masked_seq = replace_with_mask(seq, replacement_ratio=0.1, add_masks_distr_mean=3, snp=decide_if_snp)
  generated_seq = generate(masked_seq, model, tokenizer, device)
  logprobs = get_sequence_probs([generated_seq], model, tokenizer, max_length, device, batch_size=1).cpu().item()
  return generated_seq, logprobs

def evolve(seq, num_children=100, max_generations=20):
  generation_indx = 1
  max_length = 30
  logprobs = get_sequence_probs([seq], model, tokenizer, max_length, device, batch_size=1)
  logprobs = logprobs.cpu().item() if isinstance(logprobs, torch.Tensor) else logprobs
  score = linear_model(np.exp(logprobs), *params) - 15.0
  print(f'Strting seqeunce: {seq} at generation {generation_indx} with score: {score}')
  #logprobs = -float('inf')
  while score < 0: #logprobs > init_logprobs:

    if(generation_indx > max_generations):
      break

    best_child_score = -float('inf')
    best_child = ''

    for i in tqdm(range(num_children)):
      candidate_seq, candidate_logprobs = snp(seq)
      candidate_logprobs = candidate_logprobs.cpu() if isinstance(candidate_logprobs, torch.Tensor) else candidate_logprobs
      candidate_score = linear_model(np.exp(candidate_logprobs), *params) - 15.0
      if(candidate_score > best_child_score):
        #seq = candidate_seq
        best_child_score = candidate_score
        best_child = candidate_seq
        logprobs = candidate_logprobs

    score = best_child_score
    seq = best_child
    generation_indx += 1
    print(f'Generation {generation_indx} -> sequence: {seq} with score: {score}')

  return seq, score, generation_indx

def __main__():
  evolve('EEEIRTTNPVATEQYGMSPYGLLGRLEA', num_children=500, max_generations=20)