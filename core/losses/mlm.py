# Function to compute MLM loss for a batch of protein pairs
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.distributions import Categorical
from tokenizers import Tokenizer
from dpo import get_log_prob
P_MASK=0.15
def compute_mlm_loss_batch(model, preferred_input_ids, preferred_mask, dispreferred_input_ids, dispreferred_mask, tokenizer=None):

    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    losses = None

    for input_ids, mask in [(preferred_input_ids, preferred_mask), (dispreferred_input_ids, dispreferred_mask)]:

        # Clone input IDs for labels
        labels = input_ids.clone()

        # Randomly mask 15% of the residues for each sequence in the batch
        masked_input_ids = input_ids.clone()
        for idx in range(input_ids.shape[0]):
            non_pad_indices = torch.where(input_ids[idx] != pad_token_id)[0]
            mask_indices = np.random.choice(non_pad_indices.cpu().numpy(), size=int(P_MASK * len(non_pad_indices)), replace=False)
            masked_input_ids[idx, mask_indices] = mask_token_id
            labels[idx, [i for i in range(input_ids.shape[1]) if i not in mask_indices]] = -100

        # Compute the MLM loss
        outputs = model(masked_input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        if(losses is None):
          losses = loss
        else:
          losses += loss

    return losses

def replace_with_mask(original_string, replacement_ratio, add_masks_distr_mean=2, max_length=50, snp=False):

    if(snp):
      num_replacements = 1
    else:
      num_replacements = int(len(original_string) * replacement_ratio)

    indices_to_replace = random.sample(range(len(original_string)), num_replacements)
    str_list = list(original_string)

    if(not snp):
      for index in indices_to_replace:
          str_list[index] = "<mask>"


      num_additional_masks = np.random.poisson(add_masks_distr_mean)
      for i in range(num_additional_masks):
          if(len(str_list) < max_length):
            add_position = random.choice(['beginning', 'end', 'none'])
            if add_position == 'beginning':
                str_list = ['<mask>'] + str_list
            elif( add_position == 'end'):
                str_list = str_list + ['<mask>']
            else:
              str_list = str_list
          else:
            break

    else:

      action = random.choice(['none', 'deletion', 'insertion', 'substitution'])
      if(action == 'deletion'):
        str_list[indices_to_replace[0]] = ''
      elif(action == 'insertion' and len(str_list) < max_length):
        str_list.insert(indices_to_replace[0], '<mask>')
      elif(action == 'substitution'):
        str_list[indices_to_replace[0]] = '<mask>'
      else:
        pass

    masked_sequence = ''.join(str_list)

    return masked_sequence

def replace_masks(masked_string, replacement_string):

    segments = masked_string.split("<mask>")
    result = ""
    for i, segment in enumerate(segments):
        result += segment
        if i < len(replacement_string):
            result += replacement_string[i]

    return result

def generate(seq, model, tokenizer, device):
  model.eval()
  top_k = 5
  inputs = tokenizer(seq, return_tensors="pt").to(model.device)

  with torch.no_grad():
      logits = model(**inputs).logits
  mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
  logits_at_masks = logits[0, mask_token_indices]

  # Apply top-k sampling
  top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
  probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
  predicted_indices = Categorical(probabilities).sample()
  predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

  aa_seq_masks = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')

  return replace_masks(seq, aa_seq_masks)

def get_sequence_probs(sequences, model, tokenizer, max_length, device, batch_size=16):
    model.eval()
    all_probs = []

    with torch.no_grad():
        # Iterate through sequences in batches
        for i in range(0, len(sequences), batch_size):

            batch_sequences = sequences[i:i + batch_size]

            tokens = tokenizer(batch_sequences, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = get_log_prob(outputs.logits, input_ids, attention_mask)

            all_probs.append(batch_probs)

    # Concatenate the results from all batches
    return torch.cat(all_probs, dim=0)

def sample_lognormal_between_0_and_1(mean, sigma):
    while True:
        sample = np.random.lognormal(mean, sigma)
        if 0 < sample < 1:
            return sample