
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embedding(sequences, max_length=40, tokenizer=None, model=None):

  tokens = tokenizer(sequences, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt").to(device)
  input_ids = tokens["input_ids"]
  attention_mask = tokens["attention_mask"]
  embeddings = model.esm.embeddings(input_ids, attention_mask)
  #print(embeddings.shape)

  return embeddings

def get_param_by_name(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param
    print(f"Error fetching param: {param_name}")
    return None  # If the parameter name is not found

def compute_attention(model, x, model_size=(30, '150M')):
  layers = model_size[0]
  attention_sum = None
  for layer_indx in range(layers):
    attention = model.esm.encoder.layer[layer_indx].attention(x)
    attention_sum = attention[0] if attention_sum is None else attention_sum + attention[0]
    return attention[0]/layers
  
def plot_attention(attention_heatmap):
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_heatmap, aspect='auto', cmap='viridis')

    # Add color bar
    plt.colorbar()

    # Add labels and title
    plt.xlabel('Token positions')
    plt.ylabel('Attention value')
    plt.title('Heatmap of Token Attentions')

    # Display the plot
    plt.show()

def get_relative_attention_scores(sequence, indexes=None, ignore_padding=True, model=None):
  embedding = get_embedding([sequence])
  attention = compute_attention(model, embedding)
  if(ignore_padding):
    attention_heatmap = attention[:, 0:len(sequence), :]
  else:
    attention_heatmap = attention
  attention_heatmap = torch.sum(attention_heatmap, dim=2).detach().cpu()

  if(indexes):
    ln = len(indexes[1])
    avg_for_cutsite = np.average((attention_heatmap[:, indexes[0]:indexes[0]+ln]))
    avg_overall = np.average(attention_heatmap)
    return avg_for_cutsite, avg_overall
  else:
    return attention_heatmap
  
def __main__(hits):
    cutsite_averages = []
    overall_averages = []
    for hit in hits:
        sequence = hit[0]
        indexes = hit[1][0]
        avg_for_cutsite, avg_overall = get_relative_attention_scores(sequence, indexes)
        cutsite_averages.append(avg_for_cutsite)
        overall_averages.append(avg_overall)

    plt.hist(cutsite_averages, label='cutsite av', alpha=0.7)
    plt.hist(overall_averages, label='overall av', alpha=0.7)
    plt.legend()
    plt.show()

    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(cutsite_averages, overall_averages, )

    print(f"K-S test statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    print(f"Mean of probs_ala_delta: {np.mean(cutsite_averages)}, median of probs_ala_delta: {np.median(cutsite_averages)}")
    print(f"Mean of probs_random_delta: {np.mean(overall_averages)}, , median of probs_random_delta: {np.median(overall_averages)}")

    heat_map_sum = None
    for hit in hits:
        sequence = hit[0]
        indexes = hit[1][0]
        heat_map = get_relative_attention_scores(sequence, None, False)
        heat_map_sum = heat_map if heat_map_sum is None else heat_map_sum + heat_map

    plot_attention(heat_map_sum)

if __name__ == "__main__":
    __main__()
     