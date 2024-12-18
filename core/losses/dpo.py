import torch
import torch.nn.functional as F

def get_log_prob(logits, labels, attention_mask):
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_and_apply_attention = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1) * attention_mask

    return gathered_and_apply_attention.mean(dim=-1)

def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob, strengths,
                       beta=1.0, ipo=True, label_smoothing=0.0, reference_free=False):

    pi_logratios = model_prefered_logprob - model_disprefered_logprob
    ref_logratios = ref_prefered_logprob - ref_disprefered_logprob
    # losses = -F.logsigmoid(beta * logits) = -F.logsigmoid(beta * (pi_logratios - ref_logratios))

    if(reference_free):
      ref_logratios = 0.0
      ref_prefered_logprob = 0.0
      ref_disprefered_logprob = 0.0

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)
    if(ipo):
        loss = (logits - beta/(2 * strengths)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        loss = loss.mean()
    else:
        loss = strengths * -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing # -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)) #
        loss = loss.mean()

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins, pi_logratios, ref_logratios