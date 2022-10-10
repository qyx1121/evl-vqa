import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import json
import math

def tokenize(
    seq,
    tokenizer,
    add_special_tokens=True,
    max_length=10,
    dynamic_padding=True,
    truncation=True,
):
    """
    :param seq: sequence of sequences of text
    :param tokenizer: bert_tokenizer
    :return: torch tensor padded up to length max_length of bert tokens
    """
    tokens = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        padding="longest" if dynamic_padding else "max_length",
        truncation=truncation,
    )["input_ids"]
    return torch.tensor(tokens, dtype=torch.long)

def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    """ Precomputes GloVe answer embeddings for all answers in the vocabulary """
    a2id = json.load(open(vocab_path, "r"))
    # a2id['[UNK]'] = 0
    id2a = {v: k for k, v in a2id.items()}
    a2v = tokenize(
        list(a2id.keys()),
        bert_tokenizer,
        add_special_tokens=True,
        max_length=amax_words,
        dynamic_padding=True,
        truncation=True,
    )
    if torch.cuda.is_available():
        a2v = a2v.cuda()  # (vocabulary_size, 1, we_dim)
    return a2id, id2a, a2v


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)