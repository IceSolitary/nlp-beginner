import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np


def train(model,
          dataloader,
          optimizer,
          word2idx,
          max_gradient_norm):
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    running_F1 = 0.0

    batch_iterator = tqdm(dataloader)

    for batch_index, batch in enumerate(batch_iterator):
        batch_start = time.time()

        token_seq = batch[0].to(device)
        label_seq = batch[1].to(device)
        mask = get_pad_mask(token_seq, word2idx).to(device)

        optimizer.zero_grad()

        loss = model(token_seq, label_seq, mask)
        loss = loss.mean()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        with torch.no_grad():
            pred_score, pred_seq = model.predict(token_seq, mask)
            running_F1 += compute_F1(pred_seq, label_seq, mask)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_F1 = running_F1 / len(dataloader)

    return epoch_time, epoch_loss, epoch_F1


def validate(model, dataloader, word2idx):
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_F1 = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            token_seq = batch[0].to(device)
            label_seq = batch[1].to(device)
            mask = get_pad_mask(token_seq, word2idx).to(device)
            loss = model(token_seq, label_seq, mask).to(device)
            loss = loss.mean()

            running_loss += loss.item()
            pred_score, pred_seq = model.predict(token_seq, mask)
            running_F1 += compute_F1(pred_seq, label_seq, mask)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_F1 = running_F1 / (len(dataloader))

    return epoch_time, epoch_loss, epoch_F1


def load_vocab(vocab_file):
    vocab = np.load(vocab_file)
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}

    return word2idx, idx2word, vocab


def get_pad_mask(vec, word2idx):
    """
    :param vec: input_vector,shape [batch_size, seq_length]
    :return: mask: input_mask,shape [batch_size, seq_length]
    """

    mask = vec.ne(word2idx['PAD_TAG'])
    mask = mask.int()
    return mask


def compute_F1(pred, target, mask):
    eps = 1e-9
    length_list = torch.sum(mask, dim=-1) - 2

    confmartix = torch.zeros(9, 9)

    for batch_idx, length in enumerate(length_list):
        for len_idx in range(length):
            len_idx = len_idx + 1
            i = target[batch_idx, len_idx]
            j = pred[batch_idx, len_idx]
            if (i not in (9, 10, 11)) and (j not in (9, 10, 11)):
                confmartix[i][j] += 1

    all_pred = torch.mean(torch.diag(confmartix) / (torch.sum(confmartix, dim=0) + eps))
    all_recall = torch.mean(torch.diag(confmartix) / (torch.sum(confmartix, dim=1) + eps))
    all_f1 = 2 * all_recall * all_recall / (all_pred + all_recall + eps)

    return all_f1.item()
