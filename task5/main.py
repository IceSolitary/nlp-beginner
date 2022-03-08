import numpy as np
import torch
from torch.autograd import Variable
from dataset import Poem_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Seq_Generation_model

def main(batch_size,
         embedding_size,
         hidden_size,
         lr=0.001,
         epochs=50,
         checkpoint='net_params.pkl'
         ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    poem_data = Poem_Dataset("poetryFromTang.txt")
    vocab = poem_data.words
    word_int_map = poem_data.word_int_map
    print(10*"-", "Finish data loading", 10*"-")

    train_dataloader = DataLoader(poem_data, shuffle=True, batch_size=batch_size, collate_fn=poem_data.collate_fn)

    model = Seq_Generation_model(vocab_size=len(vocab),
                                 embedding_size=embedding_size,
                                 hidden_size=hidden_size,
                                 device=device
                                 ).to(device)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    train_losses = []
    train_perplexity = []
    epochs_count = []

    print("\n",
          20 * "=",
          "Training model on device: {}".format(device),
          20 * "=")

    # train
    for epoch in range(epochs):
        epochs_count.append(epoch)
        epoch_time, epoch_loss, epoch_perplexity = train(model,
                                                         train_dataloader,
                                                         optimizer=optimizer,
                                                         loss_function=loss_function,
                                                         scheduler=scheduler,
                                                         vocab_size=len(vocab),
                                                         max_gradient_norm=5.0)
        train_losses.append(epoch_loss)
        train_perplexity.append(epoch_perplexity)
        print("-> Epoch: {:d}, Training time: {:.4f}s, loss = {:.4f}, perplexity: {:.4f}"
              .format(epoch, epoch_time, epoch_loss, epoch_perplexity))

    torch.save(model.state_dict(), 'net_params.pkl')
    print("finish save model")


def train(model,
          dataloader,
          optimizer,
          scheduler,
          loss_function,
          vocab_size,
          max_gradient_norm):

    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0


    batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(batch_iterator):
        batch = torch.tensor(batch)
        x = batch[:, 0:-1].to(device)
        y = batch[:, 1:].to(device)

        optimizer.zero_grad()

        pre = model(x)
        loss = loss_function(pre.transpose(1, 2), y)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        # scheduler.step()
        running_loss += loss.item()
        perplexity = torch.exp(loss).item()

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_perplexity = torch.exp(torch.tensor(epoch_loss)).item()

    return epoch_time, epoch_loss, epoch_perplexity


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = predict
    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige = []
    for w in poem:
        if w == 'B':
            break
        if w == 'E':
            shige.append('。')
            break
        shige.append(w)
    str = ""
    for s in shige:
        str =  str + s
    print(str)
    # poem_sentences = poem.split('。')
    # for s in poem_sentences:
    #     if s != '' and len(s) > 10:
    #         print(s + '。')


def generate(begin_word,
             embedding_size=50,
             hidden_size=32,
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    poem_data = Poem_Dataset("poetryFromTang.txt")
    vocab = poem_data.words
    word_int_map = poem_data.word_int_map

    model = Seq_Generation_model(vocab_size=len(vocab),
                                 embedding_size=embedding_size,
                                 hidden_size=hidden_size,
                                 device=device
                                 ).to(device)

    model.load_state_dict(torch.load('net_params.pkl'))

    model.eval()

    poem = begin_word
    word = begin_word
    while word != poem_data.end_token:
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = torch.from_numpy(input).to(device)
        input = input.unsqueeze(0)
        output = model(input, is_test=True)
        word = to_word(output, vocabs=vocab)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem


# main(batch_size=1,
#      embedding_size=50,
#      hidden_size=32)

pretty_print_poem(generate("春"))
pretty_print_poem(generate("红"))
pretty_print_poem(generate("山"))
pretty_print_poem(generate("夜"))
pretty_print_poem(generate("湖"))
pretty_print_poem(generate("君"))




