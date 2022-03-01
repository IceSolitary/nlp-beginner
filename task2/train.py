import torch
import torch.nn as nn
import numpy as np
from model import LSTM, TextCNN, init_network
import torch.optim as optim
import dataloader
from dataloader import getData
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, val_iter, test_iter, len_vocab, vocab = getData(".", "data/train.csv", "data/val.csv", "data/test.csv",
                                                            batch_size=128)

model = LSTM(len_vocab)
model_ = TextCNN(len_vocab, 50)
init_network(model)
init_network(model_)
"""
将前面生成的词向量矩阵拷贝到模型的embedding层
这样就自动的可以将输入的word index转为词向量
"""
model.embedding.weight.data.copy_(vocab.vectors)
model_.embedding.weight.data.copy_(vocab.vectors)
model.to(DEVICE)
model_.to(DEVICE)

# 训练
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer_ = optim.Adam(model_.parameters(), lr=0.0001)

n_epoch = 20

best_val_acc_lstm = 0
best_val_acc_textCNN = 0

loss_function = nn.CrossEntropyLoss()
loss_function_ = nn.CrossEntropyLoss()

epoch_accs_lstm = []
epoch_losses_lstm = []
epoch_accs_textCNN = []
epoch_losses_textCNN = []

for epoch in range(n_epoch):
    batch_loss_lstm = 0.0
    batch_loss_textCNN = 0.0

    model.train()
    model_.train()
    for batch_idx, batch in enumerate(train_iter):
        data, length = batch.Phrase
        target = batch.Sentiment
        target = target.to(DEVICE)
        optimizer.zero_grad()

        out_lstm = model(data, length)
        loss_lstm = loss_function(out_lstm, target)

        out_textCNN = model_(data)
        loss_textCNN = loss_function_(out_textCNN, target)

        loss_lstm.backward()
        loss_textCNN.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer_.step()

        batch_loss_lstm += loss_lstm.item()
        batch_loss_textCNN += loss_textCNN.item()

    print("epoch{:d}: LSTM loss : {:.4f}".format(epoch+1, batch_loss_lstm/len(train_iter)))
    print("epoch{:d}: TextCNN loss : {:.4f}".format(epoch + 1, batch_loss_textCNN / len(train_iter)))
    epoch_losses_lstm.append(batch_loss_lstm/len(train_iter))
    epoch_losses_textCNN.append(batch_loss_textCNN / len(train_iter))

    model.eval()
    val_accs_lstm = []
    val_accs_textCNN = []
    for batch_idx, batch in enumerate(val_iter):
        data, length = batch.Phrase
        target = batch.Sentiment
        # target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        out_lstm = model(data, length)
        out_textCNN = model_(data)

        _, y_pre_lstm = torch.max(out_lstm, -1)
        _, y_pre_textCNN = torch.max(out_textCNN, -1)

        val_acc_lstm = torch.mean(y_pre_lstm.eq(target).float(), dim=-1)
        val_acc_textCNN = torch.mean(y_pre_textCNN.eq(target).float(), dim=-1)

        val_accs_lstm.append(val_acc_lstm)
        val_accs_textCNN.append(val_acc_textCNN)

    acc_lstm = torch.tensor(val_accs_lstm).mean()
    acc_textCNN = torch.tensor(val_accs_textCNN).mean()
    if acc_lstm > best_val_acc_lstm:
        print('val_lstm acc : %.4f > %.4f saving model' % (acc_lstm.item(), best_val_acc_lstm))
        torch.save(model.state_dict(), 'lstm_params.pkl')
        best_val_acc_lstm = acc_lstm
    print('val acc: %.4f' % acc_lstm)
    epoch_accs_lstm.append(acc_lstm)

    if acc_textCNN > best_val_acc_textCNN:
        print('val_textCNN acc : %.4f > %.4f saving model' % (acc_textCNN.item(), best_val_acc_textCNN))
        torch.save(model.state_dict(), 'textCNN_params.pkl')
        best_val_acc_textCNN = acc_textCNN
    print('val acc: %.4f' % acc_textCNN)
    epoch_accs_textCNN.append(acc_textCNN)

plt.figure(figsize=(10,8))
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(np.arange(len(epoch_losses_lstm)), np.array(epoch_losses_lstm), color="blue", label='LSTM')
plt.plot(np.arange(len(epoch_losses_textCNN)), np.array(epoch_losses_textCNN), color="red", label='TextCNN')
plt.legend()
plt.grid()

plt.figure(figsize=(10,8))
plt.title("accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(np.arange(len(epoch_accs_lstm)), np.array(epoch_accs_lstm), color="blue", label="LSTM")
plt.plot(np.arange(len(epoch_accs_textCNN)), np.array(epoch_accs_textCNN), color="red", label="TextCNN")
plt.legend()
plt.grid()
plt.show()





