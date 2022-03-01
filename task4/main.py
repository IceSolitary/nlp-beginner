from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import writer
import os
import numpy as np
import torch
from dataset import CONLL2003

from model.model import SLModel
import matplotlib.pyplot as plt

from utils import train, validate, load_vocab
torch.manual_seed(256)


def main(trainfile, devfile, testfile, vocabfile,
         batch_size=256,
         hidden_size=50,
         learning_rate=0.01,
         epochs=20,
         checkpoint=None, #r"./checkpoint/best.pth.tar",
         patience=5,
         max_grad_norm=5.0,
         target_dir=r"./checkpoint",
         max_len=30
         ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    word2idx, idx2word, vocab = load_vocab(r"./glove_data/wordsList.npy")
    embedding = torch.from_numpy(np.load(r"./glove_data/wordVectors.npy"))

    # prepare data
    print(10 * "-", "preparing data", 10 * "-")
    train_data = CONLL2003(trainfile,
                           word2idx,
                           idx2word,
                           max_len=max_len)
    dev_data = CONLL2003(devfile,
                         word2idx,
                         idx2word,
                         max_len=max_len)
    print(10 * "-", "OK", 10 * "-")

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=train_data.collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size, collate_fn=dev_data.collate_fn)

    # 训练
    model = SLModel(vocab_size=len(vocab), embedding_size=50, hidden_size=hidden_size, device=device,
                    batch_size=batch_size, label_size=12, tag_to_ix=train_data.labels_idx_dict, embedding=embedding)\
        .to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=10)

    best_F1 = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_f1 = []
    valid_f1 = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_F1 = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training SLModel model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs + start_epoch):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_F1 = train(model,
                                                 train_dataloader,
                                                 optimizer,
                                                 word2idx,
                                                 max_grad_norm)
        # if tensorboardX:
        #     for param in model.named_parameters():
        #         writer.add_histogram(param[0], param[1].grad, epochs)

        train_losses.append(epoch_loss)
        train_f1.append(epoch_F1)
        print("-> Training time: {:.4f}s, loss = {:.4f}, F1_score: {:.4f}"
              .format(epoch_time, epoch_loss, epoch_F1))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_F1 = validate(model,
                                                    dev_dataloader,
                                                    word2idx)

        valid_losses.append(epoch_loss)
        valid_f1.append(epoch_F1)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, F1_score: {:.4f}"
              .format(epoch_time, epoch_loss, epoch_F1))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_F1)

        # Early stopping on validation accuracy.
        if epoch_F1 < best_F1:
            patience_counter += 1
        else:
            best_F1 = epoch_F1
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_F1,
                        "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_f1, "-r", label="Training F1")
    plt.plot(epochs_count, valid_f1, "-b", label="Validation F1")
    plt.xlabel("epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.title("F1-score")
    plt.show()


main(trainfile=r"./CoNLL-2003/train.txt",
     devfile=r"./CoNLL-2003/dev.txt",
     testfile=r"./CoNLL-2003/test.txt",
     vocabfile=r"./glove_data/wordsList.npy")

