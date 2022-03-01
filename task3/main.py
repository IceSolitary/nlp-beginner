import torch
from data.dataset import SNLI
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import writer
import numpy as np

from model.model import ESIM
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils import train, validate, load_vocab


def main(trainfile, devfile, testfile, vocabfile,
         batch_size=1024,
         hidden_size=100,
         learning_rate=0.004,
         num_class=3,
         tensorboardX=True,
         epochs=20,
         checkpoint=None,#r"./checkpoint/best.pth.tar",
         patience=5,
         max_grad_norm=10.0,
         target_dir=r"./checkpoint"
         ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    word2idx, idx2word, vocab = load_vocab(r"./glove_data/wordsList.npy")

    embedding = torch.from_numpy(np.load(r"./glove_data/wordVectors.npy"))
    # prepare data
    print(10*"-", "preparing data", 10*"-")
    train_data = SNLI(trainfile,
                      word2idx,
                      idx2word,
                      vocab)
    dev_data = SNLI(devfile,
                    word2idx,
                    idx2word,
                    vocab)
    # test_data = SNLI(testfile, vocabfile,
    #                   word2idx=word2idx,
    #                   idx2word=idx2word,
    #                   vocab=vocab)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=train_data.collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size, collate_fn=dev_data.collate_fn)
    # test_data = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=test_data.collate_fn)

    # 训练
    model = ESIM(vocab_size=len(vocab), embedding_size=300, hidden_size=hidden_size,device=device,embedding=embedding).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=10)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             dev_dataloader,
                                             loss_function)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch,  start_epoch + epochs):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_dataloader,
                                                       optimizer,
                                                       loss_function,
                                                       epoch,
                                                       max_grad_norm)
        # if tensorboardX:
        #     for param in model.named_parameters():
        #         writer.add_histogram(param[0], param[1].grad, epochs)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_accuracy)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          dev_dataloader,
                                                          loss_function)

        valid_losses.append(epoch_loss)
        valid_accs.append(valid_accs)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # # Save the model at each epoch.
        # torch.save({"epoch": epoch,
        #             "model": model.state_dict(),
        #             "best_score": best_score,
        #             "optimizer": optimizer.state_dict(),
        #             "epochs_count": epochs_count,
        #             "train_losses": train_losses,
        #             "valid_losses": valid_losses},
        #            os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r", label="train_losses")
    plt.plot(epochs_count, valid_losses, "-b", label="valid_losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Cross entropy loss")
    plt.show()

    plt.figure()
    plt.plot(epochs_count, train_accs, "-r", label="train_losses")
    plt.plot(epochs_count, valid_accs, "-b", label="valid_losses")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.show()


main(trainfile=r"./snli_1.0/snli_1.0_train.jsonl",
     devfile=r"./snli_1.0/snli_1.0_dev.jsonl",
     testfile=r"./snli_1.0/snli_1.0_test.jsonl",
     vocabfile=r"./glove_data/wordsList.npy")
