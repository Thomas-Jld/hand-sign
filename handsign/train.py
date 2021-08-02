import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import signclassifier
from data import SignCollection, load
from utils import onehot



def train(epochs = 100):
    DATA = "data/trainning.csv"
    SIZE = 16

    BATCH_SIZE = 256
    NUM_WORKERS = 6

    EPOCHS = epochs
    LR = 1e-3
    DECAY = 1e-4

    sign_data = load(DATA)
    sign_collection = SignCollection(sign_data)

    loader = DataLoader(
        sign_collection,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = signclassifier().cuda()

    try:
        model.load_state_dict(torch.load("saves/best.pt"))
    except:
        print("No model saved")

    optim = AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

    criterion = nn.MSELoss().cuda()

    best_acc = 0.0


    for epoch in tqdm(range(EPOCHS), "Epoch", position=0, leave=True):

        with tqdm(loader, f"Train [{epoch + 1}/{EPOCHS}]", position=0, leave=True) as pbar:
            total_loss = 0.0
            acc = 0.0

            model.train()
            for id, data in pbar:
                id, data = id.cuda(), data.cuda()
                optim.zero_grad()

                pred = model(data)

                loss = criterion(pred, id)
                loss.backward()
                optim.step()

                total_loss += loss.item() / len(loader)

                acc += (
                    torch.argmax(pred, dim=1) == torch.argmax(id, dim=1)
                ).sum().item() / len(sign_data)


                pbar.set_postfix(
                    total_loss=total_loss,
                    acc=f"{acc * 100:.2f}%s"
                )

        if acc > best_acc:
            torch.save(model.state_dict(), "saves/best.pt")
            tqdm.write('Saved best.')
            best_acc = acc
    # torch.save(model.state_dict(), f"saves/best.pt")

train(1000)
