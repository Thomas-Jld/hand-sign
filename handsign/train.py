import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import signclassifier
from data import SignCollection, load
from utils import onehot

import onnxruntime
import numpy as np

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
        model.load_state_dict(torch.load("saves/handsign.pt"))
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
            torch.save(model.state_dict(), "saves/handsign.pt")
            tqdm.write('Saved best.')
            best_acc = acc
    # torch.save(model.state_dict(), f"saves/best.pt")

train(1000)

def export_to_onnx():
    """ Exports the pytorch model to onnx """
    DATA = "data/trainning.csv"
    sign_data = load(DATA)
    sign_collection = SignCollection(sign_data)

    model = signclassifier()
    model.load_state_dict(torch.load("saves/handsign.pt"))
    model.eval()
    input = sign_collection[2000][1]
    torch_out = model(input)

    torch.onnx.export(
        model,
        input,
        "saves/handsign.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}})

    ort_session = onnxruntime.InferenceSession("saves/handsign.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(np.argmax(ort_outs[-1]))
    print(torch.argmax(torch_out.squeeze(0)))
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

export_to_onnx()
