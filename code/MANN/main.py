import argparse
import data
import os
import importlib.util
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Check if submission module is present.  If it is not, then main() will not be executed.
use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import DataGenerator, MANN


def eval(y_pred, Y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_pred = torch.squeeze(y_pred[:,-1,:])
    Y = torch.squeeze(Y[:, -1, :])
    (B,) = y_pred.size()
    for i in range(B):
        if Y[i] == 1:
            if torch.round(y_pred[i]) == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if torch.round(y_pred[i]) == 1:
                FP = FP + 1
            else:
                TN = TN + 1
    try:
        p = TP/(TP + FP)
        r = TP/(TP + FN)
        F1 = 2 * p * r / (p + r)
        accu = (TP+TN) / (TP+TN+FP+FN)

        return F1,p,r,accu
    except:
        return 0,0,0,0


def meta_step(images, labels, model, optim, eval=False):

    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: ", device)

    torch.manual_seed(config.random_seed)

    writer = SummaryWriter(f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_H{config.hidden_dim}_B{config.meta_batch_size}_L2_RS_{config.data_type}")

    # Create Data Generator
    # This will sample meta-training and meta-testing tasks

    meta_train_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="train",
        file_name=f"C:\\Temp\\CS330\\Project\\data\\american_bankruptcy_{config.data_type}.csv",

        DEVICE=device,

        has_title=True,

    )
    meta_train_loader = iter(
        torch.utils.data.DataLoader(
            meta_train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    meta_test_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="test",
        file_name=r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",
        DEVICE=device,

        has_title=True,
    )
    meta_test_loader = iter(
        torch.utils.data.DataLoader(
            meta_test_iterable,
            batch_size=config.meta_batch_size * 16,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # Create model
    model = MANN(config.num_classes, config.num_shot + 1, config.hidden_dim)

    if(config.compile == True):
        try:
            model = torch.compile(model, backend=config.backend)
            print(f"MANN model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    maxF1 = 0
    for step in tqdm(range(config.meta_train_steps)):
        ## Sample Batch
        ## Sample some meta-training tasks

        i, l = next(meta_train_loader)
        i, l = i.to(device), l.to(device)


        ## Train
        _, ls = meta_step(i, l, model, optim)
        writer.add_scalar("Loss/train", ls, step)

        i, l = next(meta_test_loader)
        i, l = i.to(device), l.to(device)
        pred, tls = meta_step(i, l, model, optim, eval=True)


        ## Evaluate
        ## Get meta-testing tasks
        if (step + 1) % config.eval_freq == 0:
            F1, p, r, accu = eval(pred, l)
            if F1 > maxF1:
                maxF1 = F1
            writer.add_scalar("Loss/test", tls, step)
            #print("Precision ",str(p)," Recall ",str(r) , " F1 ",str(F1), " accu ", str(accu))
            writer.add_scalar("F1/test", F1, step)
            pass
    print("Max F1: " + str(maxF1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # normal normal_TSSyn  normal_SMOTE_Half  normal_SMOTE_By_Year
    parser.add_argument("--data_type", type=str, default="normal")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_shot", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--meta_train_steps", type=int, default=10000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()

    main(args)


