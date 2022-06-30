import sys
sys.path.append('/cluster/home/kjoanna/denoising_flows')
sys.path.append('/cluster/home/kjoanna/StyleGAN2-TensorFlow-2.x')
import os
import numpy as np
from tqdm import tqdm

from datasets import get_dataloader
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from nflows.flows import MaskedAutoregressiveFlow
from stylegan2_generator import StyleGan2Generator
from settings import SEED, LR, FEATURES, HIDDEN_FEATURES, CONTEXT_FEATURES, NUM_EPOCHS, LOGGING_INTERVAL
from flow_utils import get_new_model_log_paths, using

rnd = np.random.RandomState(SEED)
logpath, checkpoints_path = get_new_model_log_paths(conditional=False)
writer = SummaryWriter(log_dir=logpath)


def save_checkpoint(epoch, model, optimizer, loss):
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoints_path + '/' + str(epoch))


def train_step(x, optimizer, epoch, flow):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    return loss


def train_loop():
    step = 0
    flow = MaskedAutoregressiveFlow(features=FEATURES,
                                    hidden_features=HIDDEN_FEATURES)
    optimizer = optim.Adam(flow.parameters(), lr=LR)
    train_loader = get_dataloader()
    for epoch in range(NUM_EPOCHS):
        print('epoch ' + str(epoch))
        for x, c in train_loader:
            step += 1
            noisy_inp = x[:, 0:1, :] + np.random.normal(scale=0.00005, size=x[:, 0:1, :].shape).astype(np.float32)
            loss = train_step(noisy_inp[:, 0, :], c, optimizer, epoch, flow)
            if step % LOGGING_INTERVAL == 0:
                writer.add_scalar("Loss/train", loss, step)
        if epoch % 10 == 0:
            save_checkpoint(epoch, flow, optimizer, loss)


if __name__ == '__main__':
    train_loop()