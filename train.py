import sys
sys.path.append('/cluster/home/kjoanna/denoising_flows')
from datasets import get_dataloader
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from nflows.flows import MaskedAutoregressiveFlow
from torch.utils.tensorboard import SummaryWriter



import os
import numpy as np
sys.path.append('/cluster/home/kjoanna/StyleGAN2-TensorFlow-2.x')
from stylegan2_generator import StyleGan2Generator
from skimage.transform import resize
from tqdm import tqdm

import torch
import torch.optim as optim

from nflows.flows import MaskedAutoregressiveFlow
from settings import SEED, LR, FEATURES, HIDDEN_FEATURES, CONTEXT_FEATURES, NUM_EPOCHS, LOGGING_INTERVAL
from flow_utils import get_new_model_log_paths
rnd = np.random.RandomState(SEED)
logpath, checkpoints_path = get_new_model_log_paths()
writer = SummaryWriter(log_dir=logpath)


def create_stylegan():
    impl = 'cuda'  # 'ref' if cuda is not available in your machine
    gpu = True  # False if tensorflow cpu is used
    weights_name = 'ffhq'  # face model trained by Nvidia
    return StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)


def write_summaries(features, z_in, loss, generator, step):
    writer.add_scalar("Loss/train", loss, step)
    w = generator.mapping_network(np.reshape(z_in[0][:3].detach(), [3, 512]), training=False)
    out_images = generator.synthesis_network(w, training=False)
    resized_images = resize(out_images, [3, 10, 10, 3])
    import pdb; pdb.set_trace()
    writer.add_image('output_image', out_images.numpy()[0], step)
    writer.add_image('downsized_image', np.transpose(resized_images[0], [2,0,1], step)
    writer.add_image('features', np.transpose(np.reshape(features[0], [10,10,3]), [1,2,0]), step)
    mse_downscaled = torch.nn.MSELoss(features, resized_images)
    writer.add_scalar('MSE_downscaled/train', mse_downscaled, step)


def train_step(x, y, optimizer, epoch, flow):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    return loss


def save_checkpoint(epoch, model, optimizer, loss):
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoints_path + '/' + str(epoch))


def train_loop():
    step = 0
    flow = MaskedAutoregressiveFlow(features=FEATURES,
                                    hidden_features=HIDDEN_FEATURES,
                                    context_features=CONTEXT_FEATURES)
    generator = create_stylegan()
    optimizer = optim.Adam(flow.parameters(), lr=LR)
    train_loader = get_dataloader() 
    for epoch in range(NUM_EPOCHS):
        print('epoch ' + str(epoch)) 
        for x, y in train_loader:
            step += 1
            loss = train_step(x, y, optimizer, epoch, flow)
            if step % LOGGING_INTERVAL:
                output, probs = flow.sample_and_log_prob(3, context=y)
                write_summaries(y, output, loss, generator, step)
        save_checkpoint(epoch, flow, optimizer, loss)

if __name__ == '__main__':
    train_loop()

