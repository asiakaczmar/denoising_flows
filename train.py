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
from flow_utils import get_new_model_log_paths, using
rnd = np.random.RandomState(SEED)
logpath, checkpoints_path = get_new_model_log_paths()
writer = SummaryWriter(log_dir=logpath)


def create_stylegan():
    impl = 'cuda'  # 'ref' if cuda is not available in your machine
    gpu = True  # False if tensorflow cpu is used
    weights_name = 'ffhq'  # face model trained by Nvidia
    return StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)


def normalize(img):
    return (img - np.min(img))/np.ptp(img)


def write_summaries(features,  w_real, w_real_not_clipped, w_created, loss, generator, step):
    writer.add_scalar("Loss/train", loss, step)
    w_created_matrix = np.tile(w_created,  [1, 18, 1])
    w_real_matrix = np.tile(w_real, [1,18,1])
    out_created_images = generator.synthesis_network(w_created_matrix, training=False)
    out_real_images = generator.synthesis_network(w_real_matrix, training=False)
    out_full_matrix = generator.synthesis_network(w_real_not_clipped, training=False)
    resized_images = resize(out_created_images[0], [3, 10, 10])
    writer.add_image('created_image', resize(normalize(out_created_images.numpy()[0]), [3,128,128]), step)
    writer.add_image('real_image', resize(normalize(out_real_images.numpy()[0]), [3, 128, 128]), step)
    #writer.add_image('real_image_full_w', normalize(out_full_matrix.numpy()[0]), step)
    writer.add_image('downsized_created_image', normalize(resized_images), step)
    features_reshaped = np.reshape(features, [features.shape[0],3, 10,10])[0]
    writer.add_image('features', normalize(features_reshaped.numpy()), step)
    mse_downscaled = np.mean((features[0].detach().numpy() - np.reshape(resized_images, [300]))**2)
    writer.add_scalar('MSE_downscaled/train', mse_downscaled, step)
    writer.flush()


def train_step(x, context, optimizer, epoch, flow):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=context).mean()
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
        for x, c in train_loader:
            step += 1
            noisy_inp = x[:,0:1,:] + np.random.normal(scale=0.00005, size=x[:,0:1,:].shape).astype(np.float32)
            loss = train_step(noisy_inp[:,0,:], c, optimizer, epoch, flow)
            if step % LOGGING_INTERVAL == 0:
                with torch.no_grad():
                    output, probs = flow.sample_and_log_prob(1, context=c)
                    write_summaries(c, noisy_inp, x, output, loss, generator, step)
        if epoch%10==0:
            save_checkpoint(epoch, flow, optimizer, loss)

if __name__ == '__main__':
    train_loop()

