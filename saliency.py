"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

import utils
import model.net as net
import model.data_loader as data_loader
import matplotlib.pyplot as plt
from torchvision.utils import save_image



parser = argparse.ArgumentParser()
parser.add_argument('--image_file', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

preprocess_for_save = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])


def calculate_saliency(model, image_file, model_dir):
    # Make sure the model is in "test" mode
    model.eval()

    x, x_for_save = load_image_as_tensor(image_file)

    if not os.path.isdir(os.path.join(model_dir, 'saliency_outputs')):
        os.mkdir(os.path.join(model_dir, 'saliency_outputs'))
    file_name = os.path.splitext(image_file)[0].replace('/', '_')
    full_directory = os.path.join(model_dir, 'saliency_outputs', file_name)
    if not os.path.isdir(full_directory):
        os.mkdir(full_directory)

    save_image(x_for_save, os.path.join(full_directory, image_file.replace('/', '_')))

    # Make input tensor require gradient
    x = x.unsqueeze(0)
    x.requires_grad_()
    y_hat = model(x)


    for i in range(5):
        cur_y_hat = y_hat[0][i]
        backward = torch.ones_like(cur_y_hat)
        cur_y_hat.backward(backward, retain_graph=True)

        saliency = x.grad
        saliency = torch.abs(saliency)
        saliency = torch.max(saliency, 1)[0]

        plt.imsave(os.path.join(full_directory,
                                os.path.splitext(image_file)[0].replace('/', '_') + '_' + data_loader.CheXPertDataset.observations[i] + '.png'),
                   saliency.squeeze(0), cmap=plt.cm.hot)



def load_image_as_tensor(file_path):
    image = Image.open(file_path)  # PIL image
    return data_loader.preprocess(image), preprocess_for_save(image)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    # use GPU if available
    cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Define the model
    model = net.build_pretrained_densenet(False, False)
    model = model.cuda() if cuda else model

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    calculate_saliency(model, args.image_file, args.model_dir)


