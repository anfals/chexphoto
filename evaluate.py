"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--dataloader', default="val", help="dataloader to use")


def evaluate(model, loss_fn, dataloader, params, calculate_full_metrics=True, limit_number_iterations=False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
    """

    # check if model was training
    was_training = model.training

    # set model to evaluation mode
    model.eval()

    all_outputs = None
    all_labels = None
    num_batches = 0
    running_loss = 0

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # Update running stats
            num_batches += 1
            running_loss += loss.item()
            if all_outputs is None:
                all_outputs = output_batch
                all_labels = labels_batch
            else:
                all_outputs = np.concatenate([all_outputs, output_batch], axis=0)
                all_labels = np.concatenate([all_labels, labels_batch], axis=0)

            if limit_number_iterations and num_batches > 150:
                break


    if calculate_full_metrics:
        metrics_dict = net.calculate_metrics(all_outputs, all_labels)
    else:
        metrics_dict = {}
    metrics_dict["loss"] = running_loss / num_batches

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_dict.items())
    logging.info("- Eval metrics : " + metrics_string)

    if was_training:
        model.train()
    return metrics_dict


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader([args.dataloader], args.data_dir, params)
    test_dl = dataloaders[args.dataloader]

    logging.info("- done.")

    # Define the model
    model = net.build_pretrained_densenet()
    model = model.cuda() if params.cuda else model

    loss_fn = nn.BCEWithLogitsLossLoss()

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
