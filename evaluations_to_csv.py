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
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--dataloader', default="val", help="dataloader to use")


def evaluate(model, dataloader, params):
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

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # filename -> [labels, predictions, num_total_right]
    dict = {}

    # compute metrics over the dataset
    with torch.no_grad():
        for data_batch, file_names_batch, labels_batch in dataloader:

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

            cur_predictions = sigmoid(output_batch)
            cur_predictions_rounded = np.rint(cur_predictions)
            matches = np.sum(labels_batch == cur_predictions_rounded, axis=1)

            for index, file_name in enumerate(file_names_batch):
                labels = labels_batch[index].astype(int).tolist()
                predictions = cur_predictions_rounded[index].astype(int).tolist()
                num_matches = matches[index]
                combined_list = labels + predictions + [num_matches]
                dict[file_name] = combined_list
    columns = data_loader.CheXPertDataset.observations + [o + " Prediction" for o in data_loader.CheXPertDataset.observations] + ["Total Correct"]
    df = pd.DataFrame.from_dict(dict, orient='index', columns=columns)
    return df


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
    dataloaders = data_loader.fetch_dataloader([args.dataloader], args.data_dir, params, include_filenames=True)
    test_dl = dataloaders[args.dataloader]

    logging.info("- done.")

    # Define the model
    model = net.build_pretrained_densenet(False, False)
    model = model.cuda() if params.cuda else model

    loss_fn = nn.BCEWithLogitsLoss()

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    df = evaluate(model, test_dl, params)
    save_path = os.path.join(
        args.model_dir, "predictions_{}.csv".format(args.restore_file))
    df.to_csv(save_path, index=True)
