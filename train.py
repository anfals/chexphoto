"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
import model.data_loader_mixed as data_loader_mixed
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--mixed',  action='store_true', help="Whether this is a mixed training experiment")
parser.add_argument('--freeze',  action='store_true', help="Whether to freeze layers from a restored model")
parser.add_argument('--moco',  action='store_true', help="Whether to initialize the model using Facebook moco pretrained model")


def train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        scheduler: (torch.optim) scheduler used to decay the learning rate if validation loss not improving
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    n_iterations_no_change = 0
    best_validation_loss = float('inf')
    early_stop_reached = False

    # Use tqdm for progress bar
    with tqdm(total=len(train_dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_dataloader):
            print(f'Starting iteration {i}')

            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = net.calculate_metrics(output_batch, labels_batch)
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            if params.early_stopping and i % params.validation_steps == 0 and i > 0:
                # Verify loss is improving on the validation set, else early stop
                cur_val_loss = evaluate(model, loss_fn, val_dataloader, params, calculate_full_metrics=False, limit_number_iterations=True)['loss']
                if cur_val_loss > best_validation_loss - params.tolerance:
                    n_iterations_no_change += 1
                    if n_iterations_no_change >= params.n_iterations_no_change:
                        early_stop_reached = True
                        logging.info("Patience hit. Early stopping")
                        break
                    logging.info(f'Validation scores did not improve. Patience {n_iterations_no_change} hit. Decaying learning rate')
                    scheduler.step()
                else:
                    best_validation_loss = cur_val_loss
                    logging.info(f'Validation scores improved. Loss is {cur_val_loss}')
                    n_iterations_no_change = 0

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return early_stop_reached


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir,
                       restore_file=None, freeze=False):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        freeze: (boolean) if restore_file, whether to freeze all but the last layer
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

        if freeze:
            print("Freezing restore file parameters")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

    best_auc_average = 0.0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    epoch_patience = 0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        early_stop_reached = train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, params)

        val_auc_average = val_metrics['AUC Average']
        is_best = val_auc_average >= best_auc_average

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_auc_average = val_auc_average
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        elif params.epoch_early_stopping:
            epoch_patience += 1
            scheduler.step()
            logging.info(f'Validation score did not improve after epoch. Decaying learning rate')
            if epoch_patience >= params.n_iterations_no_change:
                early_stop_reached = True
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        if early_stop_reached:
            logging.info(f'Early stopping on Epoch {epoch}')
            break


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    print(f"Cuda is available {params.cuda}")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    if args.mixed:
        dataloaders = data_loader_mixed.fetch_dataloader(
            ['train', 'val'], args.data_dir, params)
    else:
        dataloaders = data_loader.fetch_dataloader(
            ['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.build_pretrained_densenet(params.cuda, args.moco)
    model = model.cuda() if params.cuda else model
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = nn.BCEWithLogitsLoss()
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, params, args.model_dir, args.restore_file, args.freeze)
