"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/budget_search',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --budget".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)

    # Do evaluations on the CheXpert and CheXPhoto validation sets
    # CheXpert
    data_dir_pert = os.path.join(data_dir, "CheXpert-v1.0-small")
    cmd = "{python} evaluate.py --model_dir={model_dir} --data_dir {data_dir} --restore_file best".format(python=PYTHON,
                                                                                                          model_dir=model_dir,
                                                                                                          data_dir=data_dir_pert)
    print(cmd)
    check_call(cmd, shell=True)

    # CheXphoto
    data_dir_photo = os.path.join(data_dir, "CheXphoto-v1.0")
    cmd = "{python} evaluate.py --model_dir={model_dir} --data_dir {data_dir} --restore_file best".format(python=PYTHON,
                                                                                                          model_dir=model_dir,
                                                                                                          data_dir=data_dir_photo)
    print(cmd)
    check_call(cmd, shell=True)




if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Change the photo counts
    #photo_counts = [0, 2000, 4000, 6000, 8000, 10000]
    photo_counts = [5000]

    for photo_count in photo_counts:
        params.photo_count = photo_count

        # Launch job (name has to be unique)
        job_name = "photo_count_{}".format(photo_count)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
