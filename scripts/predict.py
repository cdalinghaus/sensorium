import shutil
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils import get_best_model_path
from src.predictors import Predictor
from src.data import get_mouse_data
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-s", "--split", required=True, type=str)
    parser.add_argument("-m", "--mice", default="all", type=str)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    return parser.parse_args()


def predict_trial(trial_data: dict, predictor: Predictor):
    length = trial_data["length"]
    video = np.load(trial_data["video_path"])[..., :length]
    responses = predictor.predict_trial(video=video)
    return responses


def predict_mouse(experiment: str, split: str, mouse_index: int, device: str):
    mouse = constants.index2mouse[mouse_index]
    print(f"Predict: {experiment=}, {split=}, {mouse_index=}, {mouse=}, {device=}")
    mouse_data = get_mouse_data(mouse=mouse, split=split)
    model_path = get_best_model_path(constants.experiments_dir / experiment / f"mouse_{mouse_index}")
    print("Model path:", model_path)
    predictor = Predictor(model_path=model_path, device=device)
    prediction_dir = constants.predictions_dir / experiment / split / f"mouse_{mouse_index}"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for trial_data in tqdm(mouse_data["trials"]):
        responses = predict_trial(trial_data, predictor)
        np.save(str(prediction_dir / f"{trial_data['trial_id']}.npy"), responses)


def make_submission(experiment: str, split: str):
    prediction_dir = constants.predictions_dir / experiment / split
    data = []
    for mouse_index in constants.mice_indexes:
        mouse = constants.index2mouse[mouse_index]
        neuron_ids = list(range(1, constants.num_neurons[mouse_index] + 1))
        mouse_prediction_dir = prediction_dir / f"mouse_{mouse_index}"
        for prediction_path in mouse_prediction_dir.glob("*.npy"):
            prediction = np.load(prediction_path)[..., constants.submission_skip_first:]
            data.append((mouse, prediction_path.name, prediction.tolist(), neuron_ids))
    submission_df = pd.DataFrame.from_records(
        data,
        columns=['mouse', 'trial_indices', 'predictions', 'neuron_ids']
    )
    submission_path = prediction_dir / f"predictions_file_{split}_track.parquet.brotli"
    submission_df.to_parquet(submission_path, compression='brotli', engine='pyarrow', index=False)
    print(f"Submission saved to '{submission_path}'")


if __name__ == "__main__":
    args = parse_arguments()

    if args.mice == "all":
        mice_indexes = constants.mice_indexes
    else:
        mice_indexes = [int(index) for index in args.mice.split(",")]

    shutil.rmtree(constants.predictions_dir / args.experiment, ignore_errors=True)

    for mouse_index in mice_indexes:
        predict_mouse(args.experiment, args.split, mouse_index, args.device)

    make_submission(args.experiment, args.split)