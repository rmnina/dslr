import pandas as pd
import argparse
import numpy as np
import random

def ft_normalize(X: np.ndarray, min_train: np.ndarray, max_train: np.ndarray) -> np.ndarray:
    """
    Normalizes the values contained in the array passed as an argument according to the
    minimum and maximum values of the training dataset.
    Adds 1e-20 to the result in order to avoid potential divisions by 0.
    
    Args:
        X (np.ndarray): Array in which the values will be normalized;
        min_train (float): Minimum value in the training array.
        max_train (float): Maximum value in the training array.

    Returns:
        (np.ndarray): The resulting normalized array.

    """
    return (X - min_train) / ((max_train - min_train) + 1e-20) 


def prep_df(path: str) -> pd.DataFrame:
    """
    Return a dataframe with features selected in the 'features' variable.

    Parameters:
        path (str): Path of the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The dataframe returned.
    """
    df = load(path)
    features = [
        "Hogwarts House",
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying"
        ]
    df = df[features]
    return df


def parse_argument(description: str) -> str:
    """
    Parses arguments to expect dataset path as first positional
    argument.

    Parameters:
        None

    Returns:
        parser.path (str): The string value of argument to process OR an error if
        no or too many arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("path", type=str, help="Path of the csv file")
    parser = parser.parse_args()
    return parser.path


def load(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the specified path into a Pandas DataFrame.

    Parameters:
        path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded DataFrame if successful, None if an error occurs.
    """
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path)
        print(f"Loading dataset '{path}' of dimensions {df.shape}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{__name__}: {type(error).__name__}: {error}")
        return None


def split_dataset(X: np.ndarray, y: np.ndarray, eval_size: float, seed: int) -> tuple[np.ndarray]:
    """
    Splits each of the two original datasets into, for each original:
        - One training dataset
        - One evaluating dataset
    The split is operated according to the _eval_size_ proportion.
    Example:
        _eval_size_ = 0.2 -> The resulting evaluating datasets will consist
        in 20% of the original datasets. Therefore, the resulting training datasets
        will consist in the remaining 80% of the original datasets.
    The split is made at random to ensure representativity in each resulting dataset.

    Args:
        X (np.ndarray): The original dataset to split, containing the labels.
        y (np.ndarray): The original dataset to split containing the classes.
        eval_size (float): The proportion of the resulting eval datasets, relatively to
        the original datasets.
        seed (int): Seed passed to randomization function.

    Returns:
        X_train, y_train, X_eval, y_eval (tuple[np.ndarray]) : The training and evaluating
        datasets retrieved from the two original datasets.
    """
    if not 0 < eval_size < 1:
        raise Exception("split_dataset(): EVAL_DATASET_SIZE must be between 0 and 1.")
    m = X.shape[0]
    random.seed(seed)
    eval_idx = random.sample(range(m), int(m * eval_size))
    train_idx = list(set((range(m))) - set(eval_idx))

    X_train = X.iloc[train_idx, :]
    y_train = y.iloc[train_idx]
    X_eval = X.iloc[eval_idx, :]
    y_eval = y.iloc[eval_idx]

    return X_train, X_eval, y_train, y_eval

