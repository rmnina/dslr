import pandas as pd
import argparse
import numpy as np


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
