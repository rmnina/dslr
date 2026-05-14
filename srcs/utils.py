import pandas as pd
import argparse


def parse_argument(description: str) -> str:
    """
    Parses arguments to expect dataset path as first positional
    argument.

    Parameters:
    none

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
    - path (str): The path to the CSV file to be loaded.

    Returns:
    - The loaded DataFrame if successful, None if an error occurs.
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
