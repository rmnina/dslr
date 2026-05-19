import pandas as pd
import argparse


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
