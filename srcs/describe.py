import argparse
import pandas as pd
from utils import load


def count(df: pd.DataFrame) -> list[int]:
    """
    Counts the number of values for each feature in dataframe.
    Excludes NaN values.

    Parameters:
    df (pd.DataFrame): the dataframe.

    Returns:
    count (list[int]): A list of count values for each row.
    """
    num_features = df.shape[1]
    row = df.iloc[:, 0].notna()
    count = []

    for row in range(num_features):
        count.append(int(df.iloc[:, row].notna().sum()))
    return count


def min(df: pd.DataFrame) -> list[float]:
    """
    Returns min value for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.

    Returns:
    min_values (list[float]): A list of min values for each row.
    """
    min_values = df.iloc[0, :].to_list()
    return min_values


def max(df: pd.DataFrame) -> list[float]:
    """
    Returns max value for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.

    Returns:
    max_values (list[float]): A list of max values for each row.
    """
    num_features = df.shape[1]
    max_values = []

    for row in range(num_features):
        serie = pd.Series(df.iloc[:, row]).dropna()
        max_values.append(float(serie.iloc[-1]))
    return max_values


def __parse_argument() -> str:
    """
    Parses arguments to expect dataset path as first positional
    argument.

    Parameters:
    none

    Returns:
    parser.path (str): The string value of argument to process OR an error if
    no or too many arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculates statistical values on csv file"
        )
    parser.add_argument("path", type=str, help="Path of the csv file")
    parser = parser.parse_args()
    return parser.path


def main():
    path = __parse_argument()
    df = load(path)
    features = ["Arithmancy", "Astronomy", "Herbology",
                "Defense Against the Dark Arts", "Divination", "Ancient Runes",
                "History of Magic", "Transfiguration", "Potions",
                "Care of Magical Creatures", "Charms", "Flying"]
    df = df[features]
    df_sorted = df.apply(lambda x: x.sort_values().values)
    count(df)
    min(df_sorted)
    max(df_sorted)
    # print(df.describe())


if __name__ == "__main__":
    main()
