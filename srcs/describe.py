import argparse
import pandas as pd
import numpy as np
from utils import load
from math import sqrt, pow


def count(df: pd.DataFrame) -> list[float]:
    """
    Counts the number of values for each feature in dataframe.
    Excludes NaN values.

    Parameters:
    df (pd.DataFrame): the dataframe.

    Returns:
    count (list[float]): A list of count values for each col.
    """
    num_features = df.shape[1]
    col = df.iloc[:, 0].notna()
    count = []

    for col in range(num_features):
        count.append(float(df.iloc[:, col].notna().sum()))
    return count


def min(df: pd.DataFrame) -> list[float]:
    """
    Returns min value for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.

    Returns:
    min_values (list[float]): A list of min values for each col.
    """
    min_values = df.iloc[0, :].to_list()
    return min_values


def max(df: pd.DataFrame) -> list[float]:
    """
    Returns max value for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.

    Returns:
    max_values (list[float]): A list of max values for each col.
    """
    num_features = df.shape[1]
    max_values = []

    for col in range(num_features):
        serie = df.iloc[:, col].dropna()
        max_values.append(float(serie.iloc[-1]))
    return max_values


def quantile(df: pd.DataFrame, n: list[float], q: float) -> list[float]:
    """
    Returns quartile q for each feature in sorted dataframe. Uses linear type
    of interpolation as per pandas default setting : i + (j - i) * fract.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.
    n (list[float]): the list of counts of values for each col.
    q (float) : the quartile, must be between 0 and 1

    Returns:
    quartile (list[float]): A list of the qth quartile for each col.
    """
    if not 0 <= q <= 1:
        raise ValueError("describe.py: quantile(): ValueError: 'q' parameter must be between [0, 1].")

    num_features = df.shape[1]
    quantile = []

    for col in range(num_features):
        serie = df.iloc[:, col].dropna()
        index = (n[col] - 1) * q
        if int(index) == index:
            quantile.append(float(serie.iloc[int(index)]))
        else:
            fraction = index - int(index)
            left = int(index)
            right = left + 1
            i, j = serie.iloc[left], serie.iloc[right]
            quantile.append(float(i + (j - i) * fraction))
    return quantile


def std(df: pd.DataFrame, n: list[float], mean_val: list[float]) -> list[float]:
    """
    Returns sample standard deviation for each feature in dataframe.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.
    n (list[float]): the list of counts of values for each col.
    mean_val (list[float]) : the list of mean values for each col.

    Returns:
    std_val (list[float]) : a list of each col's standard deviation.
    """
    num_features = df.shape[1]
    std_val = []

    for col in range(num_features):
        serie = df.iloc[:, col].dropna()
        var = sum([pow(x - mean_val[col], 2) for x in serie]) / (n[col] - 1)
        std_val.append(sqrt(var))
    return std_val


def mean(df: pd.DataFrame, n: list[float]) -> list[float]:
    """
    Returns the mean of all values for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the dataframe.
    n (list[float]): the list of counts of values for each col.

    Returns:
    means (list[float]): A list of the mean for each col.
    """
    num_features = df.shape[1]
    means = []

    for col in range(num_features):
        serie = df.iloc[:, col].dropna()
        means.append(float(serie.sum() / n[col]))
    return means


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


def create_describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates description of dataframe similarly to the pandas describe function.
    Shows for each col, excluding NaN values:
        - Count of values
        - Mean of values
        - Sample standard deviation of values
        - Minimum value
        - 25th percentile
        - 50th percentile
        - 75th percentile
        - Maximum value

    Parameters:
    df (pd.DataFrame): the dataframe to describe

    Returns:
    describe_df (pd.DataFrame):
    """
    features = ["Arithmancy", "Astronomy",
                "Herbology", "Defense Against the Dark Arts",
                "Divination", "Ancient Runes",
                "History of Magic", "Transfiguration",
                "Potions", "Care of Magical Creatures",
                "Charms", "Flying"]
    df = df[features]
    df_sorted = df.apply(lambda x: x.sort_values().values)
    count_val = count(df_sorted)
    mean_val = mean(df_sorted, count_val)

    indexes = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    describe_values = [
        count_val,
        mean_val,
        std(df_sorted, count_val, mean_val),
        min(df_sorted),
        quantile(df_sorted, count_val, 0.25),
        quantile(df_sorted, count_val, 0.5),
        quantile(df_sorted, count_val, 0.75),
        max(df_sorted)
        ]
    describe_df = pd.DataFrame(
        index=indexes,
        columns=features,
        data=np.array(describe_values)
        )
    return describe_df


def main():
    try:
        path = __parse_argument()
        df = load(path)
        describe_df = create_describe_dataframe(df)
        print(describe_df)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    main()
