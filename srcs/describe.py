import argparse
import pandas as pd
from utils import load
from math import sqrt


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


def quantile(df: pd.DataFrame, n: list[int], q: float) -> list[float]:
    """
    Returns quartile q each feature in sorted dataframe. Uses linear type
    of interpolation as per pandas default setting : i + (j - i) * fract.

    Parameters:
    df (pd.DataFrame): the sorted dataframe.
    n (list[int]): the list of counts of values for each row.
    q (float) : the quartile, must be between 0 and 1

    Returns:
    quartile (list[float]): A list of the qth quartile for each row.
    """
    if not 0 <= q <= 1:
        raise ValueError("describe.py: quantile(): ValueError: 'q' parameter must be between [0, 1].")

    num_features = df.shape[1]
    quantile = []

    for row in range(num_features):
        serie = pd.Series(df.iloc[:, row]).dropna()
        index = (n[row] - 1) * q
        if int(index) == index:
            quantile.append(float(serie.iloc[int(index)]))
        else:
            fraction = index - int(index)
            left = int(index)
            right = left + 1
            i, j = serie.iloc[left], serie.iloc[right]
            quantile.append(float(i + (j - i) * fraction))
    return quantile


def mean(df: pd.DataFrame, n: list[int]) -> list[float]:
    """
    Returns the mean of all values for each feature in sorted dataframe.

    Parameters:
    df (pd.DataFrame): the dataframe.
    n (list[int]): the list of counts of values for each row.

    Returns:
    means(list[float]): A list of the mean for each row.
    """
    num_features = df.shape[1]
    means = []
    for row in range(num_features):
        serie = pd.Series(df.iloc[:, row]).dropna()
        means.append(float(serie.sum() / n[row]))
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


def main():
    try:
        path = __parse_argument()
        df = load(path)
        features = ["Arithmancy", "Astronomy",
                    "Herbology", "Defense Against the Dark Arts",
                    "Divination", "Ancient Runes",
                    "History of Magic", "Transfiguration",
                    "Potions", "Care of Magical Creatures",
                    "Charms", "Flying"]
        df = df[features]
        describe = df.describe()

        df_sorted = df.apply(lambda x: x.sort_values().values)
        count_values = count(df)
        min_val = min(df_sorted)
        max_val = max(df_sorted)
        mean_val = mean(df, count_values)
        quant25 = quantile(df_sorted, count_values, -0.25)
        quant50 = quantile(df_sorted, count_values, 0.5)
        quant75 = quantile(df_sorted, count_values, 0.75)
        diff = list(set(describe.loc["mean", :]) - set(mean_val))
        print(diff)
    except Exception as error:
        print(error)
    # print(df.describe())


if __name__ == "__main__":
    main()
