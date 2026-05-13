import argparse
import pandas as pd
from utils import load


def count(df: pd.DataFrame) -> list[int]:
    """
    Counts the number of values for each feature in dataframe.
    Excludes NaN values.

    Parameters:
    df (pd.DataFrame): the dataset.

    Returns:
    list: A list of count values for each row.
    """
    num_features = df.shape[1]
    row = df.iloc[:, 0].notna()
    count = []

    for row in range(num_features):
        count.append(int(df.iloc[:, row].notna().sum()))
    return count


def __parse_argument() -> str:
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
    count(df)
    # print(df.describe().to_csv())


if __name__ == "__main__":
    main()
