#TODO: count, mean, std, min, 25th, 50th, 75th, max 

import argparse
import pandas as pd
from utils import load

def count(df: pd.DataFrame):
    pass

def __parse_argument() -> str:
    parser = argparse.ArgumentParser(description="Calculates statistical values on csv file")
    parser.add_argument("path", type=str, help="Path of the csv file")
    parser = parser.parse_args()
    return parser.path

def main():
    path = __parse_argument()
    df = load(path)
    print(df.describe())

if __name__ == "__main__":
    main()

