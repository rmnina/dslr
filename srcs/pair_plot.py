import seaborn as sns
from utils import parse_argument, prep_df


def main():
    path = parse_argument(description="Displays a pair plot of dataset")
    df = prep_df(path)
    sns_plot = sns.pairplot(
        data=df,
        hue="Hogwarts House",
        hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"],
        diag_kind="hist",
        dropna=True
        )
    sns_plot.savefig("data_visualization/pair_plot.png")


if __name__ == "__main__":
    main()