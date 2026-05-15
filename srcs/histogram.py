import seaborn as sns
from utils import parse_argument, prep_df


def main():
    path = parse_argument(description="Displays a histogtam of dataset")
    df = prep_df(path)
    sns_plot = sns.histplot(
        data=df,
        x="Care of Magical Creatures",
        hue="Hogwarts House",
        hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]
        ).set_title(
            "Care of Magical Creatures grades between Hogwarts Houses",
            loc="center"
            )
    fig = sns_plot.get_figure()
    fig.savefig("data_visualization/histogram.png")


if __name__ == "__main__":
    main()