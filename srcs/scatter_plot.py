import seaborn as sns
from utils import parse_argument, prep_df


def main():
    path = parse_argument(description="Displays a scatter plot of dataset")
    df = prep_df(path)
    sns.set_theme(font_scale=0.75)
    sns_plot = sns.scatterplot(
        data=df,
        x="Care of Magical Creatures",
        y="Arithmancy",
        hue="Hogwarts House",
        hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]
        ).set_title(
            "Arithmancy vs Care of Magical Creatures grades between Hogwarts Houses",
            loc="center"
            )
    fig = sns_plot.get_figure()
    fig.savefig("data_visualization/scatter_plot.png")


if __name__ == "__main__":
    main()