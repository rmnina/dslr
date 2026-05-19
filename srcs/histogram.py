import seaborn as sns
from utils import parse_argument, prep_df


def save_histogram(x: str) -> None:
    """
    Create a histogram of one selected dataset feature and save it as a PNG image.

    Parameters:
        x (str): Feature selected for histogram.

    Returns:
        None
    """
    path = parse_argument(description="Displays a histogram of one selected dataset feature")
    df = prep_df(path)
    sns_plot = sns.histplot(
        data=df,
        x=x,
        hue="Hogwarts House",
        hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]
        ).set_title(
            "Care of Magical Creatures grades between Hogwarts Houses",
            loc="center"
            )
    fig = sns_plot.get_figure()
    fig.savefig("data_visualization/histogram.png")


if __name__ == "__main__":
    save_histogram(x="Care of Magical Creatures")
