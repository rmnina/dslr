import seaborn as sns
from utils import parse_argument, prep_df


def save_scatter_plot(x: str, y: str) -> None:
    """
    Create a scatter plot of two selected dataset features and save it as a PNG image.

    Parameters:
        x (str): Feature selected for dimension x.
        y (str): Feature selected for dimension y.

    Returns:
        None
    """
    path = parse_argument(description="Displays a scatter plot of two selected dataset features")
    df = prep_df(path)
    sns.set_theme(font_scale=0.75)
    sns_plot = sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="Hogwarts House",
        hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]
        ).set_title(
            "Defense Against the Dark Arts vs Astronomy",
            loc="center"
            )
    fig = sns_plot.get_figure()
    fig.savefig("data_visualization/scatter_plot.png")


if __name__ == "__main__":
    save_scatter_plot(x="Defense Against the Dark Arts", y="Astronomy")
