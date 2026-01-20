import os
import plotly.express as px
import plotly.io as pio
import transformer_lens.utils as utils

def imshow(tensor, filename=None, save_dir=None, **kwargs):
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, filename)
    fig.write_image(filename)


def line(tensor, filename=None, save_dir=None, **kwargs):
    fig = px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    )
    # if filename:
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, filename)
    fig.write_image(filename)
    # fig.show()

def scatter(x, y, xaxis="", yaxis="", caxis="", filename=None, save_dir=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, filename)
    fig.write_image(filename)
