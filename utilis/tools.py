#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/22 22:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   tools.py
# @Desc     :

from pandas import DataFrame
from plotly import express
from streamlit import sidebar, header, selectbox, caption, slider
from time import perf_counter
from umap import UMAP


def params_model_getter():
    with sidebar:
        header("Model Parameters")
        options_box: list = ["moka-ai/m3e-base"]
        model: str = selectbox(
            "Embedding Model", options_box, disabled=True,
            help="Select the Embedding Model you want to use"
        )
        caption("The dimensions of the embeddings are 768")

    return model


class Timer(object):
    def __init__(self, description: str, precision: int = 5):
        self._description = description
        self._precision = precision
        self._start = None
        self._end = None
        self._elapsed = None

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        if self._elapsed is None:
            return "Timer not started yet."
        else:
            return f"{self._description} elapsed: {self._elapsed:.{self._precision}f} seconds"


class DimensionsReducer(object):
    def __init__(self, dimensions: int = 3):
        self._dimensions = dimensions

    def fit_transform(self, features: DataFrame, neighbors: int = 15, seed: int = 9527) -> DataFrame:
        """ Display the 3 dimensions chart of scatter """
        # Extract words (keys) and their vectors (values)S
        categories = features.index.tolist()
        categories_name: str = features.columns[0]
        vectors = features.drop(columns=[categories_name]).values

        # Reduce from ND to 3D using UMAP
        reducer = UMAP(n_components=self._dimensions, n_neighbors=neighbors, random_state=seed)
        vectors_3d = reducer.fit_transform(vectors)

        # Convert to DataFrame with correct column names
        df = DataFrame(vectors_3d, columns=["X", "Y", "Z"])

        # Add word labels
        df["category"] = categories

        return df


def params_plotly_getter():
    with sidebar:
        header("Plotly Parameters")
        point_size: int = slider(
            "Point Size", min_value=5, max_value=15, value=10, step=1,
            help="Set the size of the points in the 3D chart"
        )
        caption(f"Point Size: **{point_size}**")

        font_size: int = slider(
            "Font Size", min_value=10, max_value=24, value=12, step=2,
            help="Set the font size of the hover text in the 3D chart"
        )
        caption(f"Font Size: **{font_size}**")

    return point_size, font_size


def sentences_3d(features: DataFrame, point_size: int, font_size: int):
    """ Display the 3 dimensions chart of scatter """
    # Define the columns to be used for plotting
    cols: list = features.columns.tolist()
    categories: list = features["category"].tolist()

    # Define the plotting function
    fig = express.scatter_3d(
        data_frame=features,
        x=cols[0],
        y=cols[1],
        z=cols[2],
        color=categories,  # You can color by the sentence category
        text=categories,  # Display the sentences as hover text
        title="Feature Similarities after Dimension Reduction",
        height=800,
        width=1000  # Set a wider width for better visualization
    )

    # Specific adjustments
    fig.update_traces(marker=dict(size=point_size), textfont=dict(size=font_size))

    # Adjust layout settings for better positioning
    fig.update_layout(
        legend=dict(
            orientation="h",  # Set the legend to be horizontal
            yanchor="bottom",  # Position legend at the bottom
            xanchor="center",  # Center the legend
            y=-0.3,  # Adjust vertical positioning of legend (move further down)
            x=0.5  # Center the legend horizontally
        ),
        scene=dict(
            xaxis=dict(
                showbackground=True,  # Show the background
                backgroundcolor="rgba(0,0,0,0)"  # Set the web color to transparent
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)"
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)"
            )
        ),
    )

    return fig
