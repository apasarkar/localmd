import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from tqdm import tqdm
import math
import os
import sys
import re

import localmd


def make_pmd_corr_diagnostic_plot(
    standard_correlation_image: np.ndarray,
    autocorr_image: np.ndarray,
    pmd_cov_image: np.ndarray,
    residual_cov_image: np.ndarray,
):
    # Step 2: Create a 2x2 subplot layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Raw Corr",
            "Raw Autocorr",
            "Scaled Cov(UV)",
            "Scaled Cov(Y - UV)",
        ),
        shared_xaxes=True,
        shared_yaxes=True,  # Ensures synchronized zooming and panning
    )

    max_value = np.amax(standard_correlation_image)

    colorscale = "Viridis"
    # Step 3: Add images to the subplots
    fig.add_trace(
        go.Heatmap(
            z=np.array(standard_correlation_image),
            showscale=False,
            colorscale=colorscale,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=np.array(autocorr_image), showscale=False, colorscale=colorscale),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=np.array(pmd_cov_image), showscale=False, colorscale=colorscale),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.array(residual_cov_image), showscale=False, colorscale=colorscale
        ),
        row=2,
        col=2,
    )

    # Step 4: Customize layout (optional)
    fig.update_layout(
        title="Corr Images (PMD Weighted ACF(1) Image)",
        showlegend=False,  # Disable legend
        coloraxis=dict(
            colorscale=colorscale,
            cmin=0,
            cmax=max_value,  # Define the shared colorscale
        ),
        hovermode="closest",  # Hover over to see pixel values
        xaxis=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis1=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis1=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis2=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis2=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis3=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis3=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis4=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis4=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis5=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis5=dict(matches="y1", scaleanchor="x1", scaleratio=1),
    )

    fig.data[0].update(coloraxis="coloraxis")
    fig.data[1].update(coloraxis="coloraxis")
    fig.data[2].update(coloraxis="coloraxis")
    fig.data[3].update(coloraxis="coloraxis")

    return fig


def make_residual_correlation_image(
    original_movie: np.ndarray, pmd_movie: np.ndarray, mode: str = "max"
):
    """
    Simple routine to compute the following covariance image:
    Pixel i = Cov(original_i - pmd_i, original_j - pmd_j) / Sqrt(Var(original_i) * Var(original_j))

    If mode is "max", "j" is the neighboring pixel which maximizes the above value. If mode is "mean", we take the mean
    over all pixels 'j' adjacent to i.
    Args:
        original_movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2).
        pmd_movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2).
        mode (str): Either max or mean. Indicates how we make the corr image
    Returns:
        corr img (np.ndarray): Shape (fov dim 1, fov dim 2).
    """

    @partial(jit)
    def local_routine_resid_corr(original_curr, pmd_curr, original_rel, pmd_rel):
        curr_trace = original_curr - pmd_curr
        rel_trace = original_rel - pmd_rel
        cov = jnp.cov(curr_trace, rel_trace)[0, 1]
        cov_raw = jnp.sqrt(jnp.var(original_curr) * jnp.var(original_rel))
        return cov / cov_raw

    T, d1, d2 = original_movie.shape

    counts = np.zeros((d1, d2))
    net_corr = np.zeros((d1, d2))

    for k in tqdm(range(d1)):
        for j in range(d2):
            original_curr = original_movie[:, k, j]
            pmd_curr = pmd_movie[:, k, j]
            for coord1 in range(k - 1, k + 2):
                for coord2 in range(j - 1, j + 2):
                    if (
                        coord1 >= 0
                        and coord1 < counts.shape[0]
                        and coord2 >= 0
                        and coord2 < counts.shape[1]
                    ):
                        if not (coord1 == k and coord2 == j):
                            original_rel = original_movie[:, coord1, coord2]
                            pmd_rel = pmd_movie[:, coord1, coord2]
                            cov = local_routine_resid_corr(
                                original_curr, pmd_curr, original_rel, pmd_rel
                            )

                            if mode == "mean":
                                net_corr[k, j] += cov
                            elif mode == "max":
                                net_corr[k, j] = max(cov, net_corr[k, j])
                            else:
                                raise ValueError(f"mode {mode} not supported")
                            counts[k, j] += 1
    if mode == "mean":
        net_corr = net_corr / counts

    return net_corr


def make_pmd_correlation_image(
    original_movie: np.ndarray, pmd_movie: np.ndarray, mode: str = "max"
):
    """
    Computes a scaled covariance image (normalized by the variances of pixels of raw data, so this can be directly
    compared with the residual corr image and the raw data correlation image)

    Pixel i = Cov(pmd_i, pmd_j) / Sqrt(Var(original_i) * Var(original_j))

    If mode is "max", "j" is the neighboring pixel which maximizes the above value. If mode is "mean", we take the mean
    over all pixels 'j' adjacent to i.
    Args:
        original_movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2).
        pmd_movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2).
        mode (str): Either max or mean. Indicates how we make the corr image
    Returns:
        corr img (np.ndarray): Shape (fov dim 1, fov dim 2).
    """

    @partial(jit)
    def local_routine_corr(original_curr, pmd_curr, original_rel, pmd_rel):
        curr_trace = pmd_curr
        rel_trace = pmd_rel
        cov = jnp.cov(curr_trace, rel_trace)[0, 1]
        cov_raw = jnp.sqrt(jnp.var(original_curr) * jnp.var(original_rel))
        return cov / cov_raw

    T, d1, d2 = original_movie.shape

    counts = np.zeros((d1, d2))
    net_corr = np.zeros((d1, d2))

    for k in tqdm(range(d1)):
        for j in range(d2):
            original_curr = original_movie[:, k, j]
            pmd_curr = pmd_movie[:, k, j]
            for coord1 in range(k - 1, k + 2):
                for coord2 in range(j - 1, j + 2):
                    if (
                        coord1 >= 0
                        and coord1 < counts.shape[0]
                        and coord2 >= 0
                        and coord2 < counts.shape[1]
                    ):
                        if not (coord1 == k and coord2 == j):
                            original_rel = original_movie[:, coord1, coord2]
                            pmd_rel = pmd_movie[:, coord1, coord2]
                            cov = local_routine_corr(
                                original_curr, pmd_curr, original_rel, pmd_rel
                            )

                            if mode == "mean":
                                net_corr[k, j] += cov
                            elif mode == "max":
                                net_corr[k, j] = max(cov, net_corr[k, j])
                            else:
                                raise ValueError(f"mode {mode} not supported")
                            counts[k, j] += 1
    if mode == "mean":
        net_corr = net_corr / counts

    return net_corr


def make_correlation_image(movie: np.ndarray, mode: str = "max"):
    """
    Args:
        movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2).
        mode (str): Either max or mean. Indicates how we make the corr image

    Returns:
        corr img (np.ndarray): Shape (fov dim 1, fov dim 2)
    """

    @partial(jit)
    def compute_correlation(t1, t2):
        t1_centered = t1 - jnp.mean(t1, axis=0, keepdims=True)
        t1_norm = t1_centered / (jnp.linalg.norm(t1_centered, axis=0, keepdims=True))
        t2_centered = t2 - np.mean(t2, axis=0, keepdims=True)
        t2_norm = t2_centered / (jnp.linalg.norm(t2_centered, axis=0, keepdims=True))
        return jnp.sum(jnp.multiply(t1_norm, t2_norm), axis=0)

    movie_centered = movie
    counts = np.zeros(movie_centered.shape[1:])
    net_corr = np.zeros(movie_centered.shape[1:])

    for k in tqdm(range(movie_centered.shape[1])):
        for j in range(movie_centered.shape[2]):
            curr_trace = movie_centered[:, k, j]
            for coord1 in range(k - 1, k + 2):
                for coord2 in range(j - 1, j + 2):
                    if (
                        coord1 >= 0
                        and coord1 < counts.shape[0]
                        and coord2 >= 0
                        and coord2 < counts.shape[1]
                    ):
                        if not (coord1 == k and coord2 == j):
                            rel_trace = movie_centered[:, coord1, coord2]
                            corr = compute_correlation(curr_trace, rel_trace)
                            if mode == "mean":
                                net_corr[k, j] += corr
                            elif mode == "max":
                                net_corr[k, j] = max(corr, net_corr[k, j])
                            else:
                                raise ValueError(f"mode {mode} not supported")
                            counts[k, j] += 1
    if mode == "mean":
        net_corr = net_corr / counts
    return net_corr


def make_autocorrelation_image(movie: np.ndarray, lag: int = 1):
    """
    Args:
        movie (np.ndarray): Shape (frames, fov dim 1, fov dim 2)
        lag (int): The lag for which the autocorrelation is computed
    Returns:
        corr_img (np.ndarray): Shape (fov dim 1, fov dim 2)
    """

    @partial(jit)
    def compute_correlation(t1, t2):
        t1_centered = t1 - jnp.mean(t1, axis=0, keepdims=True)
        t1_norm = t1_centered / (jnp.linalg.norm(t1_centered, axis=0, keepdims=True))
        t2_centered = t2 - np.mean(t2, axis=0, keepdims=True)
        t2_norm = t2_centered / (jnp.linalg.norm(t2_centered, axis=0, keepdims=True))
        return jnp.sum(jnp.multiply(t1_norm, t2_norm), axis=0)

    corr_img = np.zeros(movie.shape[1:])

    batch_size = 100
    j_batches = math.ceil(movie.shape[2] / batch_size)
    for k in tqdm(range(movie.shape[1])):
        for j in range(j_batches):
            start = j * batch_size
            end = min(start + batch_size, movie.shape[2])
            curr_trace = movie[:, k, start:end]
            corr_img[k, start:end] = np.array(
                compute_correlation(curr_trace[lag:], curr_trace[: -1 * lag])
            )

    return corr_img


def make_pmd_component_graph(
    spatial: np.ndarray,
    mean_img: np.ndarray,
    var_img: np.ndarray,
    trace: np.ndarray,
    index: int,
    title: str,
):
    # Create a Plotly subplot
    fig = sp.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Mean",
            "Var Img",
            f"Spatial Comp {index}",
            f"Temporal Comp {index}",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"colspan": 3}, None, None],
        ],
    )

    # Adding heatmaps with synchronized zooming and custom axes (using p1 and p2)
    fig.add_trace(
        go.Heatmap(z=mean_img, showscale=False, colorscale="Viridis"), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=var_img, showscale=False, colorscale="Viridis"), row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=spatial, showscale=False, colorscale="Viridis"), row=1, col=3
    )

    # Temporal Trace
    fig.add_trace(go.Scatter(y=trace, mode="lines", name="Signal"), row=2, col=1)

    # Update the layout to adjust titles and color axes
    fig.update_layout(
        title=title,
        height=800,
        xaxis=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis1=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis1=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis2=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis2=dict(matches="y1", scaleanchor="x1", scaleratio=1),
        xaxis3=dict(matches="x1", scaleanchor="y1", scaleratio=1),
        yaxis3=dict(matches="y1", scaleanchor="x1", scaleratio=1),
    )

    return fig


def plot_pmd_components(
    pmd_movie: localmd.PMDArray, folder: str, filename_prefix: str = "Component"
):
    if not os.path.exists(folder):
        raise ValueError(
            f"folder {folder} does not exist; please make it then run this code"
        )

    u, r, s, v = pmd_movie.u, pmd_movie.r, pmd_movie.s, pmd_movie.v
    data_order = pmd_movie.order
    T, d1, d2 = pmd_movie.shape
    var_img = pmd_movie.var_img
    mean_img = pmd_movie.mean_img

    for i in range(r.shape[1]):
        current_ur = u.dot(r[:, i])
        current_ur = current_ur.reshape((d1, d2), order=data_order)
        current_temporal_trace = v[i, :]
        explained_variances = np.square(s[i]) / np.sum(np.square(s))

        curr_title = f"Comp {i}, Var explained {explained_variances:3f}"
        curr_name = f"{filename_prefix}_{i}.html"
        current_figure = make_pmd_component_graph(
            current_ur, mean_img, var_img, current_temporal_trace, i + 1, curr_title
        )

        current_figure.write_html(os.path.join(folder, curr_name))


def construct_index(
    folder: str, file_prefix: str = "neuron", index_name: str = "index.html"
):
    def numerical_sort(file):
        match = re.search(rf"{file_prefix}[_\s]*(\d+)", file)
        return (
            int(match.group(1)) if match else float("inf")
        )  # Default to large number if no match

    index_file = os.path.join(folder, index_name)

    # List all HTML files in the directory
    html_files = [f for f in os.listdir(folder) if f.endswith(".html")]
    html_files.sort(key=numerical_sort)  # Sort files by numerical order

    # Create the index.html file
    with open(index_file, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write('<html lang="en">\n')
        f.write("<head>\n")
        f.write('    <meta charset="UTF-8">\n')
        f.write(
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        )
        f.write("    <title>Navigation Index</title>\n")
        f.write("    <style>\n")
        f.write(
            "        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }\n"
        )
        f.write("        .content { margin-bottom: 20px; }\n")
        f.write("        .nav-buttons { margin-top: 20px; }\n")
        f.write(
            "        button { padding: 10px 20px; margin: 5px; font-size: 16px; }\n"
        )
        f.write("    </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("    <h1>Navigate Through Files</h1>\n")
        f.write('    <div class="content" id="content">\n')
        f.write(
            '        <iframe src="" style="width:100%; height:600px; border:none;"></iframe>\n'
        )
        f.write("    </div>\n")
        f.write('    <div class="nav-buttons">\n')
        f.write(
            '        <button id="prev-btn" onclick="navigate(-1)">Previous</button>\n'
        )
        f.write('        <button id="next-btn" onclick="navigate(1)">Next</button>\n')
        f.write("    </div>\n")
        f.write("\n")
        f.write("    <script>\n")
        f.write("        const files = [\n")
        for file in html_files:
            f.write(f"            '{file}',\n")
        f.write("        ];\n")
        f.write("        let currentIndex = 0;\n")
        f.write("        const contentDiv = document.getElementById('content');\n")
        f.write("        const prevBtn = document.getElementById('prev-btn');\n")
        f.write("        const nextBtn = document.getElementById('next-btn');\n")
        f.write("\n")
        f.write("        function loadContent() {\n")
        f.write(
            '            contentDiv.innerHTML = `<iframe src="${files[currentIndex]}" style="width:100%; height:600px; border:none;"></iframe>`;\n'
        )
        f.write("            prevBtn.disabled = currentIndex === 0;\n")
        f.write("            nextBtn.disabled = currentIndex === files.length - 1;\n")
        f.write("        }\n")
        f.write("\n")
        f.write("        function navigate(direction) {\n")
        f.write("            currentIndex += direction;\n")
        f.write("            if (currentIndex >= 0 && currentIndex < files.length) {\n")
        f.write("                loadContent();\n")
        f.write("            }\n")
        f.write("        }\n")
        f.write("\n")
        f.write("        // Initial load\n")
        f.write("        loadContent();\n")
        f.write("    </script>\n")
        f.write("</body>\n")
        f.write("</html>\n")

    print(f'Index file "{index_file}" created successfully.')
