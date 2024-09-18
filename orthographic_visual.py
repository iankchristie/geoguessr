import numpy as np
import plotly.graph_objects as go
from functions import *
import pdb

center_lat = 40.0150
center_lon = -105.2705
num_points = 10000
sigma = 0.1

x, y, z, lat, lon, phi, theta = fibonacci_sphere(num_points)

lat, lon = cartesian_to_geo(x, y, z)

distances = np.array(
    [haversine(center_lat, center_lon, lat[i], lon[i]) for i in range(num_points)]
)
heat_values = gaussian(distances, sigma)

hover_text = np.array(
    [
        "x: {:.2f}, y: {:.2f}, z: {:.2f}<br>Phi: {:.2f}, theta: {:2f}<br>Distance: {:.2f}<br>Gaussian: {:2f}".format(
            x[i],
            y[i],
            z[i],
            phi[i],
            theta[i],
            distances[i],
            heat_values[i],
        )
        for i in range(num_points)
    ]
)

fig = go.Figure(
    go.Scattergeo(
        lat=lat,
        lon=lon,
        mode="markers",
        marker=dict(
            size=8,
            color=heat_values,
            colorscale="Viridis",
            colorbar=dict(title="Heatmap"),
            showscale=True,
        ),
        text=hover_text,
    )
)
fig.update_traces(marker_size=6, line=dict(color="Red"))
fig.update_geos(projection_type="orthographic")
fig.show()
