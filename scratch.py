import numpy as np
import plotly.graph_objects as go
import pdb


center_lat, center_lon = -30, 120


def spherical_to_geo(theta, phi):
    """
    Convert spherical coordinates (theta, phi) to latitude and longitude.

    Parameters:
    - theta: Polar angle in radians (0 to pi)
    - phi: Azimuthal angle in radians (0 to 2pi)

    Returns:
    - lat: Latitude in degrees (-90 to 90)
    - lon: Longitude in degrees (-180 to 180)
    """
    # Convert theta (polar angle) to latitude
    lat = 90 - np.degrees(theta)

    # Convert phi (azimuthal angle) to longitude
    lon = np.degrees(phi)

    # Normalize longitude to be within [-180, 180] range
    lon[lon > 180] -= 360

    return lat, lon


def geo_to_spherical(lat, lon):
    """
    Convert geographic coordinates (latitude, longitude) to spherical coordinates (phi, theta).

    Parameters:
    - lat: Latitude in degrees (-90 to 90)
    - lon: Longitude in degrees (-180 to 180)

    Returns:
    - theta: Polar angle in radians (0 to pi)
    - phi: Azimuthal angle in radians (0 to 2pi)
    """
    # Convert latitude to polar angle (theta)
    theta = np.radians(90 - lat)  # θ in range [0, π]

    # Convert longitude to azimuthal angle (phi), normalize to [0, 2π]
    phi = np.radians(lon) % (2 * np.pi)  # φ in range [0, 2π]

    return theta, phi


def generate_sphere(radius=1, resolution=100):
    """
    Generate a sphere with Cartesian coordinates and corresponding latitude and longitude.
    """
    # Generate spherical coordinates
    phi = np.linspace(0, 2 * np.pi, resolution)  # Azimuthal angle (longitude)
    theta = np.linspace(0, np.pi, resolution // 2)  # Polar angle (latitude)
    phi, theta = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Convert spherical coordinates (theta, phi) to latitude and longitude
    lat, lon = spherical_to_geo(theta, phi)

    return x, y, z, lat, lon, phi, theta


def geo_to_cartesian(lat, lon, radius=1):
    """
    Convert lat lon to cartesian coordinates.
    """
    lat, lon = map(np.radians, [lat, lon])
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def haversine(lat1, lon1, lat2, lon2, R=1):
    """
    Calculate the great-circle distance between two points on a sphere using the haversine formula.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Compute differences between latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def gaussian(distances, sigma):
    """
    Map distances to a Gaussian distribution.
    """
    return np.exp(-0.5 * (distances**2) / (sigma**2))


x, y, z, lat, lon, phi, theta = generate_sphere()

center_x, center_y, center_z = geo_to_cartesian(center_lat, center_lon)

distances = np.array(
    [
        [
            haversine(lat[i, j], lon[i, j], center_lat, center_lon)
            for j in range(lat.shape[1])
        ]
        for i in range(lat.shape[0])
    ]
)

sigma = 0.35
gaussian_values = gaussian(distances, sigma)

fig = go.Figure()

hover_text = np.array(
    [
        [
            "Lat: {:.2f}°, Lon: {:.2f}°<br>Phi: {:.2f}, theta: {:2f}<br>Distance: {:.2f}<br>Gaussian: {:2f}".format(
                lat[i, j],
                lon[i, j],
                phi[i, j],
                theta[i, j],
                distances[i, j],
                gaussian_values[i, j],
            )
            for j in range(lat.shape[1])
        ]
        for i in range(lat.shape[0])
    ]
)

# Add the surface of the Earth with the Gaussian heat map
fig.add_trace(
    go.Surface(
        x=x,
        y=y,
        z=z,
        hovertext=hover_text,
        surfacecolor=gaussian_values,
        colorscale="Viridis",
        cmin=0,
        cmax=1,
        showscale=True,
        opacity=0.7,  # Set opacity for translucency (0 is fully transparent, 1 is fully opaque)
    )
)

center_theta, center_phi = geo_to_spherical(center_lat, center_lon)

point_hover_text = "Lat: {:.2f}, Lon: {:.2f}<br>Phi: {:.2f}, Theta: {:.2f}".format(
    center_lat, center_lon, center_phi, center_theta
)

# Add the center point as a dot (scatter point) on the sphere
fig.add_trace(
    go.Scatter3d(
        x=[center_x],
        y=[center_y],
        z=[center_z],
        mode="markers",
        marker=dict(size=8, color="red"),
        hovertext=point_hover_text,
    )
)


fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False, showgrid=False, zeroline=False, showticklabels=False
        ),
        yaxis=dict(
            showbackground=False, showgrid=False, zeroline=False, showticklabels=False
        ),
        zaxis=dict(
            showbackground=False, showgrid=False, zeroline=False, showticklabels=False
        ),
        camera=dict(
            eye=dict(x=0.001, y=0.001, z=2),  # Orthographic-like camera view
            projection=dict(type="orthographic"),  # Simulate orthographic projection
        ),
    ),
    title="Gaussian Distribution on Sphere with Center Point",
    width=1000,
    height=800,
)


fig.show()


# # making the plot
# # latitude = -22.2974
# # longitude = -46.6062
# def fibonacci_sphere(num_points):
#     golden_ratio = (1 + np.sqrt(5)) / 2
#     theta = np.arccos(1 - 2 * (np.arange(num_points) + 0.5) / num_points)
#     phi = 2 * np.pi * ((np.arange(num_points) + 0.5) / golden_ratio) % (2 * np.pi)
#     x = np.sin(theta) * np.cos(phi)
#     y = np.sin(theta) * np.sin(phi)
#     z = np.cos(theta)
#     return x, y, z


# # Function to convert Cartesian coordinates (x, y, z) to latitude and longitude
# def cartesian_to_lat_lon(x, y, z):
#     lat = np.degrees(np.arcsin(z))
#     lon = np.degrees(np.arctan2(y, x))
#     return lat, lon


# # Generate points on a Fibonacci sphere
# num_points = 200
# x, y, z = fibonacci_sphere(num_points)

# # Convert to latitude and longitude
# lat, lon = cartesian_to_lat_lon(x, y, z)


# # if you are passing just one lat and lon, put it within "[]"
# fig = go.Figure(go.Scattergeo(lat=lat, lon=lon))
# # editing the marker
# fig.update_traces(marker_size=20, line=dict(color="Red"))
# # this projection_type = 'orthographic is the projection which return 3d globe map'
# fig.update_geos(projection_type="orthographic")
# # layout, exporting html and showing the plot
# fig.update_layout(width=800, height=800, margin={"r": 0, "t": 0, "l": 0, "b": 0})
# # fig.write_html("3d_plot.html")
# fig.show()
