import numpy as np


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


def cartesian_to_geo(x, y, z):
    """
    Convert lat lon points to cartesian coordinates.
    """
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def haversine(lat1, lon1, lat2, lon2, R=1):
    """
    Calculate the great-circle distance between two points on a sphere using the haversine formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def gaussian(distances, sigma):
    """
    Map distances to a Gaussian distribution.
    """
    return np.exp(-0.5 * (distances**2) / (sigma**2))


def fibonacci_sphere(num_points):
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = np.arccos(1 - 2 * (np.arange(num_points) + 0.5) / num_points)
    phi = 2 * np.pi * ((np.arange(num_points) + 0.5) / golden_ratio) % (2 * np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    lat, lon = spherical_to_geo(theta, phi)
    return x, y, z, lat, lon, phi, theta
