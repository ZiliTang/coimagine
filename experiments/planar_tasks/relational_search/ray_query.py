"""Compatibility import; production consumers use random_start.lidar_service."""
from coimagine.random_start.ray_query import nearest_hits

__all__ = ["nearest_hits"]
