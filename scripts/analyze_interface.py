"""
Analyze SimJEB interface points to find their centroids.
This helps us define a 'template' shape that connects them.
"""
import numpy as np
from sklearn.cluster import DBSCAN
import sys
from pathlib import Path

def analyze_interfaces():
    # Path to simjeb data
    data_path = Path("GINN/simJEB/data/interface_points.npy")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    print(f"Loading {data_path}...")
    pts = np.load(data_path)
    print(f"Loaded {len(pts)} points")
    print(f"Bounds: min={pts.min(0)}, max={pts.max(0)}")

    # Clustering to find distinct interface patch centroids
    # SimJEB typically has 4 bolt holes and 1 pin joint (or 2 pin joints?)
    # epsilon=0.1 seems reasonable for clustering (10mm if scale is 1->1m and units are small? No wait, bounds are ~1.0)
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pts)
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    
    centroids = []
    for i in range(n_clusters):
        cluster_pts = pts[labels == i]
        centroid = cluster_pts.mean(0)
        centroids.append(centroid)
        print(f"Cluster {i}: {len(cluster_pts)} points, centroid={centroid}")
    
    # Calculate connecting lines?
    # Usually one big pin and some bolts
    # Let's see the geometry first
    
    return np.array(centroids)

if __name__ == "__main__":
    analyze_interfaces()
