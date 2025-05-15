import numpy as np

def make_obstacles():
    """
    Fungsi untuk membuat dan mengembalikan koordinat hambatan (obstacles)
    beserta dengan radius hambatannya.
    """
    # Koordinat dari obstacles
    obstacles = np.array([
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 2],
        [5, 7, 6]
    ])
    
    # Radius dari masing-masing obstacles
    obstacle_radii = np.array([1, 2, 2.4, 3])  # Sesuaikan radius sesuai keinginan

    return obstacles, obstacle_radii