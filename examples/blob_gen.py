import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def generate_points(center, base_radius, num_points=7):
    """Generate random points in a circle to garantee a closed contour"""
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = base_radius + np.random.uniform(-0.4 * base_radius, 0.4 * base_radius, num_points)
    
    x = center[0] + radii * np.cos(theta)
    y = center[1] + radii * np.sin(theta)
    return np.column_stack([x, y]).astype(np.int32)

def smooth_contour(points, resolution=100):
    """Create a closed B-Spline from a set of points"""
    closed_points = np.vstack([points, points[0]])
    tck, u = splprep(closed_points.T, s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), resolution)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack([x_new, y_new]).astype(np.int32)


N = 4
width, height = 600, 600
mask = np.zeros((height, width), dtype=np.uint8)

for _ in range(N):
    center_x = np.random.randint(100, width - 100)
    center_y = np.random.randint(100, height - 100)
    radio = np.random.randint(40, 100)
    
    pts = generate_points((center_x, center_y), radio)
    curve = smooth_contour(pts)
    cv2.fillPoly(mask, [curve], 255)

cv2.imwrite('mask_example2.png', mask)
