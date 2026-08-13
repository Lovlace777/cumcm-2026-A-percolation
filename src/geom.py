"""Geometry primitives: segment-segment and segment-plane distances."""
import numpy as np

def segment_segment_distance(p1, q1, p2, q2, eps=1e-14):
    """Shortest distance between two finite 3D segments p1-q1 and p2-q2.
    Robust Lumelsky / Ericson implementation.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom > eps:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    closest = (p1 + s * d1) - (p2 + t * d2)
    return float(np.linalg.norm(closest))


def segment_plane_x_distance(p, q, x0):
    """Min distance from finite segment p-q to the plane x = x0."""
    # x(t) = p_x + t (q_x - p_x), t in [0,1]
    # distance |x(t) - x0|
    xs = np.array([p[0], q[0]], dtype=float)
    # if the segment straddles the plane, distance is 0
    if (xs[0] - x0) * (xs[1] - x0) <= 0:
        return 0.0
    return float(min(abs(xs[0] - x0), abs(xs[1] - x0)))


def point_segment_distance(c, p, q, eps=1e-14):
    d = q - p
    L2 = np.dot(d, d)
    if L2 <= eps:
        return float(np.linalg.norm(c - p))
    t = np.clip(np.dot(c - p, d) / L2, 0.0, 1.0)
    return float(np.linalg.norm(c - (p + t * d)))
