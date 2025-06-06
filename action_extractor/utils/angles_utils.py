import numpy as np
import math

def axisangle2quat(axis_angle):
    """
    Converts an axis-angle vector [rx, ry, rz] into a quaternion [x, y, z, w].
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        # nearly zero rotation
        return np.array([0, 0, 0, 1], dtype=np.float32)
    axis = axis_angle / angle
    half = angle * 0.5
    return np.concatenate([
        axis * np.sin(half),
        [np.cos(half)]
    ]).astype(np.float32)


def quat2axisangle(quat):
    """
    Convert quaternion [x, y, z, w] to axis-angle [rx, ry, rz].
    """
    w = quat[3]
    # clamp w
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    angle = 2.0 * math.acos(w)
    den = math.sqrt(1.0 - w * w)
    if den < 1e-12:
        return np.zeros(3, dtype=np.float32)

    axis = quat[:3] / den
    return axis * angle

def quat2axisangle_wxyz(quat):
    """
    Convert quaternion [w, x, y, z] to axis-angle [rx, ry, rz].
    """
    w = quat[0]
    # Clamp w to the valid range [-1, 1]
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0

    angle = 2.0 * math.acos(w)
    den = math.sqrt(1.0 - w * w)
    if den < 1e-12:
        return np.zeros(3, dtype=np.float32)

    # Use the vector part which is now at indices 1:4
    axis = quat[1:4] / den
    return axis * angle


def quat_multiply(q1, q0):
    """
    Multiply two quaternions q1 * q0 in xyzw form => xyzw
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
    ], dtype=np.float32)


def quat_inv(q):
    """
    Inverse of unit quaternion [x, y, z, w] is the conjugate => [-x, -y, -z, w].
    """
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

def rotation_matrix_to_angle_axis(R):
    """
    Convert a 3x3 rotation matrix R into its angle-axis representation
    (a 3D vector whose direction is the rotation axis and magnitude is the rotation angle).
    """
    # Numerical stability: clamp values for arccos
    trace_val = np.trace(R)
    theta = np.arccos(
        np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    )
    
    # If angle is very small, approximate as zero rotation.
    if np.isclose(theta, 0.0):
        return np.zeros(3)
    
    # Compute rotation axis using the classic formula
    # axis = (1/(2*sin(theta))) * [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / (2.0 * np.sin(theta))
    
    # Angle-axis form is axis * angle
    angle_axis = axis * theta
    return angle_axis

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix R into a quaternion [x, y, z, w].
    Assumes R is a proper rotation matrix (orthonormal, det=1).
    """
    M = np.asarray(R, dtype=np.float32)
    trace = M[0,0] + M[1,1] + M[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (M[2,1] - M[1,2]) * s
        y = (M[0,2] - M[2,0]) * s
        z = (M[1,0] - M[0,1]) * s
    else:
        # Find the major diagonal element
        if M[0,0] > M[1,1] and M[0,0] > M[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[0,0] - M[1,1] - M[2,2]))
            w = (M[2,1] - M[1,2]) / s
            x = 0.25 * s
            y = (M[0,1] + M[1,0]) / s
            z = (M[0,2] + M[2,0]) / s
        elif M[1,1] > M[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[1,1] - M[0,0] - M[2,2]))
            w = (M[0,2] - M[2,0]) / s
            x = (M[0,1] + M[1,0]) / s
            y = 0.25 * s
            z = (M[1,2] + M[2,1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + M[2,2] - M[0,0] - M[1,1]))
            w = (M[1,0] - M[0,1]) / s
            x = (M[0,2] + M[2,0]) / s
            y = (M[1,2] + M[2,1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    return quat_normalize(q)

def quaternion_norm(q):
    """
    Compute the Euclidean norm of a quaternion.
    """
    return np.sqrt(np.dot(q, q))

def quaternion_normalize(q):
    """
    Normalize a quaternion to make it a unit quaternion.
    """
    norm = quaternion_norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero quaternion.")
    return q / norm

def quat_exp(q):
    """
    Computes the exponential of a pure quaternion (with zero scalar part) given in [x, y, z, w] order.
    The function assumes the input q represents an axis-angle vector in its vector part (q[:3])
    and q[3] is 0.
    
    Parameters:
        q (np.array): A quaternion [x, y, z, w] with w assumed to be 0.
    
    Returns:
        np.array: A unit quaternion in [x, y, z, w] order representing the rotation.
    """
    # Extract the vector part
    v = q[:3]
    theta = np.linalg.norm(v)
    
    # When theta is near zero, return the identity quaternion to avoid division by zero.
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    else:
        # Compute the scaled vector part and the scalar part
        v_scaled = (np.sin(theta) / theta) * v
        return np.concatenate([v_scaled, [np.cos(theta)]])
    
def quat_log(q):
    """
    Computes the logarithm of a unit quaternion given in [x, y, z, w] order.
    Returns a pure quaternion (with scalar part 0) in [x, y, z, 0] order that represents the rotation vector.
    
    For a unit quaternion q = [v, w] where:
        - v = [x, y, z] = sin(theta) * (rotation axis)
        - w = cos(theta)
    the logarithm is defined as:
    
        log(q) = [theta * (v / ||v||), 0]   if ||v|| > 0
               = [0, 0, 0, 0]              if ||v|| is close to 0
               
    where theta = arccos(w).
    
    Parameters:
        q (np.array): Unit quaternion in [x, y, z, w] order.
        
    Returns:
        np.array: Pure quaternion representing the logarithm of q in [x, y, z, 0] order.
    """
    v = q[:3]
    w = np.clip(q[3], -1.0, 1.0)  # Clamp w to avoid numerical issues with arccos
    theta = np.arccos(w)
    sin_theta = np.sin(theta)
    
    if sin_theta < 1e-8:
        # If the rotation angle is very small, return a zero rotation vector.
        return np.array([0.0, 0.0, 0.0, 0.0])
    else:
        return np.concatenate([(theta * v / sin_theta), [0.0]])


def transform_hand_orientation_to_world(q_WO, q_in_hand):
    """
    Transform an arbitrary orientation q_in_hand (hand frame)
    into the world frame using q_WO.
    
    Returns: q_in_world = q_WO * q_in_hand
    """
    # Normalize for safety, especially if there's floating-point drift
    q_in_hand = quaternion_normalize(q_in_hand)
    q_out = quat_multiply(q_WO, q_in_hand)
    return quaternion_normalize(q_out)

def quat_normalize(q):
    """Normalize a quaternion."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    return q / norm

def axis_angle_vector_to_rotation_matrix(rvec):
    """
    Converts a single 3D axis-angle vector to a 3x3 rotation matrix.
    The input vector's direction is the rotation axis, and its magnitude
    is the rotation angle in radians.

    Args:
        rvec (array-like): length-3 vector [rx, ry, rz]. 
                           The direction of rvec is the rotation axis.
                           The magnitude of rvec is the rotation angle (in radians).

    Returns:
        R (ndarray): A 3x3 rotation matrix.
    """
    # Convert input to float NumPy array
    rvec = np.asarray(rvec, dtype=float)

    # Compute the rotation angle as the vector norm
    angle = np.linalg.norm(rvec)

    # Handle the near-zero angle case: return the identity matrix
    if angle < 1e-15:
        return np.eye(3, dtype=float)

    # Compute the normalized axis
    axis = rvec / angle

    # Rodrigues' rotation formula:
    #   R = I + sin(theta)*[K] + (1 - cos(theta))*[K]^2
    # where [K] is the skew-symmetric cross-product matrix of 'axis'.
    ux, uy, uz = axis
    sin_theta = np.sin(angle)
    cos_theta = np.cos(angle)

    # Skew-symmetric cross-product matrix of the axis
    K = np.array([
        [0,    -uz,   uy],
        [uz,    0,   -ux],
        [-uy,  ux,    0 ]
    ], dtype=float)

    # 3x3 identity
    I = np.eye(3, dtype=float)

    # Compute rotation matrix
    R = I + sin_theta * K + (1.0 - cos_theta) * (K @ K)

    return R


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # We assume the quaternion is normalized. If not, you can normalize first.
    x2, y2, z2 = 2.0 * qx, 2.0 * qy, 2.0 * qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2

    # Construct rotation matrix
    R = np.array([
        [1.0 - (yy + zz), xy - wz,        xz + wy       ],
        [xy + wz,         1.0 - (xx + zz), yz - wx       ],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)]
    ], dtype=np.float64)
    return R