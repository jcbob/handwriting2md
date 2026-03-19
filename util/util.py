import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title=None):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)


def resize_to_consistent_resolution(img, target_width=2000, target_height=None):
    """
    Resizes an input image to a consistent resolution for downstream processing.

    Parameters:
        img: np.ndarray
            Input BGR image.
        target_width: int,......
            Desired width in pixels (default 2000).
        target_height: int or None
            Optional fixed height. If None, height is scaled to preserve aspect ratio.

    Returns:
        resized_img: np.ndarray
            Image resized to the desired resolution.
        scale_x, scale_y: float
            Scaling factors applied to width and height (for coordinate adjustments).
    """
    h, w = img.shape[:2]
    
    if target_height is None:
        # Preserve aspect ratio
        scale = target_width / w
        target_height = int(h * scale)
    else:
        # Fixed dimensions (may distort slightly)
        scale = None
    
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Calculate scaling factors (useful if you want to map coordinates back to original image)
    scale_x = target_width / w
    scale_y = target_height / h

    return resized_img, scale_x, scale_y


def find_edges(img_gray, kernel_size, binary_min_threshold):
    gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = np.uint8(gradient_magnitude)

    _, edges = cv2.threshold(gradient_magnitude, binary_min_threshold, 255, cv2.THRESH_BINARY)

    return edges


def dilate(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.dilate(img, kernel, iterations)
    return eroded


def erode(img, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img, kernel, iterations)
    return eroded


def draw_lines(img, lines, title=None):
    line_img = img.copy()

    for line in lines:
        # handle both shapes
        if isinstance(line, (list, np.ndarray)) and len(line) == 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=15)

    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)


def filter_lines2(lines, horizontal_angle_variance, vertical_angle_variance):
    horiz_thresh = np.radians(horizontal_angle_variance)
    vert_thresh = np.radians(vertical_angle_variance)

    horizontal_lines, vertical_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)

        # normalize angle to [0, π)
        if angle < 0:
            angle += np.pi

        if abs(angle) < horiz_thresh or abs(angle - np.pi) < horiz_thresh:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(angle - np.pi/2) < vert_thresh:
            vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines



def select_edge_lines(lines, orientation, min_separation):
    """
    Selects 2 longest lines that are sufficiently separated.
    Returns them sorted:
      - horizontal: top first, then bottom
      - vertical: left first, then right
    """
    lines_with_length = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.hypot(x2 - x1, y2 - y1)
        lines_with_length.append((length, line))
    lines_with_length.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for _, line in lines_with_length:
        if not selected:
            selected.append(line)
        else:
            x1, y1, x2, y2 = line
            if orientation == "horizontal":
                prev_y = np.mean([l[1] for l in selected] + [l[3] for l in selected])
                line_y = (y1 + y2) / 2
                if abs(line_y - prev_y) >= min_separation:
                    selected.append(line)
            elif orientation == "vertical":
                prev_x = np.mean([l[0] for l in selected] + [l[2] for l in selected])
                line_x = (x1 + x2) / 2
                if abs(line_x - prev_x) >= min_separation:
                    selected.append(line)
        if len(selected) == 2:
            break

    # Sort selected lines spatially
    if len(selected) == 2:
        if orientation == "horizontal":
            # sort by y (top first)
            selected.sort(key=lambda l: np.mean([l[1], l[3]]))
        elif orientation == "vertical":
            # sort by x (left first)
            selected.sort(key=lambda l: np.mean([l[0], l[2]]))

    return selected


def extend_line_to_image(line, img_shape, orientation):
    """
    Extends a line to the full image boundary in the specified orientation.

    Parameters:
        line: tuple (x1, y1, x2, y2)
        img_shape: shape of the image (height, width, ...)
        orientation: "horizontal" or "vertical"

    Returns:
        (x1, y1, x2, y2) - extended line clipped to image boundaries
    """
    height, width = img_shape[:2]
    x1, y1, x2, y2 = map(float, line)

    if orientation == "horizontal":
        # Avoid division by zero if the line is perfectly vertical
        if x2 == x1:
            x2 += 1e-6

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Extend line across entire image width
        new_x1, new_x2 = 0, width - 1
        new_y1 = slope * new_x1 + intercept
        new_y2 = slope * new_x2 + intercept

    elif orientation == "vertical":
        # Avoid division by zero if the line is perfectly horizontal
        if y2 == y1:
            y2 += 1e-6

        slope = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope * y1

        # Extend line across entire image height
        new_y1, new_y2 = 0, height - 1
        new_x1 = slope * new_y1 + intercept
        new_x2 = slope * new_y2 + intercept

    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    # Clip coordinates to image boundaries
    new_x1 = np.clip(new_x1, 0, width - 1)
    new_x2 = np.clip(new_x2, 0, width - 1)
    new_y1 = np.clip(new_y1, 0, height - 1)
    new_y2 = np.clip(new_y2, 0, height - 1)

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def compute_intersection(line1, line2):
    """
    Compute intersection point of two lines given as (x1, y1, x2, y2).
    Returns (x, y) as int tuple, or None if lines are parallel.
    """
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

def clip_point_to_image(point, img_shape):
    h, w = img_shape[:2]
    x = np.clip(point[0], 0, w - 1)
    y = np.clip(point[1], 0, h - 1)
    return (int(x), int(y))


def draw_corners(img, corners, title=None):
    corner_vis = img.copy()
    corner_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
    corner_labels = ["tl", "tr", "br", "bl"]
    
    for (corner, color, label) in zip(corners, corner_colors, corner_labels):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(corner_vis, (x, y), 20, color, -1)
        # Offset the label slightly so it's readable
        cv2.putText(corner_vis, label, (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
    
    plt.imshow(cv2.cvtColor(corner_vis, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)

def rescale_corners_to_original(corners, scale_x, scale_y):
    """
    Rescales a list of (x, y) coordinates from resized image space
    back to the original image dimensions.

    Parameters:
        points: list of (x, y)
            Coordinates from the resized image.
        scale_x: float
            Scale factor used for width during resizing.
        scale_y: float
            Scale factor used for height during resizing.

    Returns:
        rescaled_points: list of (x, y)
            Coordinates mapped back to the original image size.
    """
    rescaled_corners = []
    for (x, y) in corners:
        orig_x = int(x / scale_x)
        orig_y = int(y / scale_y)
        rescaled_corners.append((orig_x, orig_y))
    return rescaled_corners



def crop_from_corners(img, corners, title=None):
    x_coords = [c[0] for c in corners]
    y_coords = [c[1] for c in corners]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)


def find_text_region(img_path, show_images=True):
    img_original = cv2.imread(img_path)
    resize_image = False
    if img_original.shape != (4032, 3024):
        resize_image = True
        img, scale_x, scale_y = resize_to_consistent_resolution(img_original, target_width=3024, target_height=4032)
    else:
        img = img_orignal.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = find_edges(gray, kernel_size=5, binary_min_threshold=30)
    # show(edges)
    
    dilated = dilate(edges, kernel_size=7, iterations=1)
    if show_images:
        plt.figure(figsize=(14,12))
        plt.subplot(1,3,1)
        show(dilated, title="step 1 - dilate")
    
    eroded = erode(dilated, kernel_size=7, iterations=1)
    if show_images:
        plt.subplot(1,3,2)
        show(eroded, title="step 2 - erode")
    
    dilated = dilate(eroded, kernel_size=7, iterations=1)
    if show_images:
        plt.subplot(1,3,3)
        show(dilated, title="step 3 - dilate")

    # Apply the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        dilated,              # Edge-detected image
        rho=1,                # Distance resolution in pixels
        theta=np.pi / 180,    # Angle resolution in radians
        threshold=100,        # Minimum number of intersections to detect a line
        minLineLength=200,    # Minimum length of line to be detected
        maxLineGap=30         # Maximum allowed gap between line segments to be connected
    )
    
    # Filter lines to only have horizontal & vertical
    horizontal_angle_variance = 5
    vertical_angle_variance = 5
    horizontal_lines, vertical_lines = filter_lines2(lines, horizontal_angle_variance, vertical_angle_variance)
    filtered_lines = horizontal_lines + vertical_lines

    if show_images:
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        draw_lines(img, lines, title="lines found")
        
        plt.subplot(1, 2, 2)
        draw_lines(img, filtered_lines, title="lines filtered")
        
        plt.show()

    edges_horizontal = select_edge_lines(horizontal_lines, "horizontal", min_separation=2500)
    edges_vertical = select_edge_lines(vertical_lines, "vertical", min_separation=2000)

    if show_images:
        plt.figure(figsize=(10,8))
        draw_lines(img, edges_horizontal+edges_vertical)
        plt.show()

    extended_horizontal_edges = [extend_line_to_image(l, img.shape, "horizontal") for l in edges_horizontal]
    extended_vertical_edges = [extend_line_to_image(l, img.shape, "vertical") for l in edges_vertical]

    if show_images:
        plt.figure(figsize=(10,8))
        draw_lines(img, extended_horizontal_edges+extended_vertical_edges, title="extended page edges")
        plt.show

    top_line, bottom_line = edges_horizontal
    left_line, right_line = edges_vertical
    
    tl = compute_intersection(top_line, left_line)    # top-left
    tr = compute_intersection(top_line, right_line)   # top-right
    bl = compute_intersection(bottom_line, left_line) # bottom-left
    br = compute_intersection(bottom_line, right_line)# bottom-right
    
    corners = [tl, tr, br, bl]
    for x, y in corners:
        if x < 0 or y < 0:
            print(x, y)
    
    corners = [clip_point_to_image(cnr, img.shape) for cnr in corners]
    print(corners)

    if show_images:
        plt.figure(figsize=(10,8))
        draw_corners(img, corners, "Detected page corners")
        plt.show()

    if resize_image:
        corners_original = rescale_corners_to_original(corners, scale_x, scale_y)
        if show_images:
            plt.figure(figsize=(10,8))
            draw_corners(img_original, corners_original, "Page corners in original image size")
            plt.show()


    plt.figure(figsize=(10,8))
    if resize_image:
        crop_from_corners(img_original, corners_original, "Detected page on image")
    else:
        crop_from_corners(img_original, corners, "Detected page on image")
    plt.show()

    return corners














    