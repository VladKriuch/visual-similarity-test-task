import numpy as np

def iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Compute intersection coordinates
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    
    # Compute intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    
    # Compute areas
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    # Compute IoU
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def merge_footwear_boxes(boxes, distance_threshold=50):
    """Merge close or overlapping footwear boxes into a single bounding box."""
    if len(boxes) < 2:
        return boxes  # No need to merge if only one or zero boxes exist

    merged_boxes = []
    used = set()

    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        x1, y1, x2, y2 = box1
        for j, box2 in enumerate(boxes):
            if i == j or j in used:
                continue
            
            x1_p, y1_p, x2_p, y2_p = box2
            # Check if box2 is inside box1
            if x1 <= x1_p and y1 <= y1_p and x2 >= x2_p and y2 >= y2_p:
                used.add(j)
                continue

            # Compute distance between box centers
            center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            center2 = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)
            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

            if distance < distance_threshold:  # Merge if close enough
                x1, y1 = min(x1, x1_p), min(y1, y1_p)
                x2, y2 = max(x2, x2_p), max(y2, y2_p)
                used.add(j)

        merged_boxes.append((x1, y1, x2, y2))
        used.add(i)

    return merged_boxes

def custom_non_max_suppression(labels, boxes, confs, iou_threshold=0.3, distance_threshold=50):
    """Perform NMS and merge footwear boxes."""
    boxes = np.array(boxes)
    confs = np.array(confs)
    labels = np.array(labels)

    sorted_indices = np.argsort(confs)[::-1]
    keep = []
    footwear_boxes = []

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        current_label = labels[current_idx]

        if current_label == "Footwear":
            footwear_boxes.append(boxes[current_idx])
            sorted_indices = sorted_indices[1:]
            continue

        keep.append(current_idx)
        remaining_indices = sorted_indices[1:]
        to_keep = []

        for idx in remaining_indices:
            if labels[current_idx] != labels[idx] or iou(boxes[current_idx], boxes[idx]) < iou_threshold:
                to_keep.append(idx)

        sorted_indices = np.array(to_keep)

    if footwear_boxes:
        merged_footwear_boxes = merge_footwear_boxes(footwear_boxes, distance_threshold)
        for box in merged_footwear_boxes:
            keep.append(len(boxes))  # Adding new index
            boxes = np.vstack([boxes, box])
            labels = np.append(labels, "Footwear")
            confs = np.append(confs, 1.0)  # Assign highest confidence

    return {
        "labels": labels[keep].tolist(),
        "boxes": boxes[keep].tolist(),
        "conf": confs[keep].tolist()
    }