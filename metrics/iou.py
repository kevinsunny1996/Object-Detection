import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format = 'midpoint'):
    """
    Calculates intersection / union to find the correctness of the predicted object bounding box.

    Parameters:
        boxes_preds (tensor): Predictions of bounding boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels for each object boxes (BATCH_SIZE, 4)
        box_format (str): Contains the metadata on how the details are provided for each bounding boxes
        [Midpoint = (x, y, w, h) , Corners = (x1, y1, x2, y2)] 

    Returns:
        tensor: Intersection over union of all examples
    """

    # Slicing idx:idx + 1 to keep tensor dimensionality (To pick up individual coordinates)
    # Doing ... in indexing if there would be additional dimensions (more than 2 , in this example we assume 2D input)
    # Like for YOLO algorithm , the input shape could be (N, S, S, 4).

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[3:4] / 2
        box2_x1 = boxes_preds[..., 0:1] - boxes_preds[2:3] / 2
        box2_y1 = boxes_preds[..., 1:2] - boxes_preds[3:4] / 2
        box2_x2 = boxes_preds[..., 0:1] + boxes_preds[2:3] / 2
        box2_y2 = boxes_preds[..., 1:2] + boxes_preds[3:4] / 2
    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_preds[..., 0:1]
        box2_y1 = boxes_preds[..., 1:2]
        box2_x2 = boxes_preds[..., 2:3]
        box2_y2 = boxes_preds[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)