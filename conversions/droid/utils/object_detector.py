"""Object detection utilities for DROID point cloud generation."""

import numpy as np
import os
import sys
import torch
from torchvision.ops import box_convert
from PIL import Image

# Grounding DINO
sys.path.append("/workspace/third_party/groundingdino-cu128")
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T


class ObjectDetector:
    """
    Object detector using Grounding DINO for open-vocabulary detection.
    
    Wraps Grounding DINO model for detecting objects based on text prompts.
    """
    
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        """
        Initialize object detector.
        
        Args:
            config_path: Path to Grounding DINO config file
            checkpoint_path: Path to model checkpoint (will download if not exists)
            device: Device to run inference on (default: "cuda")
        """
        self.device = device
        if not os.path.exists(checkpoint_path):
            print(f"[INFO] Downloading GroundingDINO weights to {checkpoint_path}...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            os.system(f"wget {url} -O {checkpoint_path}")
        
        self.model = load_model(config_path, checkpoint_path, device=device)

    def detect(self, image_numpy, prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Detect objects in image based on text prompt.
        
        Args:
            image_numpy: (H, W, 3) RGB numpy array
            prompt: Text prompt describing objects to detect
            box_threshold: Confidence threshold for bounding boxes (default: 0.35)
            text_threshold: Confidence threshold for text matching (default: 0.25)
            
        Returns:
            Bounding boxes tensor in cxcywh format (normalized 0-1)
        """
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(image_numpy)
        image_tensor, _ = transform(image_pil, None)
        
        boxes, logits, phrases = predict(
            self.model,
            image_tensor,
            prompt,
            box_threshold,
            text_threshold,
            device=self.device
        )
        return boxes


def boxes_to_mask(boxes, H, W):
    """
    Convert bounding boxes to binary mask.
    
    Args:
        boxes: Tensor of bounding boxes in cxcywh format (normalized 0-1)
        H: Image height in pixels
        W: Image width in pixels
        
    Returns:
        Binary mask (H, W) with True inside any bounding box
    """
    mask = np.zeros((H, W), dtype=bool)
    if boxes.shape[0] == 0:
        return mask
        
    # Convert to xyxy pixel coords
    boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
    boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        mask[y1:y2, x1:x2] = True
    return mask
