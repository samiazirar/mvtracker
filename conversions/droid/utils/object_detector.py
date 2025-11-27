"""Object detection utilities for DROID point cloud generation."""

import numpy as np
import os
import sys
import torch
from torchvision.ops import box_convert
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from pathlib import Path

# Grounding DINO
sys.path.append("/workspace/third_party/groundingdino-cu128")
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

# SAM2
sys.path.append("/workspace/third_party/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_vlm_description(image_array, prompt, api_key, model_name="gpt-4o"):
    client = OpenAI(api_key=api_key)
    
    # Convert numpy image to base64
    pil_img = Image.fromarray(image_array)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    base64_image = base64.b64encode(buff.getvalue()).decode('utf-8')

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

class Sam2Wrapper:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[INFO] SAM2 using device: {self.device}")
        self._init_model()

    def _resolve_file(self, env_value: str | None, candidates):
        if env_value:
            path = Path(env_value).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"SAM2 asset specified via environment does not exist: {path}"
                )
            return path
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                return resolved
        raise FileNotFoundError(
            "Unable to locate SAM2 asset. Set $SAM2_CHECKPOINT/$SAM2_CONFIG or place one of: "
            + ", ".join(str(c) for c in candidates)
        )

    def _init_model(self):
        sam_checkpoint_env = os.environ.get("SAM2_CHECKPOINT")
        sam_config_env = os.environ.get("SAM2_CONFIG")

        workspace_root = Path.cwd()
        if not (workspace_root / "third_party").exists():
            workspace_root = Path("/workspace")

        # Define valid pairs of (checkpoint_filename, config_filename)
        # Checkpoints are expected in third_party/sam2/checkpoints/
        # Configs are expected in third_party/sam2/sam2/configs/sam2.1/
        model_pairs = [
            ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml"),
            ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
            ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"),
            ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"),
        ]

        sam_checkpoint = None
        sam_config = None

        # 1. Try environment variables first
        if sam_checkpoint_env and sam_config_env:
            sam_checkpoint = Path(sam_checkpoint_env).resolve()
            sam_config = Path(sam_config_env).resolve()
            if not sam_checkpoint.exists():
                raise FileNotFoundError(f"SAM2 checkpoint from env not found: {sam_checkpoint}")
            if not sam_config.exists():
                raise FileNotFoundError(f"SAM2 config from env not found: {sam_config}")
        
        # 2. Try to find a matching pair
        else:
            checkpoint_dir = workspace_root / "third_party/sam2/checkpoints"
            config_dir = workspace_root / "third_party/sam2/sam2/configs/sam2.1"

            for ckpt_name, cfg_name in model_pairs:
                ckpt_path = checkpoint_dir / ckpt_name
                cfg_path = config_dir / cfg_name
                
                if ckpt_path.exists() and cfg_path.exists():
                    sam_checkpoint = ckpt_path
                    sam_config = cfg_path
                    print(f"[INFO] Found matching SAM2 pair: {ckpt_name} / {cfg_name}")
                    break
            
            if sam_checkpoint is None:
                 # Fallback to searching recursively if standard paths fail (legacy behavior support)
                 # But for now, let's just raise error if no pair found to avoid mismatch
                 raise FileNotFoundError(
                     f"Could not find any valid SAM2 checkpoint/config pair in {checkpoint_dir} and {config_dir}"
                 )

        sam_module_root = (workspace_root / "third_party/sam2/sam2").resolve()
        # Calculate relative path for config if possible, else use absolute
        try:
            config_rel = str(sam_config.resolve().relative_to(sam_module_root)).replace(os.sep, "/")
        except ValueError:
             config_rel = str(sam_config)

        print(f"[INFO] Loading SAM2 model from {sam_checkpoint} with config {config_rel}")
        self.sam_model = build_sam2(config_file=config_rel, ckpt_path=str(sam_checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam_model)

    def predict_mask(self, image, boxes_cxcywh_norm):
        """
        Predict masks from image and normalized cxcywh boxes.
        """
        self.predictor.set_image(image)
        
        H, W = image.shape[:2]
        
        # Convert boxes to xyxy pixel coordinates
        boxes_xyxy = boxes_cxcywh_norm * torch.Tensor([W, H, W, H]).to(boxes_cxcywh_norm.device)
        boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        
        all_masks = []
        for box in boxes_xyxy:
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            # masks is (1, H, W)
            all_masks.append(masks[0])
            
        if not all_masks:
            return np.zeros((H, W), dtype=bool)
            
        # Combine all masks (logical OR)
        final_mask = np.any(np.stack(all_masks), axis=0)
        return final_mask

    def draw_masks(self, image, mask, save_path="sam_debug_masks.jpg"):
        import cv2
        
        overlay = image.copy()
        # Mask is boolean (H, W)
        
        # Color for mask (e.g., Red)
        color = np.array([255, 0, 0], dtype=np.uint8) 
        
        # Apply alpha blending
        alpha = 0.5
        
        # Where mask is True, blend with color
        # Ensure mask is boolean
        mask_bool = mask.astype(bool)
        
        overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved SAM debug image with masks to {save_path}")


class ObjectDetector:
    """
    Object detector using Grounding DINO for open-vocabulary detection.
    
    Wraps Grounding DINO model for detecting objects based on text prompts.
    """
    
    def __init__(self, config_path, checkpoint_path, device=None):
        """
        Initialize object detector.
        
        Args:
            config_path: Path to Grounding DINO config file
            checkpoint_path: Path to model checkpoint (will download if not exists)
            device: Device to run inference on (default: auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[INFO] GroundingDINO using device: {self.device}")
        
        if not os.path.exists(checkpoint_path):
            print(f"[INFO] Downloading GroundingDINO weights to {checkpoint_path}...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            os.system(f"wget {url} -O {checkpoint_path}")
        
        self.model = load_model(config_path, checkpoint_path, device=self.device)

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
        return boxes, logits

    def draw_boxes(self, image_numpy, boxes, logits, save_path="dino_debug_boxes.jpg"):
        import cv2
        
        drawn_image = image_numpy.copy()
        h_img, w_img, _ = drawn_image.shape
        
        # Convert boxes to pixel coords
        boxes_xyxy = boxes * torch.Tensor([w_img, h_img, w_img, h_img]).to(boxes.device)
        boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.astype(int)
            confidence = logits[i].item()
            
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(drawn_image, f"{confidence:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        cv2.imwrite(save_path, cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved DINO debug image with boxes to {save_path}")


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
