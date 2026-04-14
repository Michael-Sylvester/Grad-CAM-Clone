import os
import csv
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import cv2


def _default_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class PascalVOCDataset(Dataset):
    """Simple wrapper around torchvision VOCDetection that returns preprocessed tensors,
    the original PIL image and the annotation dict.
    Note: set `download=True` if you haven't got VOC data locally.
    """

    def __init__(self, root: str, year: str = '2007', image_set: str = 'val',
                 transform: Callable = None, download: bool = False):
        self.voc = datasets.VOCDetection(root, year=year, image_set=image_set, download=download)
        self.transform = transform if transform is not None else _default_preprocess()

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        # Keep original image (resized copy for visualization)
        orig = img.copy()
        img_tensor = self.transform(img).unsqueeze(0)  # 1,C,H,W
        # image id / filename
        image_id = target['annotation'].get('filename', str(idx))
        return {
            'image_id': image_id,
            'img_tensor': img_tensor,
            'orig_image': orig,
            'annotation': target['annotation']
        }


def make_pascal_dataloader(root: str, year: str = '2007', image_set: str = 'val',
                           batch_size: int = 1, num_workers: int = 2, download: bool = False):
    ds = PascalVOCDataset(root=root, year=year, image_set=image_set, download=download)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


def save_heatmap(heatmap: np.ndarray, orig_img: Image.Image, out_path: str, alpha: float = 0.5, cmap: str = 'jet'):
    """Overlay heatmap onto original image and save to `out_path`.
    `heatmap` expected in [0,1] float, shape (H,W).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hmap = (heatmap * 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    orig = np.array(orig_img.resize((hmap.shape[1], hmap.shape[0])))
    if orig.dtype != np.uint8:
        orig = (orig * 255).astype(np.uint8)
    overlay = cv2.addWeighted(hmap_color, alpha, orig[..., ::-1], 1 - alpha, 0)
    # convert BGR -> RGB before saving with PIL
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    Image.fromarray(overlay).save(out_path)


def batch_generate_gradcam(model: torch.nn.Module,
                           dataloader: DataLoader,
                           device: torch.device,
                           gradcam_fn: Callable,
                           target_layer_idx: int = 28,
                           out_dir: str = 'gradcam_outputs',
                           topk: int = 1) -> List[Dict[str, Any]]:
    """Run Grad-CAM over a dataloader and save heatmaps and a CSV summary.

    gradcam_fn: function(model, input_tensor, class_index, target_layer_idx) -> heatmap (H,W)
    Returns list of result dicts with keys: image_id, saved_path, class_index, score
    """
    model.to(device)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'summary.csv')
    rows = []

    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['image_id', 'saved_heatmap', 'class_index', 'class_score'])

        for item in tqdm(dataloader, desc='Grad-CAM batches'):
            # handle batch_size default 1; item is a dict or list
            if isinstance(item, list) or isinstance(item, tuple):
                sample = item[0]
            else:
                sample = item

            image_id = sample['image_id']
            img_tensor = sample['img_tensor'].to(device)
            orig = sample['orig_image']
            # model forward
            out = model(img_tensor)
            probs = F.softmax(out[0], dim=0)
            top_prob, top_catid = torch.topk(probs, topk)

            for k in range(topk):
                cls = top_catid[k].item()
                score = top_prob[k].item()
                heatmap = gradcam_fn(model, img_tensor, cls, target_layer_idx=target_layer_idx)
                # ensure heatmap is numpy float in [0,1]
                if isinstance(heatmap, torch.Tensor):
                    heatmap = heatmap.detach().cpu().numpy()
                heatmap = np.clip(heatmap, 0.0, 1.0)
                fname = f"{os.path.splitext(image_id)[0]}_cls{cls}.png"
                saved = os.path.join(out_dir, fname)
                save_heatmap(heatmap, orig, saved)
                writer.writerow([image_id, saved, cls, score])
                rows.append({'image_id': image_id, 'saved_heatmap': saved, 'class_index': cls, 'class_score': score, 'annotation': sample['annotation']})

    return rows


def pointing_game_eval(results: List[Dict[str, Any]]) -> Tuple[float, List[int]]:
    """Compute pointing game hit rate.

    `results` is the list produced by `batch_generate_gradcam` containing 'saved_heatmap' and 'annotation'.
    A hit occurs when the argmax of the heatmap falls inside any ground-truth bbox for the image.
    Returns hit_rate and list of per-sample hits (1/0).
    """
    hits = []
    for r in results:
        hmap = np.array(Image.open(r['saved_heatmap']).convert('RGB'))
        # heatmap overlay saved as RGB; recover grayscale importance by taking red channel
        gray = hmap[..., 0].astype(np.float32)
        # find argmax coordinates in resized image space
        idx = np.unravel_index(np.argmax(gray), gray.shape[:2])  # (y,x)
        y, x = idx

        ann = r['annotation']
        objs = ann.get('object', [])
        if isinstance(objs, dict):
            objs = [objs]

        hit = 0
        for obj in objs:
            b = obj.get('bndbox', None)
            if b is None:
                continue
            xmin = int(float(b['xmin']))
            ymin = int(float(b['ymin']))
            xmax = int(float(b['xmax']))
            ymax = int(float(b['ymax']))
            # The saved heatmap image is resized to the model input size (e.g., 224x224).
            # Annotation bboxes are in the original image coordinate system; rescale them to heatmap size.
            size = ann.get('size', {})
            try:
                orig_w = int(size.get('width'))
                orig_h = int(size.get('height'))
            except Exception:
                # fallback: assume original size equals heatmap size
                orig_h, orig_w = gray.shape[0], gray.shape[1]

            heat_h, heat_w = gray.shape[0], gray.shape[1]
            sx = heat_w / float(orig_w) if orig_w > 0 else 1.0
            sy = heat_h / float(orig_h) if orig_h > 0 else 1.0

            xmin_s = int(np.round(xmin * sx))
            xmax_s = int(np.round(xmax * sx))
            ymin_s = int(np.round(ymin * sy))
            ymax_s = int(np.round(ymax * sy))

            if x >= xmin_s and x <= xmax_s and y >= ymin_s and y <= ymax_s:
                hit = 1
                break

        hits.append(hit)

    hit_rate = float(np.sum(hits)) / max(1, len(hits))
    return hit_rate, hits


def deletion_sensitivity(model: torch.nn.Module,
                         img_tensor: torch.Tensor,
                         heatmap: np.ndarray,
                         class_index: int,
                         device: torch.device,
                         steps: int = 20,
                         baseline: str = 'mean') -> Dict[str, Any]:
    """Compute deletion sensitivity curve and AUC.

    This progressively masks the most important pixels and records the model score for `class_index`.
    Note: This is computationally heavy (steps * forward passes). Prefer small subsets or region-based masking.
    """
    model.to(device)
    model.eval()

    img = img_tensor.clone().detach().cpu()[0]  # C,H,W
    C, H, W = img.shape
    flat_heat = heatmap.ravel()
    order = np.argsort(-flat_heat)  # descending importance
    total = H * W

    if baseline == 'mean':
        base_val = float(img.mean())
    else:
        base_val = 0.0

    scores = []
    for s in range(steps + 1):
        n = int((s / steps) * total)
        mask_flat = np.zeros(total, dtype=bool)
        if n > 0:
            mask_flat[order[:n]] = True
        mask2d = mask_flat.reshape(H, W)
        masked = img.clone()
        for c in range(C):
            channel = masked[c].numpy()
            channel[mask2d] = base_val
            masked[c] = torch.from_numpy(channel)

        inp = masked.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            prob = F.softmax(out[0], dim=0)[class_index].item()
        scores.append(prob)

    # compute AUC (normalized) for deletion curve without relying on np.trapz
    xs = np.linspace(0, 1, len(scores))
    # trapezoidal rule manually to avoid dependency on numpy.trapz
    s = np.array(scores, dtype=float)
    dx = xs[1:] - xs[:-1]
    auc = float(np.sum(dx * (s[1:] + s[:-1]) / 2.0) / (xs[-1] - xs[0]))
    return {'scores': scores, 'auc': auc}
