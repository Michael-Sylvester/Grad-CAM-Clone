import os
import csv
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import csv
import torch
from torchvision import models
from eval_utils import PascalVOCDataset, deletion_sensitivity

OUT_DIR = os.path.join('outputs', 'pascal_smoke')
VIS_DIR = os.path.join(OUT_DIR, 'visuals')
os.makedirs(VIS_DIR, exist_ok=True)

# read summary.csv produced by batch_generate_gradcam
summary_path = os.path.join(OUT_DIR, 'summary.csv')
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"Missing {summary_path}. Run batch_generate_gradcam first.")

rows = []
with open(summary_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# load VOC dataset to access annotations and tensors
ds = PascalVOCDataset(root='VOCdevkit', year='2007', image_set='val', download=False)
# build index by image_id -> dataset idx
id_to_idx = {}
for i in range(len(ds)):
    sample = ds[i]
    img_id = sample['image_id']
    id_to_idx[img_id] = i

model = models.vgg16(pretrained=True)
model.eval()
device = torch.device('cpu')
model.to(device)

results = []
for i, r in enumerate(rows):
    image_id = r['image_id']
    heat_path = r['saved_heatmap']
    cls = int(r['class_index'])
    score = float(r['class_score'])

    himg = Image.open(heat_path).convert('RGB')
    h_np = np.array(himg)[..., 0].astype(np.float32) / 255.0
    yx = np.unravel_index(np.argmax(h_np), h_np.shape[:2])
    y, x = int(yx[0]), int(yx[1])

    pointing = 0
    deletion_auc = None
    viz = himg.copy()
    draw = ImageDraw.Draw(viz)

    ann = None
    if image_id in id_to_idx:
        sample = ds[id_to_idx[image_id]]
        ann = sample['annotation']
        objs = ann.get('object', [])
        if isinstance(objs, dict):
            objs = [objs]
        try:
            orig_w = int(ann.get('size', {}).get('width', h_np.shape[1]))
            orig_h = int(ann.get('size', {}).get('height', h_np.shape[0]))
        except Exception:
            orig_w, orig_h = h_np.shape[1], h_np.shape[0]
        sx = h_np.shape[1] / float(orig_w) if orig_w>0 else 1.0
        sy = h_np.shape[0] / float(orig_h) if orig_h>0 else 1.0
        for obj in objs:
            b = obj.get('bndbox', None)
            if b is None:
                continue
            xmin = int(float(b['xmin'])*sx)
            ymin = int(float(b['ymin'])*sy)
            xmax = int(float(b['xmax'])*sx)
            ymax = int(float(b['ymax'])*sy)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(0,255,0), width=2)
            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                pointing = 1

    # draw argmax
    rrad = 3
    draw.ellipse([x-rrad, y-rrad, x+rrad, y+rrad], fill=(255,0,0))
    viz_path = os.path.join(VIS_DIR, f"{os.path.splitext(image_id)[0]}_viz.png")
    viz.save(viz_path)

    # deletion sensitivity for first 3 images only
    if i < 3 and image_id in id_to_idx:
        sample = ds[id_to_idx[image_id]]
        img_t = sample['img_tensor'].to(device)
        cam = np.array(himg.convert('L')).astype(np.float32)/255.0
        ds_res = deletion_sensitivity(model, img_t, cam, cls, device=device, steps=10, baseline='mean')
        deletion_auc = float(ds_res.get('auc', None))
        # save deletion curve plot
        xs = np.linspace(0,1,len(ds_res['scores']))
        plt.figure(figsize=(4,3))
        plt.plot(xs, ds_res['scores'], marker='o')
        plt.xlabel('Fraction pixels removed')
        plt.ylabel('Class score')
        plt.title(f"Deletion curve: {image_id} (AUC={deletion_auc:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"{os.path.splitext(image_id)[0]}_deletion.png"))
        plt.close()

    results.append({'image_id': image_id, 'saved_heatmap': heat_path, 'class_index': cls, 'class_score': score, 'pointing_hit': pointing, 'deletion_auc': deletion_auc, 'viz_path': viz_path})

csv_out = os.path.join(OUT_DIR, 'results.csv')
with open(csv_out, 'w', newline='', encoding='utf-8') as cf:
    fieldnames = ['image_id', 'saved_heatmap', 'class_index', 'class_score', 'pointing_hit', 'deletion_auc', 'viz_path']
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('Wrote', csv_out)
print('Saved visuals to', VIS_DIR)

# print summary
hits = [r['pointing_hit'] for r in results]
print('Pointing hit rate:', float(sum(hits)) / max(1, len(hits)))

print('First rows:')
for r in results[:5]:
    print(r)
