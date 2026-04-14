import time
import torch
import numpy as np
import cv2
from torchvision import models
from torch.utils.data import DataLoader, Subset

from eval_utils import PascalVOCDataset, batch_generate_gradcam, pointing_game_eval

# local gradcam (same as in notebook)
def generate_gradcam_local(model, input_tensor, class_index, target_layer_idx=28):
    activations = {}
    grads = {}
    def forward_hook(module, inp, out):
        activations['feat'] = out
        def save_grad(grad):
            grads['grad'] = grad.clone().detach()
        out.register_hook(save_grad)
    handle = model.features[target_layer_idx].register_forward_hook(forward_hook)
    model.zero_grad()
    out = model(input_tensor)
    score = out[0, class_index]
    score.backward()
    handle.remove()
    act = activations['feat'].detach().cpu()[0]
    grad = grads['grad'].detach().cpu()[0]
    weights = grad.mean(dim=(1,2))
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = torch.relu(cam)
    cam_np = cam.numpy()
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    cam_resized = cv2.resize(cam_np, (w, h))
    if cam_resized.max() > 0:
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
    else:
        cam_resized = cam_resized - cam_resized.min()
    return cam_resized


def main():
    device = torch.device('cpu')
    model = models.vgg16(pretrained=True)
    model.eval()

    root = 'VOCdevkit'
    ds = PascalVOCDataset(root=root, year='2007', image_set='val', download=False)
    num_images = min(80, len(ds))
    print(f'Dataset size {len(ds)}; running on {num_images} images')
    subset = Subset(ds, list(range(num_images)))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    start = time.time()
    try:
        rows = batch_generate_gradcam(model, loader, device, generate_gradcam_local, out_dir='outputs/pascal_eval_80', topk=1)
    except Exception as e:
        print('Error during batch_generate_gradcam:', e)
        raise
    elapsed = time.time() - start
    print(f'Finished batch on {len(rows)} images in {elapsed:.1f}s (~{elapsed/len(rows):.2f}s/img)')

    hit_rate, hits = pointing_game_eval(rows)
    print(f'Pointing game hit rate: {hit_rate:.3f} ({sum(hits)}/{len(hits)})')

    with open('outputs/pascal_eval_80/summary_run.txt', 'w') as f:
        f.write(f'num_images={len(rows)}\n')
        f.write(f'elapsed_s={elapsed:.3f}\n')
        f.write(f'sec_per_image={elapsed/len(rows):.6f}\n')
        f.write(f'pointing_hit_rate={hit_rate:.6f}\n')
    print('Wrote outputs/pascal_eval_80/summary_run.txt')

if __name__ == '__main__':
    main()
