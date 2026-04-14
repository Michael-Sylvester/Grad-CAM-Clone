# Grad-CAM Clone — Project README

This repository implements Grad-CAM and guided backprop visualizations, provides evaluation utilities (pointing game and deletion sensitivity), and includes a small smoke-test using PASCAL VOC. It is intended for experimentation and reproducible evaluation of class-discriminative saliency methods.

## Project structure

- `GRAD_CAM_Clone.ipynb` — primary notebook: imports, model load, implementations of `generate_gradcam()` and `guided_backprop()`, visualizations, smoke tests and evaluation cells.
- `eval_utils.py` — evaluation helpers:
  - `PascalVOCDataset`, `make_pascal_dataloader`
  - `batch_generate_gradcam(model, dataloader, device, gradcam_fn, ...)` — runs Grad-CAM and saves overlays
  - `save_heatmap()` — write overlay images
  - `pointing_game_eval()` — computes pointing-game hit rate
  - `deletion_sensitivity()` — computes deletion curve and AUC
- `outputs/` — default output folder for saved heatmaps and CSV summaries (created when running the smoke test).

Results logger:
- Running the smoke test now also creates `outputs/pascal_smoke/results.csv` containing per-image metrics: `image_id`, `saved_heatmap`, `class_index`, `class_score`, `pointing_hit`, `deletion_auc`, and `viz_path` (visualization overlay path).

## High-level flow

1. Load a pretrained model (VGG16 by default) and set `model.eval()`.
2. Preprocess images with ImageNet transforms (resize→224×224, normalize).
3. Compute Grad-CAM for a target layer (default: `features[28]`) by:
   - registering a forward hook on the target conv layer to capture activations,
   - registering a tensor hook on the activation to capture gradients during backward (cloned to avoid view/inplace autograd errors),
   - forward pass to get logits, backward on the class score to populate gradients,
   - compute channel weights by global average pooling (GAP) of gradients,
   - weighted sum of activations, ReLU, normalize to [0,1], upsample to input size.
4. Optionally compute Guided Backprop (gradients w.r.t. input) using tensor-level hooks on ReLUs to clamp negative gradients.
5. Visualize: overlay Grad-CAM heatmaps on original images; combine Guided Backprop × Grad-CAM for Guided Grad-CAM.
6. Batch evaluation: `batch_generate_gradcam()` runs the above for many images, saves overlays, and records per-image metadata.

## Methodology — what & why

Grad-CAM identifies spatial regions within a convolutional feature map that a target class depends on by using gradients as importance weights. Guided Backprop highlights input pixels that positively influence the output; combining guided backprop with Grad-CAM yields high-resolution, class-discriminative visualizations (Guided Grad-CAM).

Design choices and rationale:
- Tensor-level hooks: hooking tensor outputs and cloning the gradient avoids PyTorch autograd errors when backward hooks return modified views.
- Normalizing heatmaps to [0,1] makes metrics and visual comparisons consistent.
- Using a single target layer (deep conv layer) follows the original Grad-CAM paper; multi-layer analysis can be added later.

## Evaluation metrics (detailed)

1. Pointing Game (Localization Accuracy)
   - For each image with bounding box annotations, compute Grad-CAM heatmap and find the single most salient point (argmax).
   - A hit is recorded if that point lies inside any ground-truth bounding box for the target object.
   - Report hit rate = hits / total_images.
   - Notes: requires bounding boxes (PASCAL VOC provides them). Useful to measure coarse localization ability.

2. Deletion / Insertion Sensitivity (Causal Fidelity)
   - Deletion: progressively remove (mask) the most important pixels according to the heatmap and record the model's class score at each step. A sharp, early drop indicates the heatmap correctly identifies causally influential pixels.
   - Insertion: start from a baseline (e.g., blurred image or mean image) and progressively add the most important pixels; a sharp increase indicates faithfulness.
   - We compute curves (score vs fraction of pixels removed/added) and summarize with AUC. Lower deletion AUC (faster drop) and higher insertion AUC (faster rise) indicate better explanations.
   - Practical considerations: per-pixel curves are expensive; region-based or superpixel masking can drastically reduce cost.

## Reproducibility and performance tips

- Run `GRAD_CAM_Clone.ipynb` from top→bottom after any edits, or use `%autoreload` to reload modules when iterating on code.
- For large-scale evaluation, use a GPU if available (set `device = torch.device('cuda')`) and increase `batch_size` in the dataloader when safe.
- Cache heatmaps to disk to avoid recomputing for multiple downstream evaluations.
- To reduce compute for deletion/insertion:
  - use `steps=10..20` instead of per-pixel steps,
  - use region-based masking (superpixels) — less forward passes, comparable signal,
  - downsample heatmaps when ranking pixels and apply masks at low resolution.

## How to run the smoke test (what we ran)

1. Ensure dependencies are installed in the virtualenv (`.venv`) as shown in `requirements.txt`.
2. In the notebook, the smoke-test cell uses `make_pascal_dataloader(..., download=True)` so PASCAL VOC will download if missing.
3. Run the smoke-test cell: it runs `batch_generate_gradcam()` over 10 images and saves overlays to `outputs/pascal_smoke`.
4. Run the evaluation cell to compute pointing-game hit rate and deletion sensitivity (AUC) for a small subset.

## Smoke-test evaluation (results)

The smoke-test produced evaluation artifacts under `outputs/pascal_smoke`:

- `summary.csv` — per-image summary written by the batch runner (image id, saved heatmap path, predicted class, score).
- `results.csv` — per-image evaluation and visualization links written by the results script with columns: `image_id`, `saved_heatmap`, `class_index`, `class_score`, `pointing_hit`, `deletion_auc`, `viz_path`.
- `visuals/` — contains per-image overlay visualizations with bounding boxes and the heatmap argmax point (`*_viz.png`) and deletion-curve plots for images where deletion sensitivity was computed (`*_deletion.png`).

Smoke-test numeric summary (10-image subset):
- Pointing game hit rate: 0.600 (6 / 10).
- Deletion sensitivity AUCs (computed for first 3 images): 0.050596, 0.343853, 0.025426 (these were computed with steps=10 on the smoke subset).

Notes:
- `pointing_hit` is 1 when the single most-salient pixel (heatmap argmax) falls inside any PASCAL bounding box for the image — this is the pointing-game implementation.
- `deletion_auc` was computed for a small subset (first 3 images) because per-pixel deletion is computationally expensive; see the Recommendations below for faster alternatives.

## Next steps (recommended)

1. Scale evaluation: run `batch_generate_gradcam()` on a larger subset (100–1000 images) and keep heatmaps cached. Use a GPU for speed.
2. Add region-based deletion/insertion (superpixels) to speed sensitivity analysis. See `skimage.segmentation.slic`.
3. Add multi-layer comparison utilities to compare heatmaps from different convolutional layers.
4. Add a `results.csv` logger (per-image metrics: hit, deletion AUC, predicted class, score) and plotting utilities for aggregated analysis.

## Conclusion

On a small 10-image smoke test the Grad-CAM pipeline produced a pointing-game hit rate of 0.60, indicating reasonable coarse localization on this tiny subset. Deletion sensitivity results vary across images (AUCs reported above) — some images show a strong drop in class score when top pixels are removed (lower AUC), while at least one image had a higher AUC (less sensitivity), suggesting mixed causal fidelity on this subset.

These results are preliminary: they demonstrate the pipeline end-to-end and validate that the evaluation artifacts (CSV, overlay visuals, deletion curves) are being produced. To draw robust conclusions we should:

- Scale the evaluation to a larger PASCAL subset (100+ images) or full val set and cache heatmaps.
- Use region-based masking (superpixels) or fewer steps for deletion/insertion to reduce compute while preserving signal.
- Run experiments on GPU for speed and enable reproducible seeds for consistent comparisons.

After scaling, we can produce aggregate plots (per-class and overall) and compare variants (different target layers, Grad-CAM++, guided Grad-CAM).

## Quick commands

Run smoke-test (notebook cell already provided):

```
# in notebook
loader = make_pascal_dataloader(root='VOCdevkit', year='2007', image_set='val', batch_size=1, download=True)
rows = batch_generate_gradcam(model, loader, device=torch.device('cpu'), gradcam_fn=generate_gradcam, out_dir='outputs/pascal_smoke', topk=1)
```

Run pointing game and deletion (notebook cell provided):

```
hit_rate, hits = pointing_game_eval(rows)
res = deletion_sensitivity(model, img_tensor, cam, class_index, device, steps=10)
```

## Contact / notes
- The code and README are minimal experimental scaffolding — feel free to suggest preferred evaluation parameters (dataset size, steps, region-based vs pixel-based). If you want, I can now run a larger evaluation; tell me how many images and whether to use CPU or GPU.
