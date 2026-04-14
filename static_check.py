import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms


def preprocess_image(img_path):
    """Load and preprocess images for PyTorch models."""
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def replace_relu(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu(child)


def generate_gradcam(model, input_tensor, class_index, target_layer_idx=28):
    """Generate a Grad-CAM heatmap for a given model, input and class index."""
    activations = []
    gradients = []

    def save_activations(module, input, output):
        activations.append(output.detach().cpu().numpy()[0])

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu().numpy()[0])

    fhook = model.features[target_layer_idx].register_forward_hook(save_activations)
    bhook = model.features[target_layer_idx].register_full_backward_hook(save_gradient)

    model.zero_grad()
    output = model(input_tensor)
    target = output[0, class_index]
    target.backward()

    fhook.remove()
    bhook.remove()

    if len(activations) == 0 or len(gradients) == 0:
        raise RuntimeError('Hooks did not capture activations/gradients.')

    acts = activations[0]
    grads = gradients[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(acts * weights[:, np.newaxis, np.newaxis], axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    h = input_tensor.size(2)
    w = input_tensor.size(3)
    cam_upsampled = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    return cam_upsampled


def guided_backprop(model, input_tensor, class_index):
    model.eval()
    inp = input_tensor.clone().detach().requires_grad_(True)

    hooks = []
    def relu_backward_hook(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_full_backward_hook(relu_backward_hook))

    model.zero_grad()
    out = model(inp)
    loss = out[0, class_index]
    loss.backward()

    grad = inp.grad.detach().cpu().numpy()[0]
    for h in hooks:
        h.remove()

    gb = np.transpose(grad, (1, 2, 0))
    gb = np.sum(np.abs(gb), axis=2)
    gb = gb - gb.min()
    if gb.max() != 0:
        gb = gb / gb.max()
    return gb
