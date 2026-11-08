"""Shared helpers for train.py and evaluate.py (torchvision models, transforms, eval dataset)."""
import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import models, transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARCHS = {"resnet50": 224, "vgg16": 224, "inception_v3": 299,
         "efficientnet_b0": 224, "mobilenet_v3_large": 224}   # input sizes

def pick_device(pref="auto"):
    """auto = CUDA if present, else CPU. The Apple GPU ('mps') is opt-in (--device mps) because some
    PyTorch/macOS combinations crash natively inside convolutions."""
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(arch, n_classes, pretrained=True):
    """Returns (model, head_module). Only the head is new; everything else is ImageNet-pretrained."""
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}")
    w = "DEFAULT" if pretrained else None
    m = models.get_model(arch, weights=w)
    if arch == "inception_v3":
        m.aux_logits = False      # drop the auxiliary classifier so forward() returns plain logits
        m.AuxLogits = None
    if arch in ("resnet50", "inception_v3"):
        m.fc = nn.Linear(m.fc.in_features, n_classes); head = m.fc
    elif arch == "vgg16":
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes); head = m.classifier[6]
    elif arch == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes); head = m.classifier[1]
    else:  # mobilenet_v3_large
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes); head = m.classifier[3]
    return m, head

def set_backbone_trainable(model, head, flag):
    for p in model.parameters():
        p.requires_grad = flag
    for p in head.parameters():
        p.requires_grad = True

def train_tf(img):
    return transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.2),
        transforms.RandomGrayscale(0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])

def eval_tf(img):
    return transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])

def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")   # honour phone rotation

class EvalDataset(Dataset):
    """root/<class_name>/*.jpg. Folders whose name is not in `classes` get label -1 ('unknown plant')."""
    def __init__(self, root, classes, tf):
        self.tf, self.items, self.unknown_folders = tf, [], []
        idx = {c: i for i, c in enumerate(classes)}
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            label = idx.get(d, -1)
            if label == -1:
                self.unknown_folders.append(d)
            for f in sorted(os.listdir(full)):
                if os.path.splitext(f)[1].lower() in IMG_EXT and not f.startswith("."):
                    self.items.append((os.path.join(full, f), label))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(load_rgb(path)), label
