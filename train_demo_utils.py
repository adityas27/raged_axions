"""Lunar terrain binary-classification training and submission pipeline."""

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

from model_trainer import ModelTrainer


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_METADATA = PROJECT_ROOT / "data" / "train_metadata.csv"
TEST_METADATA = PROJECT_ROOT / "data" / "test_metadata.csv"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "train_images"
EVAL_IMAGE_DIR = PROJECT_ROOT / "eval_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLIT_PATH = OUTPUT_DIR / "split.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
LOG_PATH = OUTPUT_DIR / "training_log.txt"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# Easy-to-edit training configuration.
BATCH_SIZE = 32
IMAGE_SIZE = 160
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 25
SEED = 42
VALIDATION_FRACTION = 0.20
NUM_WORKERS = 4
USE_AMP = True


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print("AMP: enabled")
    else:
        print("AMP: disabled (CUDA unavailable)")
    return device


def validate_metadata(dataframe, required_columns, source):
    missing = set(required_columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if dataframe["image_id"].isna().any() or not dataframe["image_id"].is_unique:
        raise ValueError(f"{source} must contain unique, non-empty image_id values.")


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_image_hashes(dataframe, image_dir):
    dataframe = dataframe.copy()
    hashes = []
    for image_id in dataframe["image_id"]:
        image_path = Path(image_dir) / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by metadata does not exist: {image_path}")
        hashes.append(content_hash(image_path))
    dataframe["_image_hash"] = hashes
    return dataframe


def _print_split_summary(train_df, val_df):
    def describe(name, frame):
        counts = frame["label"].value_counts().reindex([0, 1], fill_value=0)
        total = len(frame)
        print(
            f"{name}: {total} | class 0: {counts[0]} ({counts[0] / total:.1%}) | "
            f"class 1: {counts[1]} ({counts[1] / total:.1%})"
        )

    describe("Train", train_df)
    describe("Validation", val_df)
    all_hashes = pd.concat([train_df["_image_hash"], val_df["_image_hash"]])
    duplicate_groups = int(all_hashes.value_counts().gt(1).sum())
    leakage = set(train_df["_image_hash"]) & set(val_df["_image_hash"])
    print(f"Duplicate-content groups: {duplicate_groups}")
    print(f"Duplicate-content leakage check: {'passed' if not leakage else 'FAILED'}")
    if leakage:
        raise ValueError("Identical image content occurs in both train and validation splits.")


def _validate_saved_split(split, hashed_df):
    if list(split.columns) != ["image_id", "split"]:
        raise ValueError("split.csv must have exactly image_id,split columns.")
    if split["image_id"].duplicated().any() or set(split["image_id"]) != set(hashed_df["image_id"]):
        raise ValueError("split.csv must contain each training image_id exactly once.")
    if not set(split["split"]).issubset({"train", "val"}) or set(split["split"]) != {"train", "val"}:
        raise ValueError("split.csv split values must be train or val, with both present.")
    assigned = hashed_df.merge(split, on="image_id", validate="one_to_one")
    train_df = assigned[assigned["split"] == "train"].copy()
    val_df = assigned[assigned["split"] == "val"].copy()
    _print_split_summary(train_df, val_df)
    return train_df.drop(columns="split"), val_df.drop(columns="split")


def load_or_create_split(metadata):
    """Reuse a valid persisted split; otherwise create and save a deterministic one."""
    hashed_df = add_image_hashes(metadata, TRAIN_IMAGE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SPLIT_PATH.exists():
        try:
            print(f"Loading existing split: {SPLIT_PATH}")
            return _validate_saved_split(pd.read_csv(SPLIT_PATH), hashed_df)
        except (ValueError, pd.errors.ParserError) as error:
            print(f"Existing split is invalid; recreating it ({error}).")

    # StratifiedGroupKFold keeps every content-hash group wholly in one partition.
    # Five folds yields the requested approximately 80/20 train/validation split.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(
        splitter.split(hashed_df, hashed_df["label"], groups=hashed_df["_image_hash"])
    )
    train_df = hashed_df.iloc[train_indices].copy()
    val_df = hashed_df.iloc[val_indices].copy()
    split = pd.concat([
        pd.DataFrame({"image_id": train_df["image_id"], "split": "train"}),
        pd.DataFrame({"image_id": val_df["image_id"], "split": "val"}),
    ]).sort_values("image_id")
    split.to_csv(SPLIT_PATH, index=False)
    print(f"Saved deterministic split: {SPLIT_PATH}")
    _print_split_summary(train_df, val_df)
    return train_df, val_df


class LunarDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform, has_labels=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / row["image_id"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        angle_rad = np.deg2rad(float(row["sun_azimuth_angle"]))
        sun_features = torch.tensor(
            [np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32
        )
        if self.has_labels:
            return image, sun_features, torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, sun_features, row["image_id"]


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


class LunarFusionNet(nn.Module):
    """ResNet18 image encoder fused with circular sun-azimuth features."""
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        image_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.sun_branch = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(image_features + 64, 256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.20), nn.Linear(64, 1),
        )

    def forward(self, images, sun_features):
        image_features = self.backbone(images)
        sun_features = self.sun_branch(sun_features)
        return self.classifier(torch.cat((image_features, sun_features), dim=1)).squeeze(1)


def build_loader(dataset, shuffle=False):
    cuda = torch.cuda.is_available()
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=cuda, persistent_workers=NUM_WORKERS > 0,
    )


def train_model():
    """Run training only when explicitly invoked by the user."""
    set_seed()
    device = select_device()
    metadata = pd.read_csv(TRAIN_METADATA)
    validate_metadata(metadata, {"image_id", "sun_azimuth_angle", "label"}, "train_metadata.csv")
    train_df, val_df = load_or_create_split(metadata)
    train_dataset = LunarDataset(train_df, TRAIN_IMAGE_DIR, train_transform())
    val_dataset = LunarDataset(val_df, TRAIN_IMAGE_DIR, evaluation_transform())
    class_counts = train_df["label"].value_counts()
    pos_weight = torch.tensor(
        [float(class_counts.get(0, 1)) / float(class_counts.get(1, 1))], device=device
    )
    model = LunarFusionNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    trainer = ModelTrainer(
        model=model, device=device, use_amp=USE_AMP,
        checkpoint_metadata={"architecture": "LunarFusionNet/ResNet18", "image_size": IMAGE_SIZE,
                             "sun_features": ["sin(azimuth)", "cos(azimuth)"]},
    )
    trainer.init_train_modules(
        build_loader(train_dataset, shuffle=True), build_loader(val_dataset),
        nn.BCEWithLogitsLoss(pos_weight=pos_weight), optimizer, scheduler,
    )
    trainer.fit(EPOCHS, CHECKPOINT_PATH, LOG_PATH)
    return trainer


def generate_submission():
    """Generate outputs/submission.csv only when explicitly invoked after training."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    set_seed()
    device = select_device()
    metadata = pd.read_csv(TEST_METADATA)
    validate_metadata(metadata, {"image_id", "sun_azimuth_angle"}, "test_metadata.csv")
    dataset = LunarDataset(metadata, EVAL_IMAGE_DIR, evaluation_transform(), has_labels=False)
    model = LunarFusionNet()
    trainer = ModelTrainer(model=model, device=device, use_amp=USE_AMP)
    checkpoint = trainer.load_checkpoint(CHECKPOINT_PATH, load_optimizer=False)
    probabilities = trainer.predict(build_loader(dataset))
    submission = pd.DataFrame({
        "image_id": metadata["image_id"],
        "label": (probabilities >= checkpoint.get("best_threshold", 0.5)).astype(int),
    })
    if list(submission.columns) != ["image_id", "label"] or not submission["image_id"].is_unique:
        raise ValueError("Submission validation failed.")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "submission"))
    args = parser.parse_args()
    if args.operation == "train":
        train_model()
    else:
        generate_submission()
