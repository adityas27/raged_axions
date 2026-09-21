"""Training utilities for binary lunar-terrain classification."""

from pathlib import Path

import numpy as np
import torch


class ModelTrainer:
    """Train, validate, checkpoint, and run inference for one-logit models."""

    def __init__(self, model, device=None, use_amp=True, checkpoint_metadata=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.train_loader = self.valid_loader = self.test_loader = None
        self.criterion = self.optimizer = self.scheduler = None
        self.best_balanced_accuracy, self.best_threshold = -1.0, 0.5

    def init_train_modules(self, train_loader, valid_loader, criterion, optimizer, scheduler=None):
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion, self.optimizer, self.scheduler = criterion, optimizer, scheduler

    def init_test_modules(self, test_loader):
        self.test_loader = test_loader

    @staticmethod
    def balanced_accuracy(targets, predictions):
        targets, predictions = np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
        tn = np.sum((targets == 0) & (predictions == 0))
        fp = np.sum((targets == 0) & (predictions == 1))
        tp = np.sum((targets == 1) & (predictions == 1))
        fn = np.sum((targets == 1) & (predictions == 0))
        recall_0 = tn / (tn + fp) if tn + fp else 0.0
        recall_1 = tp / (tp + fn) if tp + fn else 0.0
        return (recall_0 + recall_1) / 2.0

    @classmethod
    def calculate_metrics(cls, targets, probabilities, threshold=0.5):
        targets = np.asarray(targets, dtype=int)
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        tn = int(np.sum((targets == 0) & (predictions == 0)))
        fp = int(np.sum((targets == 0) & (predictions == 1)))
        tp = int(np.sum((targets == 1) & (predictions == 1)))
        fn = int(np.sum((targets == 1) & (predictions == 0)))
        recall_0 = tn / (tn + fp) if tn + fp else 0.0
        recall_1 = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall_1 / (precision + recall_1) if precision + recall_1 else 0.0
        return {"balanced_accuracy": (recall_0 + recall_1) / 2.0,
                "accuracy": (tp + tn) / len(targets) if len(targets) else 0.0,
                "recall_class_0": recall_0, "recall_class_1": recall_1,
                "precision": precision, "f1": f1, "threshold": float(threshold),
                "confusion_matrix": [[tn, fp], [fn, tp]]}

    @classmethod
    def find_best_threshold(cls, targets, probabilities):
        best_threshold, best_score = 0.5, -1.0
        for threshold in np.arange(0.05, 0.951, 0.01):
            score = cls.balanced_accuracy(targets, probabilities >= threshold)
            if score > best_score:
                best_threshold, best_score = float(threshold), score
        return best_threshold, best_score

    def _move_batch(self, images, sun_features, targets=None):
        non_blocking = self.device.type == "cuda"
        images = images.to(self.device, non_blocking=non_blocking)
        sun_features = sun_features.to(self.device, non_blocking=non_blocking)
        return images, sun_features, targets.to(self.device, non_blocking=non_blocking) if targets is not None else None

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_items = 0.0, 0
        for images, sun_features, targets in self.train_loader:
            images, sun_features, targets = self._move_batch(images, sun_features, targets)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits, loss = self.model(images, sun_features), None
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss, total_items = total_loss + loss.item() * images.size(0), total_items + images.size(0)
        return total_loss / total_items

    @torch.inference_mode()
    def validate(self):
        """Metrics are calculated once from the complete validation dataset."""
        self.model.eval()
        total_loss, total_items, targets_all, probabilities_all = 0.0, 0, [], []
        for images, sun_features, targets in self.valid_loader:
            images, sun_features, targets = self._move_batch(images, sun_features, targets)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images, sun_features)
                loss = self.criterion(logits, targets)
            total_loss, total_items = total_loss + loss.item() * images.size(0), total_items + images.size(0)
            targets_all.append(targets.cpu().numpy())
            probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
        targets, probabilities = np.concatenate(targets_all), np.concatenate(probabilities_all)
        threshold, _ = self.find_best_threshold(targets, probabilities)
        return total_loss / total_items, self.calculate_metrics(targets, probabilities, threshold), targets, probabilities

    def fit(self, epochs, checkpoint_path, log_path=None):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = Path(log_path).open("w") if log_path else None
        try:
            for epoch in range(1, epochs + 1):
                train_loss = self.train_one_epoch()
                valid_loss, metrics, _, _ = self.validate()
                if self.scheduler is not None:
                    self.scheduler.step()
                line = (f"epoch={epoch},train_loss={train_loss:.6f},val_loss={valid_loss:.6f},"
                        f"balanced_accuracy={metrics['balanced_accuracy']:.6f},threshold={metrics['threshold']:.2f}")
                print(line)
                if log_handle:
                    log_handle.write(line + "\n")
                if metrics["balanced_accuracy"] > self.best_balanced_accuracy:
                    self.best_balanced_accuracy, self.best_threshold = metrics["balanced_accuracy"], metrics["threshold"]
                    self.save_checkpoint(checkpoint_path, epoch)
                    print(f"  Saved best checkpoint (BA={self.best_balanced_accuracy:.4f}).")
        finally:
            if log_handle:
                log_handle.close()

    def save_checkpoint(self, path, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "best_balanced_accuracy": self.best_balanced_accuracy, "best_threshold": self.best_threshold,
                    "model_config": self.checkpoint_metadata}, path)

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_balanced_accuracy = checkpoint.get("best_balanced_accuracy", -1.0)
        self.best_threshold = checkpoint.get("best_threshold", 0.5)
        return checkpoint

    @torch.inference_mode()
    def predict(self, loader=None):
        self.model.eval()
        probabilities = []
        for images, sun_features, *_ in (loader or self.test_loader):
            images, sun_features, _ = self._move_batch(images, sun_features)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images, sun_features)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities)
