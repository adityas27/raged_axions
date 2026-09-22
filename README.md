🌙 Lunar Terrain Classification

A PyTorch-based binary image classification pipeline for lunar terrain recognition, combining visual features from satellite/terrain imagery with sun azimuth information.

✨ Highlights

🧠 ResNet18 image backbone

☀️ Sun azimuth encoded as sin(angle) + cos(angle)

🔗 Image + metadata feature fusion

⚖️ Class-weighted BCEWithLogitsLoss

📊 Balanced Accuracy–based model selection

🎯 Automatic validation threshold optimization

🔒 Content-hash grouped train/validation split to prevent image leakage

⚡ CUDA + Automatic Mixed Precision (AMP) support

💾 Best-model checkpointing and submission generation

📁 Project Structure
.
├── data/
│   ├── train_metadata.csv
│   └── test_metadata.csv
├── train_images/
├── eval_images/
├── model_trainer.py
├── train.py
└── outputs/
    ├── best_model.pt
    ├── split.csv
    ├── training_log.txt
    └── submission.csv

🚀 Usage

Install dependencies:

pip install torch torchvision pandas numpy pillow scikit-learn


Train the model:

python train.py train


Generate predictions:

python train.py submission


The final predictions are written to:

outputs/submission.csv

🏗️ Model
Image
  │
  ▼
ResNet18 ─────────┐
                  ├──► Feature Fusion ──► Classifier ──► Terrain Class
Sun Azimuth ──────┘
     │
     └──► sin(angle), cos(angle)


Images are resized to 160×160, while the sun azimuth is represented using circular sine/cosine features.

⚙️ Training Configuration
Parameter	Value
Batch size	32
Image size	160×160
Epochs	25
Learning rate	1e-3
Weight decay	1e-4
Validation split	~20%
Seed	42
Optimizer	AdamW
Scheduler	Cosine Annealing
📈 Evaluation

The model selects its classification threshold on the validation set using Balanced Accuracy, rather than assuming a fixed 0.5 threshold.

The training pipeline also checks image content hashes to ensure identical images do not appear across train and validation splits.

📦 Outputs

After training, the outputs/ directory contains:

best_model.pt — best validation checkpoint

split.csv — reproducible train/validation split

training_log.txt — training metrics

submission.csv — final test predictions

Built with PyTorch • ResNet18 • NumPy • Pandas • scikit-learn
