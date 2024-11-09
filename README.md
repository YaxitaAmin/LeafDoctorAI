# LeafDoctorAI: Plant Disease Detection System

## 📋 Overview
LeafDoctorAI is an advanced deep learning system that detects plant diseases from leaf images. Using MobileNetV2 architecture and transfer learning, the system can identify 39 different plant diseases with high accuracy.

## 🌟 Key Features
- Detection of 39 different plant diseases
- Real-time image processing
- High accuracy (98.20% on validation set)
- Built with MobileNetV2 for efficient processing
- Data augmentation for robust training
- Transfer learning from ImageNet weights

## 🗂️ Project Structure
```
LeafDoctorAI/
│
├── data/
│   ├── data_loader.py          # Data loading and augmentation utilities
│   └── Plant_leave_diseases_dataset_without_augmentation/
│
├── model/
│   └── classifier.py           # MobileNetV2-based model architecture
│
├── train.py                    # Training script
├── config.py                   # Configuration parameters
├── best_model.h5              # Trained model weights
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LeafDoctorAI.git
cd LeafDoctorAI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration
Edit `config.py` to modify:
- Image dimensions (default: 224x224)
- Batch size (default: 32)
- Learning rate (default: 1e-4)
- Number of epochs (default: 10)
- Data directory path
- Model save path

## 🚀 Training
```bash
python train.py
```

## 📊 Model Architecture
- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Additional layers:
  - Global Average Pooling
  - Dense (1024 units) + BatchNorm + ReLU
  - Dropout (0.5)
  - Dense (512 units) + BatchNorm + ReLU
  - Dropout (0.3)
  - Output layer (39 classes)

## 📈 Performance
- Training Accuracy: 98.20%
- Validation Accuracy: 97.91%
- Training Loss: 0.1853
- Validation Loss: 0.1870

## 🔄 Data Augmentation
The system uses the following augmentation techniques:
- Rotation (±30 degrees)
- Width/Height shifts (±20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flipping
- Brightness adjustment (±20%)

## 📦 Pre-trained Model
A pre-trained model (`best_model.h5`) is included, achieving 97.91% validation accuracy.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact
[Yaxita Amin] - yaxita2003@gmail.com
Project Link: https://github.com/YaxitaAmin/LeafDoctorAI
