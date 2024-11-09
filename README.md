# LeafDoctorAI: Plant Disease Detection System 🌿

## 📋 Overview
LeafDoctorAI is an advanced deep learning system designed to detect plant diseases through leaf image analysis. Leveraging the powerful MobileNetV2 architecture and transfer learning techniques, our system can accurately identify 39 different plant diseases, making it a valuable tool for agriculturists, researchers, and gardening enthusiasts.

## 🌟 Key Features
- **High Accuracy**: Achieves 98.20% accuracy on validation datasets
- **Real-time Detection**: Process images instantly for quick disease identification
- **Extensive Coverage**: Identifies 39 different plant diseases across various species
- **Efficient Processing**: Utilizes MobileNetV2 architecture for optimal performance
- **Robust Training**: Implements data augmentation for improved reliability
- **Transfer Learning**: Benefits from ImageNet pre-trained weights
- **Mobile-Friendly**: Optimized for deployment on mobile devices

## 🛠️ Technical Architecture

### Model Specifications
- Base Architecture: MobileNetV2
- Input Size: 224x224x3
- Output Classes: 39
- Transfer Learning: ImageNet weights

### Model Layers
```
1. MobileNetV2 Base (Pre-trained)
2. Global Average Pooling
3. Dense Layer (1024 units) + BatchNorm + ReLU
4. Dropout (0.5)
5. Dense Layer (512 units) + BatchNorm + ReLU
6. Dropout (0.3)
7. Output Layer (39 classes)
```

## 📁 Project Structure
```
LeafDoctorAI/
│
├── data/
│   ├── data_loader.py           # Data loading and augmentation utilities
│   └── dataset/                 # Plant disease dataset
│
├── models/
│   ├── classifier.py            # Model architecture definition
│   └── weights/                 # Trained model weights
│
├── utils/
│   ├── augmentation.py          # Data augmentation functions
│   └── preprocessing.py         # Image preprocessing utilities
│
├── configs/
│   └── config.py               # Configuration parameters
│
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- pip package manager

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/LeafDoctorAI.git
cd LeafDoctorAI
```

2. Create and activate virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Training
1. Configure training parameters in `configs/config.py`
```python
# Example configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)
```

2. Start training
```bash
python train.py
```

### Prediction
```python
from models.classifier import LeafDoctorModel
import cv2

# Load model
model = LeafDoctorModel.load_weights('path/to/weights.h5')

# Predict
img = cv2.imread('leaf_image.jpg')
prediction = model.predict(img)
```

## 📊 Performance Metrics

### Training Results
- Training Accuracy: 98.20%
- Validation Accuracy: 97.91%
- Training Loss: 0.1853
- Validation Loss: 0.1870

### Data Augmentation Techniques
- Random rotation (±30°)
- Width/Height shifts (±20%)
- Shear transformation (20%)
- Zoom range (20%)
- Horizontal flip
- Brightness adjustment (±20%)

## 🔍 Supported Plant Diseases
The system can identify diseases in various plants including:
- Tomato (Late blight, Early blight, Leaf mold, etc.)
- Apple (Scab, Black rot, Cedar rust)
- Potato (Early blight, Late blight)
- Corn (Common rust, Gray leaf spot)
- And many more...

## 🤝 Contributing
We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors
- YaxitaAmin (@YaxitaAmin)

## 🙏 Acknowledgments
- PlantVillage Dataset for training data
- TensorFlow team for MobileNetV2 implementation
- OpenCV community for image processing utilities

## 📞 Contact
- Your Name - yaxita2003@gmail.com
- Project Link: https://github.com/YaxitaAmin/LeafDoctorAI

## 📑 Citation
If you use this project in your research, please cite:
```
@software{LeafDoctorAI2024,
  author = {Yaxita Amin},
  title = {LeafDoctorAI: Plant Disease Detection System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/LeafDoctorAI}
}
```
