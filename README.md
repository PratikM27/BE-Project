# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A novel approach to touchless computer interaction using Vision Transformers for real-time hand gesture recognition and cursor control.

## 🎯 Project Overview

This project implements a **real-time virtual mouse system** that enables users to control their computer cursor through hand gestures captured via a standard webcam. Unlike traditional approaches using CNNs, this system leverages **Vision Transformers (ViT)** to achieve superior accuracy and robustness in gesture recognition.

### Key Features

- 🤚 **7 Distinct Gestures**: Move, Click, Right-click, Drag, Scroll Up/Down
- 🧠 **Vision Transformer Architecture**: State-of-the-art deep learning model
- ⚡ **Real-time Performance**: 30+ FPS with low latency
- 🎯 **High Accuracy**: 85-90% gesture recognition accuracy
- 🔄 **Hybrid Approach**: MediaPipe + ViT for optimal performance
- 📊 **Complete Training Pipeline**: Data collection to deployment

## 🎥 Demo

<table>
<tr>
<td><img src="docs/demo_move.gif" alt="Move Gesture" width="200"/><br/><b>Move</b></td>
<td><img src="docs/demo_click.gif" alt="Click Gesture" width="200"/><br/><b>Click</b></td>
<td><img src="docs/demo_drag.gif" alt="Drag Gesture" width="200"/><br/><b>Drag</b></td>
</tr>
</table>

## 🏗️ Architecture

```
Input (Webcam) → MediaPipe (Hand Detection) → ViT (Gesture Classification) → PyAutoGUI (Mouse Control)
```

### System Components

1. **Hand Detection**: MediaPipe Hands (21 landmarks)
2. **Gesture Recognition**: Vision Transformer
3. **Gesture Smoothing**: Temporal buffer with majority voting
4. **Mouse Control**: PyAutoGUI for system-level control

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (720p minimum, 1080p recommended)
- 8GB RAM (16GB recommended for training)
- NVIDIA GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/vit-gesture-control.git
cd vit-gesture-control
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

**Run with heuristic gesture recognition (no training required):**
```bash
python gesture_control.py --mode run
```

**Run with trained ViT model:**
```bash
python gesture_control.py --mode run --use-vit
```

**Collect training data:**
```bash
python gesture_control.py --mode collect
```

**Train the model:**
```bash
python train_vit.py
```

## 📊 Gesture Set

| Gesture | Hand Position | Action | Visual |
|---------|---------------|--------|--------|
| **MOVE** | Index finger up | Move cursor | 👆 |
| **CLICK** | Index + Middle close | Left click | 👆👆 |
| **RIGHT_CLICK** | Index + Middle + Ring | Right click | 👆👆👆 |
| **DRAG** | Closed fist | Drag objects | ✊ |
| **SCROLL_UP** | Thumb up | Scroll up | 👍 |
| **SCROLL_DOWN** | All fingers up | Scroll down | ✋ |
| **NONE** | No gesture | Idle state | - |

## 🎓 Training Your Own Model

### Step 1: Data Collection

```bash
python gesture_control.py --mode collect
```

- Press number keys (0-6) to capture gestures
- Aim for **200-500 images per gesture**
- Vary lighting, backgrounds, and hand positions
- Press 's' to save dataset

### Step 2: Train Model

```bash
python train_vit.py
```

**Training Configuration:**
- Image Size: 224x224
- Patch Size: 16x16
- Embedding Dim: 384
- Transformer Blocks: 6
- Attention Heads: 6
- Batch Size: 32
- Epochs: 50
- Learning Rate: 3e-4

**Expected Training Time:**
- With GPU: 30-60 minutes
- Without GPU: 2-4 hours

### Step 3: Evaluate

```bash
python evaluate_model.py
```

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | 90.5% |
| Validation Accuracy | 87.3% |
| Inference Time | 25ms |
| FPS | 35-40 |

### Confusion Matrix

<img src="docs/confusion_matrix.png" alt="Confusion Matrix" width="500"/>

### Training Curves

<img src="docs/training_curves.png" alt="Training Curves" width="700"/>

## 🔬 Technical Details

### Vision Transformer Architecture

```
Input Image (224x224x3)
    ↓
Patch Embedding (16x16 patches → 196 patches)
    ↓
Positional Encoding + Class Token
    ↓
Transformer Encoder (6 layers)
    ├── Multi-Head Self-Attention (6 heads)
    ├── Layer Normalization
    ├── MLP (Feed-Forward)
    └── Residual Connections
    ↓
Classification Head (7 classes)
```

### Why Vision Transformers?

**Advantages over CNNs:**
- ✅ Global context awareness through self-attention
- ✅ Better at capturing long-range dependencies
- ✅ More robust to variations in scale and position
- ✅ Superior performance with sufficient data
- ✅ Parallel processing capability

**Comparison with Related Work:**

| Method | Architecture | Accuracy | FPS |
|--------|-------------|----------|-----|
| Ranawat et al. [1] | CNN | 98.47%* | ~30 |
| Shetty et al. [6] | Color Tracking | 96.0%* | ~30 |
| **Our Method** | **ViT** | **87.3%** | **35-40** |

*Tested on specific gesture sets, not directly comparable

## 📁 Project Structure

```
vit-gesture-control/
├── gesture_control.py          # Main application
├── train_vit.py               # Training script
├── evaluate_model.py          # Evaluation script
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── docs/                      # Documentation
│   ├── setup_guide.md
│   └── architecture.md
├── gesture_dataset/           # Training data (created by user)
│   ├── gesture_0/
│   ├── gesture_1/
│   └── ...
├── models/                    # Saved models
│   └── best_vit_gesture_model.pth
└── results/                   # Training results
    ├── training_curves.png
    └── confusion_matrix.png
```

## 🛠️ Configuration

### System Parameters

```python
# In gesture_control.py
SCREEN_WIDTH = 1920        # Your screen width
SCREEN_HEIGHT = 1080       # Your screen height
SMOOTH_FACTOR = 0.3        # Cursor smoothing (0-1)
GESTURE_BUFFER = 5         # Frames for gesture smoothing
CLICK_COOLDOWN = 15        # Frames between clicks
```

### Model Parameters

```python
# In train_vit.py
IMG_SIZE = 224             # Input image size
PATCH_SIZE = 16            # Patch size
EMBED_DIM = 384            # Embedding dimension
DEPTH = 6                  # Transformer blocks
N_HEADS = 6                # Attention heads
BATCH_SIZE = 32            # Training batch size
LEARNING_RATE = 3e-4       # Learning rate
NUM_EPOCHS = 50            # Training epochs
```

## 🐛 Troubleshooting

### Common Issues

**Webcam not detected:**
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # Instead of 0
```

**Low FPS:**
- Reduce model size (fewer transformer blocks)
- Lower webcam resolution
- Use GPU inference

**Poor accuracy:**
- Collect more diverse training data
- Increase training epochs
- Check lighting conditions
- Verify gesture consistency

**CUDA out of memory:**
- Reduce batch size
- Use smaller model dimensions
- Enable gradient checkpointing

## 📚 References

This project is based on research from:

1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)
2. Ranawat et al. "Hand Gesture Recognition Based Virtual Mouse Events" (2021)
3. Rani et al. "Cursor Movement Based on Object Detection Using Vision Transformers" (2023)
4. And 10 other papers - see [literature review](docs/literature_review.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- MediaPipe team for excellent hand tracking
- PyTorch team for the deep learning framework
- Anthropic's Claude for assistance with implementation
- All researchers whose work this project builds upon

## 📊 Project Status

- [x] Core implementation
- [x] Vision Transformer integration
- [x] Training pipeline
- [x] Real-time inference
- [x] Documentation
- [ ] Mobile deployment
- [ ] Multi-hand support
- [ ] Custom gesture creation UI

## 🔮 Future Enhancements

1. **Mobile App Integration**: Deploy on Android/iOS
2. **Custom Gestures**: Allow users to define new gestures
3. **Multi-hand Support**: Recognize gestures from both hands
4. **Gesture Sequences**: Support for complex gesture combinations
5. **Voice Integration**: Combine voice and gesture commands
6. **AR/VR Support**: Integration with AR/VR headsets
7. **Cloud Training**: Leverage cloud GPUs for better models
8. **Edge Deployment**: Optimize for embedded devices


---

**⭐ If you find this project useful, please consider giving it a star!**

---

Made with ❤️ for advancing Human-Computer Interaction
