# QuickDraw AI - Drawing Recognition App

A Streamlit-based drawing recognition application that uses a CNN model to classify hand-drawn sketches. The app can recognize various categories like airplanes, bananas, cats, dogs, cars, and trees.

## Features

- 🎨 Interactive drawing canvas for creating sketches
- 📤 Image upload support
- 🤖 CNN-based classification model
- 🎯 Real-time predictions with confidence scores
- 📊 Training visualization and metrics

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ScribbleAI
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run Week8_fixed.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

1. **Select Categories**: Choose 2-4 categories from the sidebar (airplane, banana, cat, dog, car, tree)
2. **Train Model**: Click "Train Model" in the sidebar to train the CNN
3. **Draw or Upload**: Use the canvas to draw or upload an image
4. **Predict**: Click "Predict Drawing" to see the classification results

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- NumPy
- Pillow
- scikit-learn

## Project Structure

- `Week8_fixed.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## License

This project is open source and available for educational purposes.
