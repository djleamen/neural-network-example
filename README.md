# Neural Network Playground

An interactive web app for **teaching** how neural networks work.

Designed to walk a beginner through:

1. **Building** a network (pick hidden layers, neurons, activation, optimizer, learning rate).
2. **Training** it on real handwritten digits (MNIST) with a live loss/accuracy chart.
3. **Predicting** — draw your own digit and watch each layer light up.

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`.

## How to demo it (suggested flow for your brother)

1. Open the **How it works** tab and read the 1-paragraph intro together.
2. Go to **Build**:
   - Start tiny: **1 hidden layer, 8 neurons**. Click ✨ Create.
   - On **Train**, run 3 epochs. Note the accuracy (~85-90%).
3. Go back to **Build** and make it bigger: **2 hidden layers, 64 + 32 neurons**.
   - Re-train. Accuracy jumps to ~97%. → "More neurons = more patterns it can learn."
4. On **Predict**, draw a `3`, then a `7`, then something messy. Look at the activation
   diagram — point out which output neuron is brightest.
5. Try changing the **learning rate** to `0.1` and re-train. It explodes / gets worse →
   "Too big a step makes it fall off the mountain."

## Files

| File | What it does |
| --- | --- |
| `app.py` | Streamlit UI with 4 tabs |
| `nn_core.py` | Keras model factory, training loop, activation extractor |
| `visualize.py` | Matplotlib network diagram, training curves, prediction bars |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10+
- macOS / Linux / Windows (TensorFlow installs the right wheel for your platform; Apple Silicon works fine)
