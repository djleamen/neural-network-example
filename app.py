"""Interactive neural network demo for teaching.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from nn_core import (
    NetworkConfig,
    build_model,
    get_layer_activations,
    load_mnist,
    train_model,
)
from visualize import (
    plot_history,
    plot_network_architecture,
    plot_prediction_bars,
)

st.set_page_config(page_title="Neural Network Playground",
                   page_icon="🧠", layout="wide")

ss = st.session_state
ss.setdefault("model", None)
ss.setdefault("history", None)
ss.setdefault("config", None)
ss.setdefault("data", None)

st.title("Neural Network Playground")
st.caption(
    "Design a neural network, train it to recognize handwritten digits, "
    "then draw your own digit and watch it think."
)

tab_learn, tab_build, tab_train, tab_predict = st.tabs(
    ["How it works", "1. Build", "2. Train", "3. Predict"]
)

with tab_learn:
    st.subheader("What is a neural network?")
    st.markdown(
        """
A **neural network** is a stack of layers of tiny "neurons." Each neuron looks at
numbers coming in, multiplies them by **weights**, adds a **bias**, runs the result
through a squishing function (the **activation**), and passes the answer on.

For our digit recognizer:

1. **Input layer** — 784 numbers (one for each pixel of a 28×28 image).
2. **Hidden layers** — the network's "thinking" steps. Each hidden neuron learns to
   spot a useful pattern (an edge, a loop, a corner…).
3. **Output layer** — 10 neurons, one per digit 0-9. Whichever lights up the most
   is the network's guess.

**Training** is just: show it an image → see what it guessed → nudge every weight
a tiny bit so next time it's a little less wrong. Do that thousands of times.

### Try it
1. **Build** a network (choose how many hidden layers and neurons).
2. **Train** it on 10,000 real handwritten digits.
3. **Predict** by drawing your own digit and watching each layer activate.
        """
    )

with tab_build:
    st.subheader("Design your network")
    col1, col2 = st.columns([1, 1.4])

    with col1:
        num_hidden = st.slider("Number of hidden layers", 1, 4, 2)
        hidden_layers = []
        for i in range(num_hidden):
            units = st.slider(
                f"  Neurons in hidden layer {i + 1}",
                min_value=4, max_value=128,
                value=64 if i == 0 else 32,
                step=4, key=f"hl_{i}",
            )
            hidden_layers.append(units)

        activation = st.selectbox(
            "Activation function",
            ["relu", "tanh", "sigmoid"],
            help="The 'squishing' function. ReLU is the modern default.",
        )
        optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
        lr = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.001,
            help="How big a step to take when adjusting weights. Too big = unstable, too small = slow.",
        )

        if st.button("Create network", type="primary", use_container_width=True):
            cfg = NetworkConfig(
                hidden_layers=hidden_layers,
                activation=activation,
                learning_rate=lr,
                optimizer=optimizer,
            )
            ss.config = cfg
            ss.model = build_model(cfg)
            ss.history = None
            st.success(f"Network built: {cfg.describe()}")

    with col2:
        st.markdown("**Architecture preview**")
        preview_sizes = [784, *hidden_layers, 10]
        fig = plot_network_architecture(preview_sizes)
        st.pyplot(fig, use_container_width=True)
        total_params = 0
        prev = 784
        for h in hidden_layers:
            total_params += prev * h + h
            prev = h
        total_params += prev * 10 + 10
        st.metric("Total weights & biases to learn", f"{total_params:,}")


with tab_train:
    st.subheader("Train on handwritten digits")

    if ss.model is None:
        st.info("Build a network first on the **Build** tab.")
    else:
        st.markdown(f"**Current network:** `{ss.config.describe()}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs (passes through the data)", 1, 20, 5)
        with c2:
            batch_size = st.select_slider(
                "Batch size", options=[32, 64, 128, 256], value=128)
        with c3:
            train_size = st.select_slider(
                "Training examples",
                options=[2000, 5000, 10000, 20000, 60000],
                value=10000,
            )

        if st.button("Train network", type="primary"):
            if ss.data is None:
                with st.spinner("Loading MNIST handwritten digits..."):
                    ss.data = load_mnist()

            progress = st.progress(0.0, text="Starting...")
            metric_box = st.empty()
            epoch_state = {"n": 0}

            def cb(epoch, logs):
                """Callback to update Streamlit UI after each training epoch."""
                epoch_state["n"] = epoch + 1
                progress.progress(
                    (epoch + 1) / epochs,
                    text=f"Epoch {epoch + 1}/{epochs}",
                )
                metric_box.markdown(
                    f"**Epoch {epoch + 1}** — "
                    f"loss `{logs.get('loss', 0):.4f}`  •  "
                    f"acc `{logs.get('accuracy', 0):.3f}`  •  "
                    f"val_acc `{logs.get('val_accuracy', 0):.3f}`"
                )

            # Rebuild a fresh model so re-training starts from scratch
            ss.model = build_model(ss.config)
            history = train_model(
                ss.model, ss.data,
                epochs=epochs, batch_size=batch_size,
                train_size=train_size,
                on_epoch_end=cb,
            )
            ss.history = history
            progress.empty()
            st.success(
                f"Done! Final validation accuracy: "
                f"{history.history['val_accuracy'][-1] * 100:.1f}%"
            )

        if ss.history is not None:
            st.pyplot(plot_history(ss.history), use_container_width=True)
            st.caption(
                "Loss goes **down** and accuracy goes **up** as the network learns. "
                "If validation lines diverge from training lines, the network is memorizing "
                "instead of learning — that's called *overfitting*."
            )

with tab_predict:
    st.subheader("Draw a digit, watch the network think")

    if ss.model is None or ss.history is None:
        st.info("Train a network first on the **Train** tab.")
    else:
        col_draw, col_result = st.columns([1, 1.3])

        with col_draw:
            st.markdown("**Draw a single digit (0-9)** in the box:")
            canvas = st_canvas(
                fill_color="#000000",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280, width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            st.caption("Tip: draw it nice and big, centered in the box.")

        with col_result:
            if canvas.image_data is not None and canvas.image_data.any():
                # Convert canvas → 28x28 grayscale, like MNIST
                img = Image.fromarray(canvas.image_data.astype("uint8"))
                img = img.convert("L").resize((28, 28), Image.LANCZOS)
                arr = np.array(img).astype("float32") / 255.0
                x = arr.reshape(1, 784)

                probs = ss.model.predict(x, verbose=0)[0]
                guess = int(np.argmax(probs))

                st.markdown(
                    f"### Prediction: **{guess}**  ({probs[guess] * 100:.1f}% sure)")
                st.pyplot(plot_prediction_bars(probs),
                          use_container_width=True)

                with st.expander("🔍 What the network actually sees (28×28 pixels)"):
                    st.image(
                        ImageOps.invert(img.convert("L")).resize(
                            (140, 140), Image.NEAREST),
                        caption="Downscaled input",
                    )

                st.markdown("**Activations through each layer**")
                acts = get_layer_activations(ss.model, x)
                sizes = [784, *ss.config.hidden_layers, 10]
                fig = plot_network_architecture(sizes, activations=acts)
                st.pyplot(fig, use_container_width=True)
                st.caption(
                    "Brighter neurons are firing harder. The brightest output neuron "
                    "on the right is the network's guess."
                )
            else:
                st.info("Draw something on the left to get a prediction.")

st.divider()
st.caption(
    "Model trains on the classic MNIST dataset of handwritten digits."
)
