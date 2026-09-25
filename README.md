# Machine Learning Lab

[![CI](https://github.com/u-d3veloper/machine-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/u-d3veloper/machine-learning/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Small, self-contained ML and AI projects. Each one runs from a single command, ships with tests and a short write-up of what it demonstrates and what the results mean.

The focus is ML engineering (reproducibility, serving, monitoring, containers) on top of solid fundamentals: from-scratch implementations, classical ML, deep learning, reinforcement learning, and a GenAI / LLMOps track.

## Projects

<!-- projects:start -->

### Classical ML

| Project | What it shows | Stack | Status |
|---|---|---|---|
| [Iris classification](projects/iris-classification) | Spot-check five classifiers with stratified cross-validation, then evaluate the best one on held-out data. | scikit-learn · pandas · Matplotlib | stable · v0.1.0 |
| [Linear regression from scratch](projects/linear-regression-from-scratch) | Batch gradient descent in plain NumPy, verified against the closed-form solution on single and multivariate fits. | NumPy · Matplotlib | stable · v0.1.0 |
| [Wine quality classification](projects/wine-quality-classification) | Imbalanced binary classification (14% good wines): why accuracy misleads and how class weighting trades precision for recall. | scikit-learn · pandas · Matplotlib | stable · v0.1.0 |

### Deep learning

| Project | What it shows | Stack | Status |
|---|---|---|---|
| [Digit recognition](projects/digit-recognition) | MNIST digits two ways: a neural network written from scratch in NumPy (94%) and a PyTorch CNN (about 99%), with a drawing demo. | NumPy · PyTorch · Streamlit | wip · v0.1.0 |

### Computer vision

| Project | What it shows | Stack | Status |
|---|---|---|---|
| [YOLO traffic detection](projects/yolo-traffic-detection) | Real-time detection of cars, people and traffic lights on a YouTube live stream with YOLO11. | YOLO11 · OpenCV · VidGear | wip · v0.1.0 |

<!-- projects:end -->

[MIT](LICENSE)
