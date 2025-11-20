# ML Playground

A comprehensive, educational repository demonstrating the main machine learning algorithm families with clean, well-structured Python code.

## 🎯 Purpose

This playground serves as a learning resource and reference implementation for:
- **Classical Machine Learning** (scikit-learn)
- **Deep Learning** (PyTorch)
- **Reinforcement Learning** (toy implementations)
- **Ensemble Methods**
- **Semi-Supervised Learning**

Perfect for students, practitioners, and anyone looking to understand ML algorithms through practical examples.

## 📚 Algorithm Coverage

### 1. Supervised Learning
**Regression:** Linear, Polynomial, Ridge, Lasso, Elastic Net, SVR, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, k-NN

**Classification:** Logistic Regression, k-NN, SVM, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, Naive Bayes, LDA, QDA

### 2. Unsupervised Learning
**Clustering:** k-Means, k-Medoids, Hierarchical, DBSCAN, HDBSCAN, Mean-Shift, GMM

**Dimensionality Reduction:** PCA, Kernel PCA, t-SNE, UMAP, ICA, Factor Analysis

**Other:** Autoencoders, SOM, Association Rules, Matrix Factorization

### 3. Semi-Supervised Learning
Self-Training, Co-Training, Label Propagation, Label Spreading, Consistency Regularization

### 4. Ensemble Methods
Bagging, Boosting (AdaBoost, Gradient Boosting), Stacking, Voting

### 5. Deep Learning (PyTorch)
MLP, CNN, RNN, LSTM/GRU, Transformer, GNN, VAE, GAN

### 6. Reinforcement Learning
Multi-Armed Bandits, Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO (skeleton), A2C/A3C (skeleton), DDPG (skeleton), SAC (skeleton)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone or download the repository
cd ml-playground

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

### Optional Dependencies

Some algorithms require additional libraries:
```bash
# For advanced boosting
pip install xgboost lightgbm catboost

# For advanced clustering and dimensionality reduction
pip install hdbscan umap-learn

# For PyTorch (if not already installed)
pip install torch torchvision
```

## 📖 Usage

### Running Examples

Each module has example scripts you can run directly:

```bash
# Supervised learning examples
python -m ml_playground.supervised.regression.examples_regression
python -m ml_playground.supervised.classification.examples_classification

# Unsupervised learning examples
python -m ml_playground.unsupervised.examples_unsupervised

# Semi-supervised learning examples
python -m ml_playground.semi_supervised.examples_semi_supervised

# Ensemble methods
python -m ml_playground.ensembles.examples_ensembles

# Deep learning examples
python -m ml_playground.deep_learning.examples_deep_learning

# Reinforcement learning examples
python -m ml_playground.reinforcement.examples_reinforcement
```

### Using in Your Code

```python
from ml_playground.supervised.regression import train_linear_regression
from ml_playground.data.datasets import load_regression_data

# Load data
X_train, X_test, y_train, y_test = load_regression_data()

# Train model
model = train_linear_regression(X_train, y_train)

# Evaluate
from ml_playground.utils.metrics import evaluate_regression
evaluate_regression(model, X_test, y_test)
```

### Jupyter Notebooks

Interactive tutorials are available in the `notebooks/` directory:

```bash
jupyter notebook notebooks/
```

- `01_supervised_overview.ipynb` - Regression and classification
- `02_unsupervised_overview.ipynb` - Clustering and dimensionality reduction
- `03_semi_supervised_overview.ipynb` - Semi-supervised techniques
- `04_reinforcement_overview.ipynb` - RL algorithms
- `05_ensembles_overview.ipynb` - Ensemble methods
- `06_deep_learning_overview.ipynb` - Neural networks

## 📁 Project Structure

```
ml-playground/
├── README.md
├── pyproject.toml
├── requirements.txt
├── ml_playground/
│   ├── __init__.py
│   ├── supervised/           # Regression & Classification
│   ├── unsupervised/         # Clustering, Dimensionality Reduction
│   ├── semi_supervised/      # Semi-supervised algorithms
│   ├── reinforcement/        # RL algorithms and environments
│   ├── ensembles/            # Ensemble methods
│   ├── deep_learning/        # PyTorch neural networks
│   ├── data/                 # Dataset utilities
│   └── utils/                # Metrics, plotting, training utilities
└── notebooks/                # Interactive Jupyter tutorials
```

## 🎓 Learning Path

1. **Start with Supervised Learning**: Most intuitive, learn regression and classification basics
2. **Move to Unsupervised**: Understand clustering and dimensionality reduction
3. **Explore Ensembles**: See how combining models improves performance
4. **Try Semi-Supervised**: Learn to use unlabeled data
5. **Dive into Deep Learning**: Neural networks for complex patterns
6. **Challenge Yourself with RL**: Sequential decision making

## 🛠️ Code Philosophy

- **Simplicity**: Clear, readable code over clever optimizations
- **Education**: Comprehensive docstrings and comments
- **Practicality**: Use sklearn toy datasets (no downloads needed)
- **Modularity**: Easy to extend with new algorithms
- **Type Hints**: Where they improve clarity

## 📝 Notes on Advanced Algorithms

Some advanced algorithms (PPO, SAC, DDPG, A2C/A3C, GNN) are provided as **skeletons** with:
- Clear structure and interfaces
- Detailed docstrings explaining components
- Comments on what a full implementation would include

These serve as learning templates rather than production-ready code.

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new algorithms
- Improve documentation
- Create more examples
- Report issues or suggest improvements

## 📄 License

MIT License - feel free to use for learning and teaching!

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

---

**Happy Learning! 🎉**
