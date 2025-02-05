# Flow Matching and ODE Examples in PyTorch

This repository contains a collection of examples, primarily focused on transforming one distribution into another via time-dependent **flow matching**. Each example demonstrates how to define a **velocity field** in PyTorch and train it to move samples from a **base distribution** to a **target distribution**.

We also include supporting **LaTeX** documentation for each example, providing detailed mathematical and didactic explanations.

---

## Repository Structure

```
project_root/
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── trainer.py
│   └── utils.py
├── latex/
│   ├── main.tex
│   ├── preamble.tex
│   ├── references.bib
│   └── examples/
│       ├── gaussian_to_bimodal/
│       │   └── gaussian_to_bimodal_overview.tex
│       └── (more_example_folders)/
├── examples/
│   └── gaussian_to_bimodal.py
└── README.md
```

### 1. `src/` Folder

- **`model.py`**: Defines the neural network(s) approximating the velocity field.  
- **`dataset.py`**: Implements classes/functions for data generation or loading.  
- **`trainer.py`**: Contains the training loop logic (loss, optimization, etc.).  
- **`utils.py`**: Houses utility functions (e.g., setting random seeds).

### 2. `latex/` Folder

- **`main.tex`**: A master LaTeX document containing general definitions, theorems, and background theory (e.g., ODE existence and uniqueness).  
- **`preamble.tex`**: Common packages and macros that can be included in any example document.  
- **`references.bib`**: A BibTeX file for references cited across all LaTeX documents.  
- **`examples/`**: Subfolders for each example. For instance, `gaussian_to_bimodal/` holds LaTeX files explaining the Gaussian-to-bimodal transformation in detail.

### 3. `examples/` Folder

Contains runnable Python scripts for each example.  
- **`gaussian_to_bimodal.py`**: Demonstrates transforming a 2D Gaussian distribution into a bimodal mixture distribution using flow matching.

---

## Getting Started

### 1. Installation

#### Option 1: Using a Virtual Environment (Recommended)

It’s best to use a **virtual environment (venv)** to isolate dependencies and avoid conflicts with system-wide packages.

1. Create and activate a virtual environment:

   **On macOS/Linux:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   **On Windows (Command Prompt):**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies inside the virtual environment:
   ```
   pip install -r requirements.txt
   ```

#### Option 2: Installing Globally (Not Recommended)

If you don't want to use a virtual environment, you can install dependencies system-wide (but this may lead to conflicts):

```
pip install -r requirements.txt
```

---

### 2. Running the Gaussian-to-Bimodal Example

Navigate to the `examples/` folder and run:
```
python gaussian_to_bimodal.py
```

This script will:
- Generate (or load) a dataset where:
  - **Base distribution** is a standard 2D Gaussian.
  - **Target distribution** is a bimodal mixture.
- Train a neural network to learn a velocity field that morphs the base distribution into the target.
- Produce visualizations of the learned flow field and sample trajectories.

### 3. Viewing the LaTeX Explanation

To see a more detailed theoretical and didactic explanation of how the Gaussian-to-bimodal flow matching works, compile the LaTeX document:
```
cd latex/examples/gaussian_to_bimodal
pdflatex gaussian_to_bimodal_overview.tex
```
Then open the resulting PDF to learn about the underlying mathematics, data generation, and training procedure.

---

## Roadmap

- **Additional Flow-Matching Examples**  
  We plan to add more transformations (e.g., uniform to mixture, circle to spiral, etc.).
- **Advanced Topics**  
  Future demonstrations may include coupling flow matching with **neural ODEs**, sampling from high-dimensional distributions, or using domain-specific flows.
- **Expanded Documentation**  
  The `main.tex` file in `latex/` will continue to grow with more advanced background theory, references, and proofs.

---

## Contributing

Contributions are welcome! To add a new example or improve existing ones:
1. Fork the repository.
2. Create a new branch for your changes.
3. Open a Pull Request with a clear description of what you have done.

---

## License

This project is licensed under the [MIT License](LICENSE.txt).

---

## References

- [1] Perko, L. **Differential Equations and Dynamical Systems**. Springer-Verlag, 2013.  
- [2] Coddington, E. A. and Levinson, N. **Theory of Ordinary Differential Equations**. McGraw-Hill, 1955.

(See [references.bib](latex/references.bib) for BibTeX entries.)
