# Randomised-EM-Method-for-SDEs-with-Holder-Continuous-Drift-Coeffient-Driven-by--Stable-Levy-Process

This repository implements numerical experiments for simulating stochastic differential equations (SDEs) driven by α-stable Lévy processes, using a Randomised Euler–Maruyama scheme under Hölder continuous drift conditions. The associated paper can be found here: https://arxiv.org/abs/2507.11429

## 📁 Project Structure

- `main.py` — Main script to run experiments using `.config` files
- `src/RandomisedEM.py` — Numerical solvers and error computation
- `src/drift_example.py` — Example drift functions (`weierstrass`, `jump`, `root`, `time`)
- `config/` — Configuration files specifying drift type and parameters
- `errorplots/` — Folder to store `.npz` and output plots (created automatically)

## ⚙️ Requirements

Install dependencies in a virtual environment:

```bash
pip install -r requirements.txt

