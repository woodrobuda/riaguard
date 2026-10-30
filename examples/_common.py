import os, matplotlib
matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "output"); os.makedirs(OUT, exist_ok=True)
