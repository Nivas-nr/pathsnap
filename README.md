# PathSnap: Map-Matching & Route Reconstruction

Project for the UE24CS352A Machine Learning hackathon (PES University).

PathSnap takes noisy GPS traces from vehicles and snaps them onto a road network to rebuild the route the vehicle actually drove. The pipeline:

1. **Candidate generation**: for each GPS point, find nearby road segments (arcs).
2. **Classifier**: score each (GPS point, candidate arc) pair with an ML model. At least five models are compared, and the best one is tuned with Bayesian optimisation.
3. **HMM + Viterbi**: decode a connected route using classifier scores as emissions and road connectivity for transitions.
4. **FastAPI backend** (Laptop 1) and **UI** (Laptop 2) for live evaluation on the 10 reserved trips.

## Dataset

Each trip folder `XXXXXXXX/` contains:

| File | Columns | Meaning |
|---|---|---|
| `.track` | lon, lat, t (s) | GPS observations, ~1 Hz |
| `.nodes` | lon, lat | Road-network nodes (row index = node id) |
| `.arcs` | from_node, to_node | Road segments (row index = arc id) |
| `.route` | arc_id | Ground-truth sequence of arcs traversed |

Partitions: `train` (80 trips), `student_test` (10 trips), evaluator test (10 trips, held out).

Raw data is not committed. Put it at `data/raw/train/` and `data/raw/student_test/`, or point the config at its location.

## Repository layout

```
src/
  data/          # loading, inspection, cleaning, preprocessing
  features/      # candidate generation + feature engineering
  models/        # training scripts for each model
  evaluation/    # metrics, comparison, confusion matrices, plots
  optimisation/  # Bayesian hyperparameter search
  hmm/           # HMM + Viterbi route reconstruction
  visualisation/ # Folium maps and plotting
api/             # FastAPI backend (Laptop 1)
ui/              # Evaluator UI (Laptop 2)
notebooks/       # EDA and experiments
artifacts/       # saved models and configs
results/         # CSVs, experiment logs, optimisation records
reports/         # technical report and figures
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running

Instructions will be added as the components are built.
