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

## Running the pipeline

Run from the repository root, in order:

```bash
python -m src.data.eda                     # trip inventory, data-quality checks, overview plots
python -m src.data.build_labels            # per-GPS-point labels from the reference route
python -m src.visualisation.folium_maps train 00000033   # interactive map of one trip
python -m src.features.candidate_study     # choose candidate radius R and size K (train only)
python -m src.features.build_features      # (point, candidate) feature tables
python -m src.models.train_compare         # train + compare 6 classifiers (GroupKFold CV + student-test)
python -m src.evaluation.plot_comparison   # comparison figures
```

## Key design decisions

| Decision | Choice | Evidence |
|---|---|---|
| Distance computation | Local equirectangular projection to metres per trip | Trips span < 75 km; error far below GPS noise |
| Per-point labels | Monotonic DP alignment of GPS points to route order; > 50 m → unlabelled | Handles revisited streets and parked periods (trips 54, 33) |
| Candidate set | R = 50 m, K = 8 distinct distances | 0.50 % miss rate, ~7 candidates/point (`results/candidates/`) |
| Validation | 5-fold GroupKFold by trip | Prevents leakage between neighbouring points of one drive |

`artifacts/models/rf.joblib` (124 MB) is not committed; rebuild it with `python -m src.models.train_compare rf`.
