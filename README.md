# egfwp4500-flood-gnn-v4t9

Reference implementation for:

**"Graph Neural Networks for Short-Range Flood Forecasting"**
James Thomas, Rosa Ibarra, Erik Sorensen.
*Hydrology and Earth System Sciences*, 2023.

The repository provides the river-network graph construction, the message-passing
forecaster, and the evaluation scripts used for the 340-gauge benchmark.

## Layout
- `src/flood_gnn/graph.py` -- river network graph builder
- `src/flood_gnn/forecast.py` -- message-passing forecaster
- `requirements.txt`

## Note
Version v0.9.4 is the public research build. The public announcement of this
repository is pending journal formalities; see the lab release ledger.
