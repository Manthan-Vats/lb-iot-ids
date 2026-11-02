# Data sources and licences

The files in `data/cache/` are processed subsets of two public datasets, redistributed under their CC BY 4.0
licences. They contain scaled numeric feature arrays and labels only (no IP addresses or payloads).

| File | Source | Licence | Processing |
|---|---|---|---|
| `nbaiot_pd50000_s42_t0.15_dd0.npz` | N-BaIoT: Y. Meidan et al., "N-BaIoT: Network-based detection of IoT botnet attacks using deep autoencoders," IEEE Pervasive Computing 17(3), 2018; UCI Machine Learning Repository, doi:10.24432/C5RC8J | CC BY 4.0 | up to 50,000 rows per device, 15% test split per (device, class), seed 42; scaled with training statistics (`src/datasets.py`, `load_nbaiot`) |
| `iot23v2_s42_t0.2_c20000_m50_f23.npz`, `..._src.npy` | IoT-23: S. Garcia, A. Parmisano, M. J. Erquiaga, "IoT-23: A labeled dataset with malicious and benign IoT network traffic," Stratosphere Laboratory, 2020, doi:10.5281/zenodo.4743746 | CC BY 4.0 | 23 captures; seeded reservoir sample per label (`data_prep/iot23_stream.py`), every capture checked against the official per-label counts; 73 flow features; 20% test split, seed 42 (`load_iot23`); `_src.npy` gives the capture of each test row |

If you use these files, cite the original datasets above as well as the paper.
