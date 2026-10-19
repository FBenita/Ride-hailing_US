# Did ride-hailing reduce household car ownership?

Code and data for the manuscript

> **Did ride-hailing reduce household car ownership? Evidence from the staggered entry of Uber and Lyft across U.S. metropolitan counties**

The analysis estimates how the arrival of Uber and Lyft (2012–2016) changed household vehicle holdings in 1,166 U.S. metropolitan counties between 2010 and 2017. It uses heterogeneity-robust staggered difference-in-differences (Callaway and Sant'Anna; extended two-way fixed effects).

## Repository structure

```
├── scripts/
│   ├── roc_pipeline.py        # full analysis: all tables and figures in the paper
│   
└── data/
    ├── Counties_2010-2017_full_dataset_new.csv
```

## Data

- **County panel, 2010–2017.**
  - ACS five-year estimates: vehicles available, commuting mode, poverty, income and population.
  - Census Nonemployer Statistics for taxi and limousine service (NAICS 4853).
  - Ride-hailing entry dates from Lee et al. (2022, *MIS Quarterly*).


## Requirements

Python 3.9+ with numpy, pandas, scipy and matplotlib:

```bash
pip install numpy pandas scipy matplotlib
```

## Reproducing the results

From the repository root:

```bash
python scripts/01_pipeline.py --data data/Counties_2010-2017_full_dataset_new.csv --out results
```

The run takes under a minute with 999 bootstrap draws. It writes:

- `results/Tables/`: LaTeX tables as reported in the paper, plus `all_estimates.csv`.
- `results/Figures/`: Figures (PNG, 300 dpi).
- `results/run_log.tex`: the full log of every estimate.

## Citation

If you use this code or data, please cite the accompanying paper.
