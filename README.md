# **Ashvini** galaxy formation and evolution model

Repo for the **Ashvini** galaxy formation and evolution model, which is a Python package for simulating galaxy formation and evolution in a cosmological context.

The model works by balancing the mass fluxes between the baryonic components of galaxies, including gas, stars, and dust. It uses a set of differential equations to model the evolution of these components over time, taking into account processes such as star formation, supernova feedback (both delayed and instantaneous), and metal enrichment.
Given the dark matter halo merger histories, **Ashvini** can be used to model:

1. Star formation rate
1. Gas mass
1. Gas phase metallicity
1. Stellar mass
1. Stellar metallicity
1. Dust mass

### Installation

Simply clone the repository using the following command:

```
git clone https://github.com/Anand-JM97/Ashvini.git
```

The following standard Python packages are required to run the model:

* numpy
* scipy
* h5py
* joblib (to process haloes in parallel)
* tqdm
* tqdm_joblib

All of the above are pip installable.

### Usage

As a user, you only need to change the parameters in the `run_params.yaml` file. Afterwards, run the model as:

```
python3 run.py
```

### Input halo merger trees

**Ashvini** needs halo merger histories as input. You can download a merger tree file [here](https://drive.google.com/file/d/1eAiONNCOHSAw829n3zbR1izsIU6JNCO4/view?usp=sharing). This consists of merger histories of 100 haloes in each mass bin, where the bins are linearly distributed within $10^6\leq M_{\rm h}/M_\odot \leq 10^{11}$ at $z=5$.

### Citation

For more information about the model, please refer to (and kindly cite!) the following publications:

1. Menon & Power 2024, **On bursty star formation during cosmological reionisation – how does it influence the baryon mass content of dark matter halos?**, _Publications of the Astronomical Society of Australia_, 41, id.e049, 11 pp. [DOI:
10.1017/pasa.2024.39](https://ui.adsabs.harvard.edu/abs/2024PASA...41...49M/abstract)
1. Menon, Balu & Power 2025, **On bursty star formation during cosmological reionization -- influence on the metal and dust content of low-mass galaxies**, submitted to _Publications of the Astronomical Society of Australia_, [arxiv link](https://arxiv.org/abs/2508.08363)
