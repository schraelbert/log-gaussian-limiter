[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21940088.svg)](https://doi.org/10.5281/zenodo.21940088)

# Code and data for log-Gaussian continuum--ballistic limiter

This repository contains the code, data, final article figures, and final article tables for:

Bjørn Wu, "A log-Gaussian scale-space limiter for hybrid continuum--ballistic gas dynamics," Open Transport 1(1), 20260023 (2026). https://doi.org/10.1515/ot-2026-0023

## Contents

- `src/loggauss_limiter/`: weight and flux helper functions.
- `experiments/`: scripts for the numerical diagnostics reported in the article.
- `data/`: CSV data used for the reported figures, tables, and numerical statements.
- `figures/`: final figure image files used in the published article source.
- `tables/`: final table files and CSV equivalents.
- `scripts/run_article_reproduction.sh`: runs the article diagnostic workflow.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_article_reproduction.sh
```

Generated outputs are written to `results/`, `paper_tables/`, and `paper_tables_journal/`.

## Citation

Please cite the article:

Bjørn Wu, "A log-Gaussian scale-space limiter for hybrid continuum--ballistic gas dynamics," Open Transport 1(1), 20260023 (2026). https://doi.org/10.1515/ot-2026-0023

Please also cite this archived code and data release:

Bjørn Wu, Code and data for "A log-Gaussian scale-space limiter for hybrid continuum--ballistic gas dynamics", v1.0.1, Zenodo, 2026. https://doi.org/10.5281/zenodo.21940088

## License

This package uses separate licenses for code and non-code research materials.

- Code, shell scripts, and configuration files are released under the MIT License; see `LICENSE-CODE.md`.
- Data files, figure image files, table files, documentation, citation metadata, manifests, and checksum files are released under CC BY 4.0; see `LICENSE-DATA.md`.

The published article itself is not included in this package.
