# reg_regression_toolkit

`reg_regression_toolkit` provides reusable components for building regularized regression classification workflows. It draws inspiration from the original BLCA tumor classifier notebook and generalizes the pipeline to support both binary and multi-class scenarios.

## Features

- Data reshaping utilities for expression matrices
- Configurable feature filtering based on include/exclude feature lists
- Cross-validated elastic-net logistic regression with standardized preprocessing
- Support for binary and multi-class classification via `scikit-learn`
- Optional SHAP value computation for linear models
- Coefficient aggregation utilities to summarize feature importance
- Example notebook and dummy data for quick evaluation

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies.

```bash
uv sync --extra dev
```

To run the tests (ensuring the ``dev`` extra is available):

```bash
uv run --extra dev pytest
```

The example workflow notebook lives at `examples/workflow.ipynb`. Open it in JupyterLab or VS Code to explore the end-to-end pipeline on the provided dummy dataset.

## Package Layout

- `reg_regression_toolkit/`: Core library code
- `data/`: Dummy expression and metadata used in tests/examples
- `examples/`: Example notebook demonstrating the workflow
- `tests/`: Pytest suite covering binary and multi-class scenarios

## License

MIT License
