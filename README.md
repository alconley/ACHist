# Histogrammer - A Python Library for 1D and 2D Histograms Aimed for Nuclear Spectrum Analysis

The Histogrammer is a Python library that provides functionality for creating and analyzing 1D and 2D histograms. It utilizes the popular `matplotlib`, `numpy`, `polars`, and `lmfit` libraries for plotting and fitting histograms. This README provides an overview of the key features and usage of the Histogrammer library.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1D Histograms](#1d-histograms)
  - [2D Histograms](#2d-histograms)
- [Key Features](#key-features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the Histogrammer library, you need to have Python installed. You can install the required dependencies using `pip`:

```bash
pip install polars matplotlib lmfit numpy colorama
```

## Usage
### 1d-histograms

Function in histogram class to fit multiple gaussians together with the option of saving the fits which can be loaded into the histogram later.

Options: xdata: Can be formatted either as xdata=df["Column"] or xdata=[df_1["Column_i"], df_2["Column_j"],df_3["Column_k"], ...]. The latter combines the data from each df/column and plots the summed histogram.  This is especially importart when you have multiple detectors (say gamma-ray detectors) each with their own filter condition and you want to view the full statitics.

### 2d-histograms
tbd
## Key Features

- Create 1D or 2D histograms from Polars Series or NumPy arrays.
- Interactive plotting with customizable labels, titles, colors, and more.
- Gaussian peak fitting with the ability to add region, peak, and background markers.
- Store and load fitted results for later analysis.
- Extensive keybindings for user interaction.

## Contributing

We welcome contributions to the Histogrammer project! Whether you want to report a bug, propose a feature, or submit a code improvement, we appreciate your input. Here's how you can get involved:

### Reporting Issues

If you encounter any issues, bugs, or unexpected behavior while using Histogrammer, please [open an issue](https://github.com/your-username/histogrammer/issues) on our GitHub repository. Be sure to include as much detail as possible, such as your environment, steps to reproduce the issue, and expected vs. actual outcomes.

### Suggesting Enhancements

If you have ideas for enhancements or new features, feel free to [create an issue](https://github.com/alconley/histogrammer/issues) to discuss them. We're open to new ideas and value your feedback.


