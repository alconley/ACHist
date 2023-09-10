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

Variables: 
- xdata: Can be formatted either as xdata=df["Column"] or xdata=[df_1["Column_i"], df_2["Column_j"],df_3["Column_k"], ...]. The latter combines the data from each df/column and plots the summed histogram.  This is especially important when you have multiple detectors (say gamma-ray detectors) each with their own filter condition and you want to view the full statitics.
- bins: An integer value (e.g. bins=600)
- range:  A list with the range of the histogram (e.g.. range=(-300,300))

Optional Variables:
- subplots: a list of the form where subplots=(plt.figure, plt.Axes) so you can put the histograms in a figure with many plots.  If no variable is supplied, the function will create its own figure
- xlabel: The x-axis label.  Default is the column name from the xdata variable
- ylabel: The y-axis label. Default is 'Counts'
- label: A label to name the data if you have a legend
- title: The title. Default is ''
- color: The color of the line
- linestyle: The default linestyle is 'solid'
- linewidth: The default linewidth is 0.5
- display_stats: Displays the integral, mean, and stdev give the range of the plot. Can be turned off using display_stats=False


Fitting Gaussians:

This class using lmfit to interactivly fit gaussians on 1D matplotlib histograms.  The goal for this was to be able to fit multiple gaussians on python easily while being able to save and load the fits for later use.

- the keybinds can be viewed by hitting the space-bar 

First the user must supply two region markers ('r').  The user then has a couple of options,
- Put background markers ('b').  This will estimate the background with a linear line which can be visualized with 'B'.  If no background markers are supplied, the background will be estimated at the region markers.
- Auto peak find between the region markers ('P'). This uses scipy-signal function "find_peaks".  The threshold is currently set to 5% of the max value in the region
- Apply peak markers 'p'.  If no peak marker is supplied, the function will assume there is one gaussian with the center at mean of data between the region markers.

The user then has to hit 'f' to preform the fit. 

Additional binds:
- '-' removes the nearest marker to the mouse position
- '_' removes all the markers
- 'F' Stores the fit
- 'S' Saves the fit, the user must input the filename in the terminal
- 'L' Loads a fit, the user must input the filename in the terminal

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


