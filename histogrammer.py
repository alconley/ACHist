
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import numpy as np
import os
from lmfit.models import GaussianModel, LinearModel
from lmfit.model import save_modelresult, load_modelresult, save_model, load_model
from scipy.signal import find_peaks
import os
from colorama import Fore, Style
from tabulate import tabulate
from matplotlib.path import Path
import matplotlib.colors as colors
import json


# tex_fonts = {
#                 # Use LaTeX to write all text
#                 "text.usetex": True,
#                 "font.family": "serif",
#                 "font.serif": "Computer Modern Roman",
#                 # Use 10pt font in plots, to match 10pt font in document
#                 "axes.labelsize": 6,
#                 "font.size": 4,
#                 # Make the legend/label fonts a little smaller
#                 "legend.fontsize": 6,
#                 "xtick.labelsize": 6,
#                 "ytick.labelsize": 6
#             }

# plt.rcParams.update(tex_fonts)


class Histogrammer:

    def __init__(self):
        
        plt.rcParams['keymap.pan'].remove('p')
        plt.rcParams['keymap.home'].remove('r')
        plt.rcParams['keymap.fullscreen'].remove('f')
        plt.rcParams['keymap.grid'].remove('g')
        plt.rcParams['keymap.grid_minor'].remove('G')
        plt.rcParams['keymap.quit_all'].append('Q')
        plt.rcParams['keymap.xscale'].remove('L')
        plt.rcParams['keymap.xscale'].remove('k')
        plt.rcParams['keymap.yscale'].remove('l')
                    
        self.figures = []
        self.cuts = []
        
    """ CutHandler and Cut2D classes were create by Gordon McCain and modified for this histogrammer class
    Handler to recieve vertices from a matplotlib selector (i.e. PolygonSelector).
    Typically will be used interactively, most likely via cmd line interpreter. The onselect
    method should be passed to the selector object at construction. CutHandler can also be used in analysis
    applications to store cuts.
    """
    class CutHandler:
        def __init__(self):
            self.cuts: dict[str, Histogrammer.Cut2D] = {}

        def onselect(self, vertices: list[tuple[float, float]]):
            cut_default_name = f"cut_{len(self.cuts)}"
            self.cuts[cut_default_name] = Histogrammer.Cut2D(cut_default_name, vertices)
    
    """
    Implementation of 2D cuts as used in many types of graphical analyses with matplotlib
    Path objects. Takes in a name (to identify the cut) and a list of points. The Path
    takes the verticies, and can then be used to check if a point(s) is inside of the polygon using the 
    is_*_inside functions. Can be serialized to json format. Can also retreive Nx2 ndarray of vertices
    for plotting after the fact.
    """
    class Cut2D:
        def __init__(self, name: str, vertices: list[tuple[float, float]]):
            self.path: Path = Path(vertices, closed=True)
            self.name = name
            
        def is_point_inside(self, x: float, y: float) -> bool:
            return self.path.contains_point((x,  y))

        def is_arr_inside(self, points: list[tuple[float, float]]) -> list[bool]:
            return self.path.contains_points(points)

        def is_cols_inside(self, columns: pl.Series) -> pl.Series:
            return pl.Series(values=self.path.contains_points(columns.to_list()))

        def get_vertices(self) -> np.ndarray:
            return self.path.vertices

        def to_json_str(self) -> str:
            return json.dumps(self, default=lambda obj: {"name": obj.name, "vertices": obj.path.vertices.tolist()} )
                 
    def histo1d(
        self,
        xdata: list,
        bins: int,
        range: list,
        subplots:(plt.figure, plt.Axes) = None,
        xlabel: str = None,
        ylabel: str = None,
        label: str = None,
        title: str = None,
        color: str = None,
        linestyle: str = None,
        linewidth: float = None,
        display_stats:bool = True,
        ):
        
        if isinstance(xdata, np.ndarray): # handles the case for the x/y projections 
            data = xdata
            column = ""
            
        if isinstance(xdata, pl.Series): # checks if xdata is a polars series
            data = xdata.to_numpy()
            column = xdata.name

        if isinstance(xdata, list): # if xdata is a list of polars series
            data = np.concatenate([data.to_numpy() for data in xdata])
            column = '_'.join([item.name for item in xdata])
            
        hist_counts, hist_bins = np.histogram(data, bins=bins, range=range)
        
        hist_bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        hist_bin_width = (hist_bins[1] - hist_bins[0])
        
        fig, ax = (plt.subplots() if subplots is None else subplots)

        if linewidth is None: linewidth = 0.5
        
        line, = ax.step(hist_bins[:-1], hist_counts, where='post', label=label, linewidth=linewidth, color=color, linestyle=linestyle,)
        ax.set_xlim(range)
        ax.set_ylim(bottom=0)
        
        ax.set_xlabel(xlabel if xlabel is not None else column)
        ax.set_ylabel(ylabel if ylabel is not None else "Counts")
        ax.legend() if label is not None else None
        if title is not None: ax.set_title(title)
        
        ax.minorticks_on()
        ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)
             
        if display_stats:   
            def on_xlims_change(ax): # A function to update the stats box when the x-axis limits change
                
                x_lims = ax.get_xlim()
                
                filtered_data = data[(data >= x_lims[0]) & (data <= x_lims[1])] # Filter the data based on the new x-axis limits
                
                new_counts, _ = np.histogram(filtered_data, bins=bins, range=(x_lims[0], x_lims[1]))  # Calculate the new counts
                            
                stats = f"Mean: {np.mean(filtered_data):.2f}\nStd Dev: {np.std(filtered_data):.2f}\nIntegral: {np.sum(new_counts):.0f}"
                
                text_box.set_text(stats)
                                
                ax.figure.canvas.draw()

            # Create the stats box
            stats = f"Mean: {np.mean(data[(data >= range[0]) & (data <= range[1])]):.2f}\nStd Dev: {np.std(data[(data >= range[0]) & (data <= range[1])]):.2f}\nIntegral: {np.sum(hist_counts):.0f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
            text_box = ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    
            ax.callbacks.connect('xlim_changed', on_xlims_change) # Connect the on_xlims_change function to the 'xlim_changed' event

        region_markers = []
        peak_markers = []
        background_markers = []
        
        background_lines = []
        fit_lines = []
        
        temp_fits = {}
        stored_fits = {}

        def remove_lines(lines):
            for line in lines:
                line.remove()
            lines.clear()
        
        def fit_background(background_markers, hist_counts, hist_bin_centers):
            background_pos = [marker.get_xdata()[0] for marker in background_markers]
            background_pos.sort()

            background_y_values = [hist_counts[np.argmin(np.abs(hist_bin_centers - pos))] for pos in background_pos]
            background_model = LinearModel()
            
            background_result = background_model.fit(background_y_values, x=background_pos)

            background_x_values = np.linspace(background_pos[0], background_pos[-1], 1000)
            background_values = background_result.eval(x=background_x_values)
            background_line = plt.plot(background_x_values, background_values, color='green', linewidth=0.5)
                        
            return background_result, background_model, background_line[0]

        def on_key(event):  # Function to handle fitting gaussians
            
            # Define a dictionary of keybindings, their descriptions, and notes
            keybindings = {
                'r': {
                    'description': "Add region marker",
                    'note': "Must have 2 region markers to preform a fit.",
                },
                'b': {
                    'description': "Add background marker",
                    'note': "must have at least two background markers to estimate the background",
                },
                'p': {
                    'description': "Add peak marker",
                    'note': "If no peak markers are supplied, the program will assume there is one peak at the maximum value",
                },
                'P': {
                    'description': "Auto peak finder",
                    'note': "Trys to find all the peaks between the region markers",
                },
                '-': {
                    'description': "Remove all markers and temp fits",
                    'note': "",
                },
                'B': {
                    'description': "Fit background",
                    'note': "Fits the background markers using a linear line",
                },
                'f': {
                    'description': "Fit Gaussians to region",
                    'note': "Must have two region markers. If no background markers are supplied, the background will be esitamted at the region markers. Number of Gaussians fitted will depend on the number of peak markers inbetween the region markers",
                },
                'F': {
                    'description': "Store the fits",
                    'note': "",
                },
                'S': {
                    'description': "Save fits to file",
                    'note': "Saves the stored fits to a ASCII (user must input the file name)",
                },
                'L': {
                    'description': "Load fits from file",
                    'note': "",
                },
                'space-bar': {
                    'description': "Show keybindings help",
                    'note': "",
                },
            }

            # Function to display the keybindings help
            def show_keybindings_help():
                print("\nKeybindings Help:")
                for key, info in keybindings.items():
                    description = info['description']
                    note = info['note']
                    print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}: {description}")
                    if note:
                        print(f"      Note: {note}")
                        
            if event.inaxes is not None:
                
                if event.key == ' ': # display the help cheat sheet
                    show_keybindings_help()
                   
                if event.key == 'r': # region markers
                    
                    if len(region_markers) >= 2:
                        print(f"{Fore.BLUE}Removed region markers{Style.RESET_ALL}")
                        
                        remove_lines(region_markers)  # If two lines are present, remove them from the plot and the list
                                        
                    region_pos = event.xdata       
                    region_line = ax.axvline(region_pos, color='blue', linewidth=0.5)
                    print(f"{Fore.BLUE}Region marker placed at {region_pos}{Style.RESET_ALL}")
                    region_line.set_antialiased(False)
                    region_markers.append(region_line)
                    fig.canvas.draw()

                if event.key == 'P': # auto fit peaks 
                    if len(region_markers) != 2:
                        print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
                    else:
                        
                        remove_lines(peak_markers)
                        
                        region_markers_pos = [region_markers[0].get_xdata()[0], region_markers[1].get_xdata()[0]]
                        region_markers_pos.sort()
                        
                        fit_range = (hist_bin_centers >= region_markers_pos[0]) & (hist_bin_centers <= region_markers_pos[1])
                        
                        
                        if not background_markers: # if no background markers/fit estimate the background at the region markers
                            remove_lines(background_lines)
                            background_result, background_model, background_line = fit_background(region_markers, hist_counts, hist_bin_centers)
                            background_lines.append(background_line)
                        else:
                            remove_lines(background_lines)
                            background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
                            background_lines.append(background_line)

                        hist_counts_subtracted = hist_counts - background_result.eval(x=hist_bin_centers)
                        
                        peak_find_hist = hist_counts_subtracted[fit_range]
                        peak_find_hist_bin_centers = hist_bin_centers[fit_range]
                        peaks, _ = find_peaks(x=peak_find_hist, height=np.max(peak_find_hist)*0.05, threshold=0.05)
                                                
                        for peak in peak_find_hist_bin_centers[peaks]:
                            peak_line = ax.axvline(x=peak, color='purple', linewidth=0.5)
                            peak_line.set_antialiased(False)
                            peak_markers.append(peak_line)
                            
                        
                        fig.canvas.draw()
                        
                if event.key == 'b': # background markers

                    pos = event.xdata           
                    background_line = ax.axvline(x=pos, color='green', linewidth=0.5)
                    print(f"{Fore.GREEN}Background marker placed at {pos}{Style.RESET_ALL}")
                    background_line.set_antialiased(False)
                    background_markers.append(background_line)
                    fig.canvas.draw()

                if event.key == 'p': # peak markers

                    pos = event.xdata
                    peak_line = ax.axvline(x=pos, color='purple', linewidth=0.5)
                    print(f"{Fore.MAGENTA}Peak marker placed at {pos}{Style.RESET_ALL}")
                    
                    peak_line.set_antialiased(False)
                    peak_markers.append(peak_line)
                    fig.canvas.draw()
                        
                if event.key == '_': # remove all markers and temp fits

                    remove_lines(region_markers)
                    remove_lines(peak_markers)
                    remove_lines(background_markers)
                    remove_lines(background_lines)
                    remove_lines(fit_lines)
                    temp_fits.clear()
                    fig.canvas.draw()
                
                if event.key == '-': # remove the closest marker to cursor in axes
                    
                    remove_lines(background_lines)
                    remove_lines(fit_lines)
                    temp_fits.clear()

                    x_cursor = event.xdata

                    def line_distances(marker_array):
                        if len(marker_array) > 0:
                            distances = [x_cursor - line.get_xdata()[0] for line in marker_array]
                            min_distance = np.min(np.abs(distances))
                            min_distance_index = np.argmin(np.abs(distances))
                            
                            return [min_distance, min_distance_index]
                        else:
                            return None
                    
                    marker_distances = [line_distances(region_markers), line_distances(peak_markers), line_distances(background_markers)]

                    # Filter out None values and find minimum distances
                    valid_marker_distances = [min_distance for min_distance in marker_distances if min_distance is not None]

                    # Check if there are valid distances
                    if valid_marker_distances:
                        # Find the minimum distance based on the first index
                        min_distance = min(valid_marker_distances, key=lambda x: x[0])

                        if min_distance == marker_distances[0]:
                            for i, line in enumerate(region_markers):
                                if i == min_distance[1]:
                                    line.remove()
                                    region_markers.pop(i)

                        elif min_distance == marker_distances[1]:
                            for i, line in enumerate(peak_markers):
                                if i == min_distance[1]:
                                    line.remove()
                                    peak_markers.pop(i)

                        elif min_distance == marker_distances[2]:
                            for i, line in enumerate(background_markers):
                                if i == min_distance[1]:
                                    line.remove()
                                    background_markers.pop(i)       
                    else:
                        print(f"{Fore.RED}{Style.BRIGHT}No valid distances found.{Style.RESET_ALL}")

                    fig.canvas.draw()
                       
                if event.key == 'B':  # fit background
                    
                    remove_lines(background_lines)
                    background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
                    
                    background_lines.append(background_line)                    
                    fig.canvas.draw()
                             
                if event.key == 'f':  # Fit Gaussians to region
                    remove_lines(background_lines)
                    remove_lines(fit_lines)
                    temp_fits.clear()
                    
                    print(temp_fits)
                    
                    if len(region_markers) != 2:
                        print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
                    else:
                        region_markers_pos = [region_markers[0].get_xdata()[0], region_markers[1].get_xdata()[0]]
                        region_markers_pos.sort()
                        
                        # removes peak markers that are not in between the region markers
                        peak_positions = []
                        for marker in peak_markers:
                            if region_markers_pos[0] < marker.get_xdata()[0] < region_markers_pos[1]:
                                peak_positions.append(marker.get_xdata()[0])
                        peak_positions.sort()
                        
                        remove_lines(peak_markers)

                        fit_range = (hist_bin_centers >= region_markers_pos[0]) & (hist_bin_centers <= region_markers_pos[1])

                        if not background_markers: # if no background markers/fit estimate the background at the region markers
                            remove_lines(background_lines)
                            background_result, background_model, background_line = fit_background(region_markers, hist_counts, hist_bin_centers)
                            background_lines.append(background_line)
                        else:
                            remove_lines(background_lines)
                            background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
                            background_lines.append(background_line)

                        hist_counts_subtracted = hist_counts - background_result.eval(x=hist_bin_centers)
                        
                        try:
                       
                            def hist_counts_subtracted_value(number): # get the value of the closest bin to the peak posititon
                                index = np.argmin(np.abs(hist_bin_centers - number))
                                value = hist_counts_subtracted[index]
                                return value

                            def initial_para(peak_position, peak_position_guess_uncertainty, amplitude_scale:float=None): # estimates the initital fit parameters
                                
                                # avg_sigma = np.sum(hist_counts_subtracted[fit_range]) / np.std(hist_counts_subtracted[fit_range])
                                
                                sigma = dict(value=hist_bin_width, min=0, max=hist_bin_width*4)
                                
                                center = dict(value=peak_position,
                                            min=peak_position - peak_position_guess_uncertainty,
                                            max=peak_position + peak_position_guess_uncertainty)

                                height = dict(value=hist_counts_subtracted_value(peak_position),
                                            min=hist_counts_subtracted_value(peak_position - peak_position_guess_uncertainty),
                                            max=hist_counts_subtracted_value(peak_position + peak_position_guess_uncertainty))

                                if amplitude_scale is not None:
                                    amplitude = dict(value=hist_bin_width * np.sum(hist_counts_subtracted[fit_range])*amplitude_scale)
                                else:   
                                    amplitude = dict(value=hist_bin_width * np.sum(hist_counts_subtracted[fit_range]))

                                return [sigma, center, height, amplitude]
                            
                            # if there are no peak positions, assume there is one peak in the region markers at the max value
                            if len(peak_positions) == 0: 
                                
                                # Find the index where the maximum value occurs
                                max_index = np.argmax(hist_counts_subtracted[fit_range])
                                
                                peak_positions.append(hist_bin_centers[fit_range][max_index])
                            
                            total_peak_height = 0
                            for i, peak_position in enumerate(peak_positions):
                                total_peak_height += hist_counts_subtracted_value(peak_position)

                            # Initialize the list of Gaussian models and their parameters
                            gaussian_models = []
                            
                            # Loop over the peak_positions and create Gaussian models and parameters
                            for i, peak_position in enumerate(peak_positions):
                                gauss = GaussianModel(prefix=f'g{i}_')
                                
                                amp_scale = hist_counts_subtracted_value(peak_position)/total_peak_height
                                init_para = initial_para(peak_position,peak_position_guess_uncertainty=3*hist_bin_width, amplitude_scale=amp_scale)   
                                    
                                if i == 0:
                                    params = gauss.make_params(sigma=init_para[0],
                                                                center=init_para[1],
                                                                height=init_para[2],
                                                                amplitude=init_para[3])
                                    
                                else:
                                    params.update(gauss.make_params(sigma=init_para[0],
                                                                center=init_para[1],
                                                                height=init_para[2],
                                                                amplitude=init_para[3]))
                                    
                                gaussian_models.append(gauss)
                                
                            # Create the composite model by adding all Gaussian models together
                            composite_model = gaussian_models[0]
                            for gauss in gaussian_models[1:]:
                                composite_model += gauss

                            # Fit the composite model to the data
                            result = composite_model.fit(hist_counts_subtracted[fit_range], params, x=hist_bin_centers[fit_range])
                            
                            # plot result on top of the background
                            total_x = np.linspace(region_markers_pos[0], region_markers_pos[1],2000)
                            fit_p_background_line = plt.plot(total_x, result.eval(x=total_x) + background_result.eval(x=total_x), color='blue', linewidth=0.5) 
                            fit_lines.append(fit_p_background_line[0])
                            
                            print(f"{Fore.GREEN}{Style.BRIGHT}Fit Report{Style.RESET_ALL}")
                            # print(result.fit_report())
                            
                            fit_results = []
                            
                            # Decomposition of gaussians
                            for i, peak_position in enumerate(peak_positions):
                                
                                prefix = f'g{i}_'
                                sigma_plot_width = 4
                                x_comp = np.linspace(result.params[f'{prefix}center'].value - sigma_plot_width * result.params[f'{prefix}sigma'].value,
                                    result.params[f'{prefix}center'].value + sigma_plot_width * result.params[f'{prefix}sigma'].value, 1000)

                                components = result.eval_components(x=x_comp)
                                
                                # Get the center, amplitude, sigma, and FWHM for each Gaussian
                                center_value = result.params[f'{prefix}center'].value
                                center_uncertainty = result.params[f'{prefix}center'].stderr

                                amplitude_value = result.params[f'{prefix}amplitude'].value
                                amplitude_uncertainty = result.params[f'{prefix}amplitude'].stderr
                                
                                area_value = amplitude_value/hist_bin_width
                                area_uncertainty = amplitude_uncertainty/hist_bin_width

                                fwhm_value = result.params[f'{prefix}sigma'].value
                                fwhm_uncertainty = result.params[f'{prefix}sigma'].stderr
                                
                                relative_width_value = fwhm_value/center_value *100
                                relative_width_value_uncertainty = relative_width_value*np.sqrt( (fwhm_uncertainty/fwhm_value)**2 +  (center_uncertainty/center_value)**2 )
                                

                                # Format the values and their uncertainties
                                center_formatted = f"{center_value:.4f} ± {center_uncertainty:.4f}"
                                area_formatted = f"{area_value:.4f} ± {area_uncertainty:.4f}"
                                
                                fwhm_formatted = f"{fwhm_value:.4f} ± {fwhm_uncertainty:.4f}"
                                relative_width_formatted = f"{relative_width_value:.4f} ± {relative_width_value_uncertainty:.4f}"

                                # Append the formatted results to the list
                                fit_results.append([f'{i}', center_formatted, area_formatted, fwhm_formatted, relative_width_formatted])

                                # fit_line_comp = plt.plot(x_comp, components[prefix]+ background_result.eval(x=x_comp), color='purple', linewidth=0.5)  # Gaussian without background
                                # fit_lines.append(fit_line_comp[0])
                                
                                fit_line_comp_p_background = plt.plot(x_comp, components[prefix]+ background_result.eval(x=x_comp), color='blue', linewidth=0.5)  # Gaussian and background
                                fit_lines.append(fit_line_comp_p_background[0])

                                fit_peak_line = ax.axvline(x=result.params[f'{prefix}center'].value, color='purple', linewidth=0.5) # adjust the peak to be the center of the fit
                                peak_markers.append(fit_peak_line)
                                
                                # Define column headers for the table
                            headers = ["Gaussian", "Position", "Volume", "FWHM", "Relative Width [%]"]

                            # Print the table
                            table = tabulate(fit_results, headers, tablefmt="pretty")
                            print(table)
                                    
                            temp_fit_id = f"temp_fit_{len(temp_fits)}"
                            temp_fits[temp_fit_id] = {
                                "region_markers": region_markers_pos,
                                "fit_model": composite_model,
                                "fit_result": result,
                                "fit_lines": fit_lines,
                                "background_model": background_model,
                                "background_result": background_result,
                                "background_line": background_line,
                                "fit_p_background_line": fit_p_background_line[0]
                            }
                                
                            fig.canvas.draw()
                            
                        except:
                                print(f"{Fore.RED}{Style.BRIGHT}\n⚠ Fit Failed ⚠\n{Style.RESET_ALL}")
                                                                                 
                if event.key == 'F': # store the fits
                    fit_id = f"Fit_{len(stored_fits)}"
                    
                    for fit in temp_fits:
                        stored_fits[fit_id] = temp_fits[fit]
                        
                        for i, fit in enumerate(stored_fits[fit_id]["fit_lines"]):
                            stored_fits[fit_id]["fit_lines"][i].set_color('m')
                            ax.add_line(stored_fits[fit_id]["fit_lines"][i])
                            
                        stored_fits[fit_id]["background_line"].set_color('m')
                        ax.add_line(stored_fits[fit_id]["background_line"])
                        
                        stored_fits[fit_id]["fit_p_background_line"].set_color('m')
                        ax.add_line(stored_fits[fit_id]["fit_p_background_line"])
                        

                    fig.canvas.draw()
                    
                if event.key == "S":  # Save fits to file
                    if stored_fits:
                        formatted_results = {}  # Initialize a dictionary to store combined results
                        for fit_id, fit_data in stored_fits.items():
                            model_filename = f'temp_{fit_id}_model.sav'
                            background_model_filename = f'temp_{fit_id}_background_model.sav'
                            result_filename = f'temp_{fit_id}_result.sav'
                            background_result_filename = f'temp_{fit_id}_background_result.sav'

                            # Save model, background model, fit result, and background result
                            save_model(fit_data["fit_model"], model_filename)
                            save_model(fit_data["background_model"], background_model_filename)
                            save_modelresult(fit_data["fit_result"], result_filename)
                            save_modelresult(fit_data["background_result"], background_result_filename)

                            # Extract fit parameters and calculate volume
                            fit_params = fit_data["fit_result"].params
                            fit_parameters = {}
                            for param_name, param_value in fit_params.items():
                                fit_parameters[param_name] = param_value.value
                                fit_parameters[f"{param_name}_uncertainty"] = param_value.stderr

                            # fit_parameters["volume"] = fit_params['amplitude'].value / hist_bin_width
                            # fit_parameters["volume_uncertainty"] = fit_params['amplitude'].stderr / hist_bin_width

                            
                            # Append the saved contents to the formatted_results dictionary
                            formatted_results[fit_id] = {
                                "fit_parameters": fit_parameters,
                                "region_markers": fit_data["region_markers"],
                                "fit_model": open(model_filename, 'r').read(),
                                "background_model": open(background_model_filename, 'r').read(),
                                "fit_result": open(result_filename, 'r').read(),
                                "background_result": open(background_result_filename, 'r').read()
                            }

                            # Remove the temporary files
                            os.remove(model_filename)
                            os.remove(background_model_filename)
                            os.remove(result_filename)
                            os.remove(background_result_filename)

                        filename = input(f"{Fore.YELLOW}{Style.BRIGHT}Enter a filename to save the fits to: {Style.RESET_ALL}")
                        
                        # Save the formatted_results dictionary to a new file
                        with open(f"{filename}", "w") as output_file:
                            output_file.write(str(formatted_results))
                        print(f"{Fore.GREEN}{Style.BRIGHT}Saved fits to file: {filename} {Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}{Style.BRIGHT}No fits to save{Style.RESET_ALL}")
                        
                if event.key == "L":  # Load fits from file
                    filename = input(f"{Fore.YELLOW}{Style.BRIGHT}Enter the filename to load fits from: {Style.RESET_ALL}")
                    
                    if os.path.exists(f"{filename}"):
                        with open(f"{filename}", "r") as input_file:
                            loaded_fits = eval(input_file.read())
                            
                            for fit_id, fit_data in loaded_fits.items():
                                print('Fit id: ',fit_id)
                                model_filename = f'temp_{fit_id}_model.sav'
                                background_model_filename = f'temp_{fit_id}_background_model.sav'
                                result_filename = f'temp_{fit_id}_result.sav'
                                background_result_filename = f'temp_{fit_id}_background_result.sav'
                                
                                # Save model and result data to temporary files
                                with open(model_filename, "w") as temp_file:
                                    temp_file.write(f"{fit_data['fit_model']}")
                                with open(background_model_filename, "w") as temp_file:
                                    temp_file.write(f"{fit_data['background_model']}")
                                with open(result_filename, "w") as temp_file:
                                    temp_file.write(f"{fit_data['fit_result']}")
                                with open(background_result_filename, "w") as temp_file:
                                    temp_file.write(f"{fit_data['background_result']}")
                                
                                # Load model and result data from temporary files
                                loaded_fit_model = load_model(model_filename)
                                loaded_fit_background_model = load_model(background_model_filename)
                                loaded_fit_result = load_modelresult(result_filename)
                                loaded_fit_background_result = load_modelresult(background_result_filename)
                                
                                # Clean up: delete the temporary files
                                os.remove(model_filename)
                                os.remove(background_model_filename)
                                os.remove(result_filename)
                                os.remove(background_result_filename)
                                
                                loaded_region_markers = fit_data['region_markers']
                                
                                
                                loaded_total_x = np.linspace(loaded_region_markers[0], loaded_region_markers[1], 2000)
                                loaded_fit_p_background_line = plt.plot(loaded_total_x, loaded_fit_result.eval(x=loaded_total_x) + loaded_fit_background_result.eval(x=loaded_total_x), color='blue', linewidth=0.5) 
                                loaded_fit_background_line = plt.plot(loaded_total_x, loaded_fit_background_result.eval(x=loaded_total_x), 'green', linewidth=0.5)
                                
                                
                                # Get the parameter names
                                param_names = loaded_fit_result.params.keys()

                                # Initialize a set to collect unique prefixes
                                unique_prefixes = set()

                                # Iterate through parameter names and extract prefixes
                                for param_name in param_names:
                                    # Split the parameter name by '_' to get the prefix
                                    parts = param_name.split('_')
                                    
                                    if len(parts) > 1:
                                        # Add the prefix to the set
                                        unique_prefixes.add(parts[0])

                                for i, pre in  enumerate(unique_prefixes):
                                    
                                    prefix = f"{pre}_"
                                    
                                    sigma_plot_width = 4
                                    loaded_x_comp = np.linspace(loaded_fit_result.params[f'{prefix}center'].value - sigma_plot_width * loaded_fit_result.params[f'{prefix}sigma'].value,
                                                                loaded_fit_result.params[f'{prefix}center'].value + sigma_plot_width * loaded_fit_result.params[f'{prefix}sigma'].value, 1000)
                                                                
                                    loaded_components = loaded_fit_result.eval_components(x=loaded_x_comp)
                                    
                                    loaded_fit_line_comp_p_background = plt.plot(loaded_x_comp, loaded_components[prefix]+ loaded_fit_background_result.eval(x=loaded_x_comp), color='red', linewidth=0.5)  # Gaussian and background
                                    fit_lines.append(loaded_fit_line_comp_p_background[0])
                                
                                temp_fit_id = f"temp_fit_{len(temp_fits)}"
                                temp_fits[temp_fit_id] = {
                                    "region_markers": loaded_region_markers,
                                    "fit_model": loaded_fit_model,
                                    "fit_result": loaded_fit_result,
                                    "fit_lines": fit_lines,
                                    "background_model": loaded_fit_background_model,
                                    "background_result": loaded_fit_background_result,
                                    "background_line": loaded_fit_background_line[0],
                                    "fit_p_background_line": loaded_fit_p_background_line[0]
                                }

                                fig.canvas.draw()
                            
                            print(f"{Fore.GREEN}{Style.BRIGHT}Loaded {len(loaded_fits)} fits from file: {filename}{Style.RESET_ALL}")
                                    
        ax.figure.canvas.mpl_connect('key_press_event', on_key)
            
        fig.tight_layout()
        self.figures.append(fig)

        return
                
    def histo2d(
        self,
        data: list,
        bins: list,
        range: list,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        subplots:(plt.figure,plt.Axes) = None,
        display_stats: bool = True,
        cmap: str = None,
        cbar: bool = True,
        ):
 
        # Concatenate the arrays horizontally to get the final result
        x_data = np.hstack([column[0] for column in data])
        y_data = np.hstack([column[1] for column in data])
        
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins, range=range)

        # draw a 2d histogram using matplotlib. If no ax is provided, make a new fig, ax using plt.subplots
        if subplots is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots 
            
        handler = self.CutHandler()
        
        selector = PolygonSelector(ax, onselect=handler.onselect)
        selector.set_active(False)
        
        if cmap is None: "viridis"
        
        h = ax.hist2d(x_data, y_data, bins=bins, range=range, cmap=cmap, norm=colors.LogNorm())
        
        if cbar: fig.colorbar(h[3], ax=ax)
            
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        ax.minorticks_on()
        ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)
        
        if display_stats:
            
            # Create the stats box
            stats = f"Integral: {np.sum(hist):.0f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
            text_box = ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
            
            def on_lims_change(ax):
                # Get the new x-axis limits
                x_lims = ax.get_xlim()

                # Get the new y-axis limits
                y_lims = ax.get_ylim()
                
                # Create a mask to filter both x_data and y_data
                mask = (x_data >= x_lims[0]) & (x_data <= x_lims[1]) & (y_data >= y_lims[0]) & (y_data <= y_lims[1])

                filtered_x_data = x_data[mask]
                filtered_y_data = y_data[mask]
            
                # Create a new 2D histogram using the filtered data
                filtered_hist, _, _ = np.histogram2d(filtered_x_data, filtered_y_data, bins=bins, range=[[x_lims[0], x_lims[1]], [y_lims[0], y_lims[1]]])

                # Calculate the new statistics using the filtered histogram
                stats = f"Integral: {np.sum(filtered_hist):.0f}"

                # Update the text of the stats box
                text_box.set_text(stats)
                ax.figure.canvas.draw()

            # Connect the on_lims_change function to the 'xlim_changed' event
            ax.callbacks.connect('xlim_changed', on_lims_change)
            ax.callbacks.connect('ylim_changed', on_lims_change)
              
        # Keep track of the added lines for the x and y projections
        added_y_lines = []
        added_x_lines = []
        def on_press(event):  # Function to handle mouse click events like x/y projections
            
            # Define a dictionary of keybindings, their descriptions, and notes
            keybindings = {
                
                'x': {
                    'description': "Add a vertical line to view the X-projection",
                    'note': "Must have two lines to view the X-projection",
                },
                'X': {
                    'description': "Opens a 1D histogram of the X-projection between the two X-projection lines",
                    'note': "",
                },
                'y': {
                    'description': "Add a vertical line to view the Y-projection",
                    'note': "Must have two lines to view the Y-projection",
                },
                'Y': {
                    'description': "Opens a 1D histogram of the Y-projection between the two Y-projection lines",
                    'note': "",
                },
                'c': {
                    'description': "Enables Matplotlib's polygon selector tool",
                    'note': "Left click to place vertices. Once the shape is completed, the vertices can be moved by dragging them.",
                },
                'C': {
                    'description': "Saves the cut",
                    'note': "User has to input the filename in the terminal (e.g. cut.json)",
                },

                'space-bar': {
                    'description': "Show keybindings help",
                    'note': "",
                },
            }

            # Function to display the keybindings help
            def show_keybindings_help():
                print("\nKeybindings Help:")
                for key, info in keybindings.items():
                    description = info['description']
                    note = info['note']
                    print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}: {description}")
                    if note:
                        print(f"      Note: {note}")
            
            if event.inaxes is not None:
                
                if event.key == ' ': # display the help cheat sheet
                    show_keybindings_help()
            
                if event.key == 'y': # For drawing lines to do a y-projection
                    # Check if there are already two lines present
                    if len(added_y_lines) >= 2:
                        # If two lines are present, remove them from the plot and the list
                        for line in added_y_lines:
                            line.remove()
                        added_y_lines.clear()
                    
                    x_coord = event.xdata
                    line = ax.axvline(x_coord, color='red')
                    added_y_lines.append(line)

                    fig.canvas.draw()
                    
                if event.key == 'Y': # For showing the y-projection
                    
                    if len(added_y_lines) < 2: 
                        print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")
                        
                    else:
                        x_coordinates = []
                        for line in added_y_lines:
                            x_coordinate = line.get_xdata()[0]  # Assuming the line is vertical and has only one x-coordinate
                            x_coordinates.append(x_coordinate)
                        x_coordinates.sort() 
                        
                        x_mask = (x_data >= x_coordinates[0]) & (x_data <= x_coordinates[1])
                            
                        y_projection_data = y_data[x_mask]
                        
                        self.histo1d(xdata=y_projection_data, bins=bins[1], range=range[1], title=f"Y-Projection: {round(x_coordinates[0], 2)} to {round(x_coordinates[1], 2)}")
                        
                        plt.show()
                        
                if event.key == 'x': # For drawing lines to do a x-projection
                    # Check if there are already two lines present
                    if len(added_x_lines) >= 2:
                        # If two lines are present, remove them from the plot and the list
                        for line in added_x_lines:
                            line.remove()
                        added_x_lines.clear()
                    
                    y_coord = event.ydata
                    line = ax.axhline(y_coord, color='green')
                    added_x_lines.append(line)

                    fig.canvas.draw()
                    
                if event.key == 'X': # For showing the X-projection
                    
                    if len(added_x_lines) < 2: 
                        print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")

                    else:
                        y_coordinates = []
                        for line in added_x_lines:
                            y_coordinate = line.get_ydata()[0] 
                            y_coordinates.append(y_coordinate)
                        y_coordinates.sort() 
                        
                        y_mask = (y_data >= y_coordinates[0]) & (y_data <= y_coordinates[1])
                            
                        x_projection_data = x_data[y_mask]
                        
                        self.histo1d(xdata=x_projection_data, bins=bins[1], range=range[1], title=f"X-Projection: {round(y_coordinates[0], 2)} to {round(y_coordinates[1], 2)}")
                        
                        plt.show()
                        
                if event.key == 'c': # create a cut

                    print(f"{Fore.YELLOW}Activating the polygon selector tool:\n\tPress 'C' to save the cut (must enter cut name e.g. cut.json){Style.RESET_ALL}")
                    
                    selector.set_active(True)
                    plt.show()
                    
                if event.key == 'C': # save the cut to a file name that the user must enter
                    selector.set_active(False)
                    plt.show()
                    
                    handler.cuts["cut_0"].name = "cut"
                    
                    # Prompt the user for the output file name
                    output_file = input(f"{Fore.YELLOW}Enter a name for the output file (e.g., cut.json): {Style.RESET_ALL}")

                    # Write the cut to the specified output file
                    try:
                        self.write_cut_json(handler.cuts["cut_0"], output_file)
                        print(f"{Fore.GREEN}Cut saved to '{output_file}' successfully.{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error: {e}. Failed to save the cut to '{Style.RESET_ALL}'.")
                        
        ax.figure.canvas.mpl_connect('key_press_event', on_press)
        
        fig.tight_layout()
        
        self.figures.append(fig)
        
        return hist, x_edges, y_edges

    def write_cut_json(self, cut: Cut2D, filepath):
        json_str = cut.to_json_str()
        try:
            with open(filepath, "w") as output:
                output.write(json_str)
                return True
        except OSError as error:
            print(f"An error occurred writing cut {cut.name} to file {filepath}: {error}")
            return False

    def load_cut_json(self, filepath: str):
        try:
            with open(filepath, "r") as input:
                buffer = input.read()
                cut_dict = json.loads(buffer)
                if not "name" in cut_dict or not "vertices" in cut_dict:
                    print(f"Data in file {filepath} is not the right format for Cut2D, could not load")
                    return None
                return self.Cut2D(cut_dict["name"], cut_dict["vertices"])
        except OSError as error:
            print(f"An error occurred reading trying to read a cut from file {filepath}: {error}")
            return None

    def filter_df_with_cut(self, df:pl.DataFrame, XColumn:pl.Series, YColumn:pl.Series, CutFile: str):
        
        cut = self.load_cut_json(CutFile)
        df = df.filter(pl.col(XColumn).arr.concat(YColumn).map(cut.is_cols_inside))
        
        return df