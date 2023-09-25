import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import os
from lmfit.models import GaussianModel, LinearModel
from lmfit.model import save_modelresult, load_modelresult, save_model, load_model
from scipy.signal import find_peaks
from colorama import Fore, Style
from tabulate import tabulate

plt.rcParams['keymap.pan'].remove('p')
plt.rcParams['keymap.home'].remove('r')
plt.rcParams['keymap.fullscreen'].remove('f')
plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.grid_minor'].remove('G')
plt.rcParams['keymap.quit_all'].append('Q')
plt.rcParams['keymap.xscale'].remove('L')
plt.rcParams['keymap.xscale'].remove('k')
plt.rcParams['keymap.yscale'].remove('l')

def histo1d(
    xdata: list,
    bins: int,
    range: list,
    subplots: (plt.figure, plt.Axes) = None,
    xlabel: str = None,
    ylabel: str = None,
    label: str = None,
    title: str = None,
    color: str = None,
    linestyle: str = None,
    linewidth: float = None,
    display_stats: bool = True,
    ):
    
    if isinstance(xdata, np.ndarray): # when the data is a numpy array, i.e. handles the case for the x/y projections 
        data = xdata
        column = ""
        
    if isinstance(xdata, pl.Series): # checks if xdata is a polars series
        data = xdata.to_numpy()
        column = xdata.name

    if isinstance(xdata, list): # if xdata is a list of polars series
        data = np.concatenate([data.to_numpy() for data in xdata])
        column = '_'.join([item.name for item in xdata])
        
    hist_counts, hist_bins = np.histogram(data, bins=bins, range=range)
    
    fig, ax = (plt.subplots() if subplots is None else subplots)

    if linewidth is None: linewidth = 0.5
    
    ax.step(hist_bins[:-1], hist_counts, where='post', label=label, linewidth=linewidth, color=color, linestyle=linestyle)
    ax.set_xlim(range)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel if xlabel is not None else column)
    ax.set_ylabel(ylabel if ylabel is not None else "Counts")
    if title is not None: ax.set_title(title)
    ax.legend() if label is not None else None
    
    ax.minorticks_on()
    ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
    ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)
            
    if display_stats: matplotlib_1DHistogram_stats(ax=ax, data=data, bins=bins)
        
    interactive_1DHistogram_fitting(hist_counts=hist_counts, hist_bins=hist_bins, subplot=(fig,ax))

    fig.tight_layout()
    
    return

def matplotlib_1DHistogram_stats(ax, data, bins):
    
    def on_xlims_change(ax): # A function to update the stats box when the x-axis limits change
            
        x_lims = ax.get_xlim()
        
        filtered_data = data[(data >= x_lims[0]) & (data <= x_lims[1])] # Filter the data based on the new x-axis limits
        
        new_counts, _ = np.histogram(filtered_data, bins=bins, range=(x_lims[0], x_lims[1]))  # Calculate the new counts
                    
        stats = f"Mean: {np.mean(filtered_data):.2f}\nStd Dev: {np.std(filtered_data):.2f}\nIntegral: {np.sum(new_counts):.0f}"
        
        text_box.set_text(stats)
                        
        ax.figure.canvas.draw()
        
    x_lims = ax.get_xlim()
    on_screen_data = data[(data >= x_lims[0]) & (data <= x_lims[1])] # Filter the data based on the new x-axis limits
    
    # stats = f"Mean: {np.mean(data[(data >= range[0]) & (data <= range[1])]):.2f}\nStd Dev: {np.std(data[(data >= range[0]) & (data <= range[1])]):.2f}\nIntegral: {np.sum(hist):.0f}"
    stats = f"Mean: {np.mean(on_screen_data):.2f}\nStd Dev: {np.std(on_screen_data):.2f}\nIntegral: {np.sum(on_screen_data):.0f}"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
    text_box = ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)


    ax.callbacks.connect('xlim_changed', on_xlims_change) # Connect the on_xlims_change function to the 'xlim_changed' event

    return 

def fit_background(background_markers, hist_counts, hist_bin_centers, background_lines:list, ax):
    
    if len(background_markers) < 2:
        print(f"{Fore.RED}{Style.BRIGHT}Must have two or more background markers!{Style.RESET_ALL}")
        return None, None
    
    background_pos = get_marker_positions(background_markers)
    background_y_values = [hist_counts[np.argmin(np.abs(hist_bin_centers - pos))] for pos in background_pos]
    
    background_model = LinearModel()
    background_result = background_model.fit(background_y_values, x=background_pos)

    background_x_values = np.linspace(background_pos[0], background_pos[-1], 1000)
    background_values = background_result.eval(x=background_x_values)
    
    background_line = ax.plot(background_x_values, background_values, color='green', linewidth=0.5)
    background_lines.append(background_line[0])
        
    return background_result, background_model, background_line[0]

def initial_gaussian_parameters(hist_data, hist_bin_centers, peak_positions, position_uncertainty): # estimates the initital fit parameters
    
    hist_bin_width = (hist_data[1] - hist_data[0])
    
    if len(peak_positions) == 0: # if there are no peak markers, guess the center is at the max value and append that value to the list
        peak_positions.append(hist_bin_centers[np.argmax(hist_data)])
        
    def hist_counts_subtracted_value(number): # get the value of the closest bin to the peak posititon
        index = np.argmin(np.abs(hist_bin_centers - number))
        value = hist_data[index]
        return value
    
    # for guessing the amplitude
    total_peak_height = 0
    for i, peak in enumerate(peak_positions):
        total_peak_height += hist_counts_subtracted_value(peak)
    
    initial_parameters = []
    
    for peak in peak_positions:
    
        center = dict(value=peak,
                    min=peak - position_uncertainty,
                    max=peak + position_uncertainty)
        
        sigma = dict(value=hist_bin_width, min=0, max=hist_bin_width*4)
        
        height_guess = hist_counts_subtracted_value(peak)
        height = dict(value=height_guess,
                    min=hist_counts_subtracted_value(peak - position_uncertainty),
                    max=hist_counts_subtracted_value(peak + position_uncertainty))
        
        amp_scale = height_guess/total_peak_height
        amplitude = dict(value=hist_bin_width * np.sum(hist_data)*amp_scale)
        
        initial_parameters.append([sigma, center, height, amplitude])

    return initial_parameters

def fit_multiple_gaussians(hist_data, hist_bin_centers, peak_positions, initial_parameters):
    
    hist_bin_width = abs(hist_bin_centers[1]-hist_bin_centers[0])
    # Initialize the list of Gaussian models and their parameters
    gaussian_models = []
    initial_parameters = initial_gaussian_parameters(hist_data=hist_data, hist_bin_centers=hist_bin_centers, 
                                                        peak_positions=peak_positions, position_uncertainty=3*hist_bin_width)
    # Loop over the peak_positions and create Gaussian models and parameters
    for i, peak_position in enumerate(peak_positions):
        gauss = GaussianModel(prefix=f'g{i}_')
        
        init_para = initial_parameters[i] 
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
    result = composite_model.fit(hist_data, params, x=hist_bin_centers)
    
    return result, composite_model
      
def gaussian_result_formatted(result, hist_bin_width, prefix, name, print_table=False):
    
    # Get the center, amplitude, sigma, and FWHM for each Gaussian
    center_value = result.params[f'{prefix}center'].value
    amplitude_value = result.params[f'{prefix}amplitude'].value
    area_value = amplitude_value/hist_bin_width
    fwhm_value = abs(result.params[f'{prefix}fwhm'].value)
    relative_width_value = abs(fwhm_value/center_value *100)
    
    center_uncertainty = result.params[f'{prefix}center'].stderr
    if center_uncertainty is not None:
        center_formatted = f"{center_value:.4f} ± {center_uncertainty:4f}"
    else:
        center_formatted = f"{center_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
        
    amplitude_uncertainty = result.params[f'{prefix}amplitude'].stderr
    if amplitude_uncertainty is not None:
        area_uncertainty = amplitude_uncertainty/hist_bin_width
        area_formatted = f"{area_value:.4f} ± {area_uncertainty:4f}"
    else:
        area_formatted = f"{area_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
        
    fwhm_uncertainty = result.params[f'{prefix}fwhm'].stderr
    if fwhm_uncertainty is not None:
        fwhm_formatted = f"{fwhm_value:.4f} ± {fwhm_uncertainty:4f}"
    else:
        fwhm_formatted = f"{fwhm_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
        
    if fwhm_uncertainty and center_uncertainty is not None:
        relative_width_value_uncertainty = relative_width_value*np.sqrt( (fwhm_uncertainty/fwhm_value)**2 +  (center_uncertainty/abs(center_value))**2 )
        relative_width_formatted = f"{relative_width_value:.4f} ± {relative_width_value_uncertainty:.4f}"
    else:
        relative_width_formatted = f"{relative_width_value:.4f} ± {Fore.RED}N/A{Style.RESET_ALL}"
        
    # Append the formatted results to the list
    fit_result = [f'{name}', center_formatted, area_formatted, fwhm_formatted, relative_width_formatted]
    
    if print_table:
        # Define column headers for the table
        headers = ["Gaussian", "Position", "Volume", "FWHM", "Relative Width [%]"]

        # Print the table
        table = tabulate(fit_result, headers, tablefmt="pretty")
        print(table)

    return fit_result

def interactive_1DHistogram_fitting(hist_counts, hist_bins, subplot: (plt.figure, plt.Axes)):
    fig, ax = subplot
    
    hist_bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    hist_bin_width = (hist_bins[1] - hist_bins[0])
    
    region_markers = []
    peak_markers = []
    background_markers = []
    
    background_lines = []
    fit_lines = []
    
    temp_fits = {}
    stored_fits = {}
    
    
    
    def show_keybindings_help(): # Function to display the keybindings help
        
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

        print("\nKeybindings Help:")
        for key, info in keybindings.items():
            description = info['description']
            note = info['note']
            print(f"  {Fore.YELLOW}{key}{Style.RESET_ALL}: {description}")
            if note:
                print(f"      Note: {note}")

    def on_key(event):  # Function to handle fitting gaussians
        
        # Define a dictionary of keybindings, their descriptions, and notes

        if event.inaxes is not None:
            
            if event.key == ' ': # display the help cheat sheet
                show_keybindings_help()
                
            if event.key == 'r': # region markers 
                if len(region_markers) >= 2:
                    print(f"{Fore.BLUE}Removed region markers{Style.RESET_ALL}")
                    remove_lines(region_markers)  # If two lines are present, remove them from the plot and the list
                                    
                place_line_marker(event.xdata, ax, region_markers, color='blue')
                fig.canvas.draw()

            if event.key == 'b': # background markers
                place_line_marker(event.xdata, ax, background_markers, color='green')
                fig.canvas.draw()

            if event.key == 'p': # peak markers
                place_line_marker(event.xdata, ax, peak_markers, color='purple')
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
                remove_nearest_marker(event, region_markers, background_markers, peak_markers)

                fig.canvas.draw()   
                           
            if event.key == 'B':  # fit background
                remove_lines(background_lines)
                background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers, background_lines)
                fig.canvas.draw()
                
            if event.key == 'P': # auto fit peaks 
                if len(region_markers) != 2:
                    print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
                else:
                    
                    remove_lines(peak_markers)
                    
                    region_markers_pos = get_marker_positions(region_markers)
                    
                    fit_range = (hist_bin_centers >= region_markers_pos[0]) & (hist_bin_centers <= region_markers_pos[1])
                    
                    if not background_markers: # if no background markers/fit estimate the background at the region markers
                        remove_lines(background_lines)
                        background_result, background_model, background_line = fit_background(region_markers, hist_counts, hist_bin_centers, background_lines, ax)
                    else:
                        remove_lines(background_lines)
                        background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers, background_lines, ax)

                    hist_counts_subtracted = hist_counts - background_result.eval(x=hist_bin_centers)
                    
                    peak_find_hist = hist_counts_subtracted[fit_range]
                    peak_find_hist_bin_centers = hist_bin_centers[fit_range]
                    peaks, _ = find_peaks(x=peak_find_hist, height=np.max(peak_find_hist)*0.05, threshold=0.05)
                    
                    for peak in peak_find_hist_bin_centers[peaks]:
                        place_line_marker(position=peak, ax=ax, markers=peak_markers, color='purple')
                        
                    fig.canvas.draw()
                           
            if event.key == 'f':  # Fit Gaussians to region
                remove_lines(background_lines)
                remove_lines(fit_lines)
                temp_fits.clear()
                                
                if len(region_markers) != 2:
                    print(f"{Fore.RED}{Style.BRIGHT}Must have two region markers!{Style.RESET_ALL}")
                else:
                    region_markers_pos = get_marker_positions(region_markers)
                    
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
                        background_result, background_model, background_line = fit_background(region_markers, hist_counts, hist_bin_centers, background_lines, ax)
                    else:
                        remove_lines(background_lines)
                        background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers, background_lines, ax)

                    hist_counts_subtracted = hist_counts - background_result.eval(x=hist_bin_centers)
                    
                    # try:
                    
                    initial_parameters = initial_gaussian_parameters(hist_data=hist_counts_subtracted[fit_range], hist_bin_centers=hist_bin_centers[fit_range], 
                                                                     peak_positions=peak_positions, position_uncertainty=3*hist_bin_width)
                    
                    result, composite_model = fit_multiple_gaussians(hist_data=hist_counts_subtracted[fit_range], hist_bin_centers=hist_bin_centers[fit_range], peak_positions=peak_positions, initial_parameters=initial_parameters)
                    
                    # plot result on top of the background
                    total_x = np.linspace(region_markers_pos[0], region_markers_pos[1],2000)
                    fit_p_background_line = ax.plot(total_x, result.eval(x=total_x) + background_result.eval(x=total_x), color='blue', linewidth=0.5) 
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

                        fit_results.append(gaussian_result_formatted(result=result, hist_bin_width=hist_bin_width, prefix=prefix, name=i))

                        fit_line_comp = ax.plot(x_comp, components[prefix], color='blue', linewidth=0.5)  # Gaussian without background
                        fit_lines.append(fit_line_comp[0])
                        
                        fit_line_comp_p_background = ax.plot(x_comp, components[prefix]+ background_result.eval(x=x_comp), color='blue', linewidth=0.5)  # Gaussian and background
                        fit_lines.append(fit_line_comp_p_background[0])

                        place_line_marker(position=result.params[f'{prefix}center'].value, ax=ax, markers=peak_markers, color='purple')
                        
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
                        
                    # except:
                    #         print(f"{Fore.RED}{Style.BRIGHT}\n⚠ Fit Failed ⚠\n{Style.RESET_ALL}")
                                                                                
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
                            loaded_fit_p_background_line = ax.plot(loaded_total_x, loaded_fit_result.eval(x=loaded_total_x) + loaded_fit_background_result.eval(x=loaded_total_x), color='blue', linewidth=0.5) 
                            loaded_fit_background_line = ax.plot(loaded_total_x, loaded_fit_background_result.eval(x=loaded_total_x), 'green', linewidth=0.5)
                            
                            
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
                                
                                loaded_fit_line_comp_p_background = ax.plot(loaded_x_comp, loaded_components[prefix]+ loaded_fit_background_result.eval(x=loaded_x_comp), color='red', linewidth=0.5)  # Gaussian and background
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
    
def remove_lines(lines): # removes axvlines from an array
    for line in lines:
        line.remove()
    lines.clear()
    
def place_line_marker(position, ax, markers, color): # places a axvline 
    line = ax.axvline(position, color=color, linewidth=0.5)
    line.set_antialiased(False)
    markers.append(line)
    
    return 

def remove_nearest_marker(event, region_markers, background_markers, peak_markers):
    
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

def get_marker_positions(markers): #returns the x-value of a axvline marker
    
    positions = [marker.get_xdata()[0] for marker in markers]
    positions.sort()
    return positions

