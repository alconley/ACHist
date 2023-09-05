
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from lmfit.models import GaussianModel, LinearModel, Model
from lmfit.model import save_modelresult, load_modelresult
from lmfit.model import save_model, load_model
import os
from colorama import Fore, Style

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
        linewidth: int = None,
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
                
                hist_counts[:] = new_counts  # Update the counts variable
            
                stats = f"Mean: {np.mean(filtered_data):.2f}\nStd Dev: {np.std(filtered_data):.2f}\nIntegral: {np.sum(hist_counts):.0f}"
                
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
                'h': {
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
                
                if event.key == 'h': # display the help cheat sheet
                    show_keybindings_help()
                   
                if event.key == 'r': # region markers
                    
                    if len(region_markers) >= 2:
                        remove_lines(region_markers)  # If two lines are present, remove them from the plot and the list
                                        
                    region_pos = event.xdata       
                    region_line = ax.axvline(region_pos, color='#0B00E9', linewidth=0.5)
                    region_line.set_antialiased(False)
                    region_markers.append(region_line)
                    fig.canvas.draw()
                    
                if event.key == 'b': # background markers

                    pos = event.xdata           
                    background_line = ax.axvline(x=pos, color='green', linewidth=0.5)
                    background_line.set_antialiased(False)
                    background_markers.append(background_line)
                    fig.canvas.draw()

                if event.key == 'p': # peak markers

                    pos = event.xdata
                    peak_line = ax.axvline(x=pos, color='purple', linewidth=0.5)
                    peak_line.set_antialiased(False)
                    peak_markers.append(peak_line)
                    fig.canvas.draw()
                        
                if event.key == '-': # remove all markers and temp fits

                    remove_lines(region_markers)
                    remove_lines(peak_markers)
                    remove_lines(background_markers)
                    remove_lines(background_lines)
                    remove_lines(fit_lines)
                    temp_fits.clear()
                    fig.canvas.draw()
                       
                if event.key == 'B':  # fit background
                    
                    remove_lines(background_lines)
                    background_result, background_model, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
                    background_lines.append(background_line)                    
                    fig.canvas.draw()
                             
                if event.key == 'f':  # Fit Gaussians to region
                    remove_lines(background_lines)
                    remove_lines(fit_lines)
                    
                    if len(region_markers) != 2:
                        print("Must have two region markers!")
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
                                
                                avg_sigma = np.sum(hist_counts_subtracted[fit_range]) / np.std(hist_counts_subtracted[fit_range])

                                sigma = dict(value=avg_sigma, min=0, max=avg_sigma * 4)

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
                                init_para = initial_para(peak_position,peak_position_guess_uncertainty=4*hist_bin_width, amplitude_scale=amp_scale)   
                                    
                                if i == 0:
                                    params = gauss.make_params(sigma=init_para[0],
                                                                center=init_para[1],
                                                                height=init_para[2],
                                                                amplitude=init_para[3],
                                                                area=4000)
                                    
                                else:
                                    params.update(gauss.make_params(sigma=init_para[0],
                                                                center=init_para[1],
                                                                height=init_para[2],
                                                                amplitude=init_para[3],
                                                                area=4000))
                                    
                                gaussian_models.append(gauss)
                                
                            # Create the composite model by adding all Gaussian models together
                            composite_model = gaussian_models[0]
                            for gauss in gaussian_models[1:]:
                                composite_model += gauss

                            # Fit the composite model to the data
                            result = composite_model.fit(hist_counts_subtracted[fit_range], params, x=hist_bin_centers[fit_range])
                            
                            # plot result on top of the background
                            total_x = np.linspace(region_markers_pos[0], region_markers_pos[1],2000)
                            fit_p_background_line = plt.plot(total_x, result.eval(x=total_x) + background_result.eval(x=total_x), color='red', linewidth=0.5) 
                            fit_lines.append(fit_p_background_line[0])
                            
                            print(print(f"{Fore.GREEN}{Style.BRIGHT}Fit Report{Style.RESET_ALL}"))
                            print(result.fit_report())
                            
                            # Decomposition of gaussians
                            for i, peak_position in enumerate(peak_positions):
                                
                                prefix = f'g{i}_'
                                sigma_plot_width = 5
                                x_comp = np.linspace(result.params[f'{prefix}center'].value - sigma_plot_width * result.params[f'{prefix}sigma'].value,
                                    result.params[f'{prefix}center'].value + sigma_plot_width * result.params[f'{prefix}sigma'].value, 1000)

                                components = result.eval_components(x=x_comp)

                                # fit_line_comp = plt.plot(x_comp, components[prefix]+ background_result.eval(x=x_comp), color='purple', linewidth=0.5)  # Gaussian without background
                                # fit_lines.append(fit_line_comp[0])
                                
                                fit_line_comp_p_background = plt.plot(x_comp, components[prefix]+ background_result.eval(x=x_comp), color='red', linewidth=0.5)  # Gaussian and background
                                fit_lines.append(fit_line_comp_p_background[0])

                                fit_peak_line = ax.axvline(x=result.params[f'{prefix}center'].value, color='purple', linewidth=0.5) # adjust the peak to be the center of the fit
                                peak_markers.append(fit_peak_line)
                                
                                temp_fit_id = f"temp_fit_{len(temp_fits)}"
                                temp_fits[temp_fit_id] = {
                                    "fit_model": composite_model,
                                    "fit_result": result,
                                    "fit_line": fit_line_comp_p_background[0],
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
                        
                        # Change the color of the lines
                        stored_fits[fit_id]["fit_line"].set_color('purple')
                        stored_fits[fit_id]["background_line"].set_color('purple')
                        stored_fits[fit_id]["fit_p_background_line"].set_color('purple')
                                    
                        # Draw the stored fit lines      
                        ax.add_line(stored_fits[fit_id]["fit_line"])
                        ax.add_line(stored_fits[fit_id]["background_line"])
                        ax.add_line(stored_fits[fit_id]["fit_p_background_line"])
                    
                    fig.canvas.draw()
                    
                if event.key == "S":  # Save fits to file
                    if stored_fits:
                        formatted_results = {}  # Initialize a dictionary to store combined results

                        for fit_id, fit_data in stored_fits.items():
                            model_filename = f'{fit_id}_model.sav'
                            background_model_filename = f'{fit_id}_background_model.sav'
                            result_filename = f'{fit_id}_result.sav'
                            background_result_filename = f'{fit_id}_background_result.sav'

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

                            area = fit_params['amplitude'].value / hist_bin_width
                            area_uncertainty = fit_params['amplitude'].stderr / hist_bin_width

                            fit_parameters["volume"] = area
                            fit_parameters["volume_uncertainty"] = area_uncertainty

                            # Append the saved contents to the formatted_results dictionary
                            formatted_results[fit_id] = {
                                "fit_parameters": fit_parameters,
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

                        filename = input("Enter a filename to save the fits to: ")
                        # Save the formatted_results dictionary to a new file
                        with open(f"{filename}.sav", "w") as output_file:
                            output_file.write(str(formatted_results))
                        print(f"Saved fits to file: {filename}.sav")

                    else:
                        print("No fits to save")
                    
                if event.key == "L":  # Load fits from file
                    filename = input("Enter the filename to load fits from: ")
                    
                    if os.path.exists(f"{filename}.sav"):
                        with open(f"{filename}.sav", "r") as input_file:
                            loaded_fits = eval(input_file.read())
                            
                            for fit_id, fit_data in loaded_fits.items():
                                model_filename = f'{fit_id}_model.sav'
                                background_model_filename = f'{fit_id}_background_model.sav'
                                result_filename = f'{fit_id}_result.sav'
                                background_result_filename = f'{fit_id}_background_result.sav'
                                
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
                                
                                # Draw the fits
                                x = np.linspace(loaded_fit_result.params['center'].value - 4 * loaded_fit_result.params['sigma'].value,
                                                loaded_fit_result.params['center'].value + 4 * loaded_fit_result.params['sigma'].value, 1000)
                                
                                loaded_fit_line = plt.plot(x, loaded_fit_result.eval(x=x), color='red', linewidth=0.5)  # Just the gaussian
                                loaded_fit_p_background_line = plt.plot(x, loaded_fit_result.eval(x=x) + loaded_fit_background_result.eval(x=x), 'red', linewidth=0.5)  # gaussian + background
                                loaded_fit_background_line = plt.plot(x, loaded_fit_background_result.eval(x=x), 'green', linewidth=0.5)  # background
                                
                                fit_lines.extend([loaded_fit_line[0], loaded_fit_p_background_line[0], loaded_fit_background_line[0]])
                                
                                temp_fit_id = f"temp_fit_{len(temp_fits)}"
                                temp_fits[temp_fit_id] = {
                                    "fit_model": loaded_fit_model,
                                    "fit_result": loaded_fit_result,
                                    "fit_line": loaded_fit_line[0],
                                    "background_model": loaded_fit_background_model,
                                    "background_result": loaded_fit_background_result,
                                    "background_line": loaded_fit_background_line[0],
                                    "fit_p_background_line": loaded_fit_p_background_line[0]
                                }
                                
                                fig.canvas.draw()
                            
                            print(f"Loaded {len(loaded_fits)} fits from file: {filename}.sav")
        
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
        save_histogram:bool = False,
        cmap: str = None,
        ):

        import matplotlib.colors as colors
                
        # Concatenate the arrays horizontally to get the final result
        x_data = np.hstack([column[0] for column in data])
        y_data = np.hstack([column[1] for column in data])
        
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins, range=range)

        # draw a 2d histogram using matplotlib. If no ax is provided, make a new fig, ax using plt.subplots
        if subplots is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots 
            
            
        if cmap is None: "viridis"
            
        # draw a 2d histogram and add a similar stat bar to the 2d histogram, just the intergral though
        ax.hist2d(x_data, y_data, bins=bins, range=range, cmap=cmap, norm=colors.LogNorm())
        
        
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
            
        if save_histogram:
            plt.savefig(f"{ylabel}_{xlabel}.png",format='png', dpi=900)
            plt.savefig(f"{ylabel}_{xlabel}.pdf",format='pdf')
            
        # Keep track of the added lines for the x and y projections
        added_y_lines = []
        added_x_lines = []
        def on_press(event):  # Function to handle mouse click events like x/y projections
            if event.inaxes is ax and event.key == 'y': # For drawing lines to do a y-projection
                # Check if there are already two lines present
                if len(added_y_lines) >= 2:
                    # If two lines are present, remove them from the plot and the list
                    for line in added_y_lines:
                        line.remove()
                    added_y_lines.clear()
                
                # Get the x-coordinate of the current view
                x_coord = event.xdata
                
                # Add a vertical line to the 2D histogram plot
                line = ax.axvline(x_coord, color='red')
                
                # Append the line to the list of added lines
                added_y_lines.append(line)

                # Show the figure with the updated plot
                fig.canvas.draw()
                
            if event.key == 'Y': # For showing the y-projection
                
                if len(added_y_lines) < 2: 
                    print("Must have two lines dummy!")
                    
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
                    
            if event.inaxes is ax and event.key == 'x': # For drawing lines to do a y-projection
                # Check if there are already two lines present
                if len(added_x_lines) >= 2:
                    # If two lines are present, remove them from the plot and the list
                    for line in added_x_lines:
                        line.remove()
                    added_x_lines.clear()
                
                # Get the x-coordinate of the current view
                y_coord = event.ydata
                
                # Add a vertical line to the 2D histogram plot
                line = ax.axhline(y_coord, color='green')
                
                # Append the line to the list of added lines
                added_x_lines.append(line)

                # Show the figure with the updated plot
                fig.canvas.draw()
                
            if event.key == 'X': # For showing the X-projection
                
                if len(added_x_lines) < 2: 
                    print("Must have two lines dummy!")
                    
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
                    
        ax.figure.canvas.mpl_connect('key_press_event', on_press)
        
        fig.tight_layout()
        
        self.figures.append(fig)
        
        return hist, x_edges, y_edges
 
