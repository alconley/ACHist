# '''
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import os
from cursor import *
from lmfit.models import GaussianModel, LinearModel, ExponentialModel
from pygame.locals import *

# tex_fonts = {
#                 # Use LaTeX to write all text
#                 # "text.usetex": True,
#                 "font.family": "serif",
#                 "font.serif" : ["CMR10"],
#                 # Use 10pt font in plots, to match 10pt font in document
#                 "axes.labelsize": 6,
#                 "font.size": 6,
#                 # Make the legend/label fonts a little smaller
#                 "legend.fontsize": 6,
#                 "xtick.labelsize": 6,
#                 "ytick.labelsize": 6
#             }

# plt.rcParams.update(tex_fonts)
# matplotlib.rcParams['axes.unicode_minus'] = False

class Histogrammer:

    def __init__(self):
           
        plt.rcParams['keymap.pan'].remove('p')
        plt.rcParams['keymap.home'].remove('r')
        plt.rcParams['keymap.fullscreen'].remove('f')
        plt.rcParams['keymap.grid'].remove('g')
        plt.rcParams['keymap.grid_minor'].remove('G')
        plt.rcParams['keymap.quit_all'].append('Q')
        
        self.figures = []
        self.cursor = None
            
    def guassian(self, x, volume, mean, fwhm, bin_width):

        sigma = fwhm / 2.355
        
        def f(x):
            return np.exp(- 0.5 * ( (x-mean)/(sigma) )**2)
        
        res, err = quad(f, mean - (3*sigma), mean + (3*sigma))
        amplitude = (volume / res) * bin_width
        
        return amplitude * np.exp( -(x-mean)**2 / (2*sigma**2) )
           
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
        hdtv_file_file:str = None,
        save_histogram:bool = False,
        output_file:str = None,
        display_stats:bool = True,
        display_cursor:bool = True,
    ):
        
        if isinstance(xdata, np.ndarray): # handles the case for the x/y projections 
            data = xdata
            column = ""
        else:
            if not isinstance(xdata, list): # Ensure that xdata is a list of objects, if not make it a list of 1
                xdata = [xdata]
                
            if len(xdata) < 2: # Concatenate 'name' attributes with underscores between them if there are multiple columns
                column = xdata[0].name if (len(xdata) > 0 and xdata[0].name in df.columns) else ""
            else:
                column_names = [item.name for item in xdata]
                column = '_'.join(column_names)
            
            data = np.concatenate([data.to_numpy() for data in xdata])
                
        hist_counts, hist_bins = np.histogram(data, bins=bins, range=range)
        
        hist_bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        
        if save_histogram:
            # Save the histogram data to a file
            
            if output_file==None:
                saved_name = f"{column}_hist.txt"
            else:
                saved_name = output_file
            
            with open(f"{saved_name}", "w") as f:
                f.write(f"# Counts\n")
                f.write(f"# Calibrate in HDTV with: calibration position set {range[0]-0.5} 1\n")
                
                for count in hist_counts:
                    f.write(f"{count}\n")
                    
        if subplots is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = subplots 
            
        if linewidth is None: linewidth = 0.5
        
        line, = ax.step(hist_bins[:-1], hist_counts, where='post', label=label, linewidth=linewidth, color=color, linestyle=linestyle,)
        ax.set_xlim(range)
        ax.set_ylim(bottom=0)
        
        ax.set_xlabel(xlabel if xlabel is not None else column)
        ax.set_ylabel(ylabel if ylabel is not None else "Counts")
        ax.legend() if label is not None else None
        ax.set_title(title)
        
        ax.minorticks_on()
        ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)

        if hdtv_file_file is not None: # Can draw hdtv fits on the histogram, just give it the path to the fit file
                    
            bin_width = (range[1] - range[0]) / bins
            
            if not os.path.exists(hdtv_file_file):
                print(f"{hdtv_file_file} does not exist")
                return
            
            cal_data = self.general_xml(hdtv_file_file)
            
            for fit in cal_data:
                pos, pos_uncertainty, fwhm, fwhm_uncertainty, volume, volume_uncertainty = fit
                
                x_values = np.linspace(pos - (3 * fwhm), pos + (3 * fwhm), 1000)
                y_values = self.guassian(x_values, volume, pos, fwhm, bin_width)
                
                ax.plot(x_values, y_values, linewidth=1, color='lightcoral')
             
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

        if display_cursor:   
            
            if self.cursor is None:
                # If the cursor instance doesn't exist, create it
                # self.cursor = AnnotatedCursor(line=line, ax=ax, useblit=True)
                
                self.cursor = AnnotatedCursor(
                line=line,
                numberformat="({0:.1f},{1:.1f})",
                dataaxis='x', offset=[10, 10],
                textprops={'color': 'black', 'fontweight': '1'},
                ax=ax,
                useblit=True,
                color='black',
                linewidth=0.5)
                
            # Simulate a mouse move to (-2, 10), needed for online docs
            t = ax.transData
            MouseEvent(
                "motion_notify_event", ax.figure.canvas, *t.transform((0, 0))
            )._process()

        region_markers = []
        peak_markers = []
        background_markers = []
        
        temp_fits = []
        stored_fits = []
          
        background_lines = []
        
        def fit_background(background_markers, hist_counts, hist_bin_centers):
            background_pos = [marker.get_xdata()[0] for marker in background_markers]
            background_pos.sort()


            background_y_values = [hist_counts[np.argmin(np.abs(hist_bin_centers - pos))] for pos in background_pos]
            background = LinearModel()
            # background = ExponentialModel()
            
            background_result = background.fit(background_y_values, x=background_pos)

            background_x_values = np.linspace(background_pos[0], background_pos[-1], 1000)
            background_values = background_result.eval(x=background_x_values)
            background_line = plt.plot(background_x_values, background_values, color='green', linewidth=0.5)
            
            hist_counts_subtracted = hist_counts - background_result.eval(x=hist_bin_centers)
                        
            return hist_counts_subtracted, background_result, background_line[0]
                
        def on_key(event):  # Function to handle fitting gaussians
            
            if event.inaxes is not None:    
                   
                if event.key == 'r': # region markers
                    
                    if len(region_markers) >= 2:
                        # If two lines are present, remove them from the plot and the list
                        for line in region_markers:
                            line.remove()
                        region_markers.clear()
                                        
                    region_pos = event.xdata
                    
                    line = ax.axvline(region_pos, color='#0B00E9', linewidth=0.5)
                    
                    region_markers.append(line)

                    fig.canvas.draw()
                    
                    pass
                    
                if event.key == 'b': # background markers

                    pos = event.xdata
                            
                    line = ax.axvline(x=pos, color='green', linewidth=0.5)

                    background_markers.append(line)

                    fig.canvas.draw()
                        
                    pass

                if event.key == 'p': # peak markers

                    pos = event.xdata
                            
                    line = ax.axvline(x=pos, color='purple', linewidth=0.5)

                    peak_markers.append(line)

                    fig.canvas.draw()
                        
                    pass
                
                if event.key == '-': # remove all markers and temp fits

                    for line in background_markers:
                        line.remove()
                    background_markers.clear()
                    
                    for line in peak_markers:
                        line.remove()  
                    peak_markers.clear()
                    
                    for line in region_markers:
                        line.remove()  
                    region_markers.clear()
                    
                    for line in background_lines:
                        line.remove()
                    background_lines.clear()
                    
                    for fit in temp_fits:
                        for line in fit:
                            line.remove()
                    temp_fits.clear()
                    
                    fig.canvas.draw()
                       
                if event.key == 'B':  # fit background
                    for line in background_lines:
                        line.remove()
                    background_lines.clear()
                    
                    hist_counts_subtracted, background_result, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
                    background_lines.append(background_line)
                    
                    fig.canvas.draw()
                      
                if event.key == 'f': # fit gaussian to region
                    
                    for fit in temp_fits:
                        for line in fit:
                            line.remove()
                    temp_fits.clear()
                    
                    if len(region_markers) != 2:
                        print("Must have two region markers!")
                    else:
                        
                        region_markers_pos = [region_markers[0].get_xdata()[0], region_markers[1].get_xdata()[0]]
                        region_markers_pos.sort()
                                                
                        fit_range = (hist_bin_centers >= region_markers_pos[0]) & (hist_bin_centers <= region_markers_pos[1])
                        
                        if len(background_markers) == 0:
                            hist_counts_subtracted, background_result, background_line = fit_background(region_markers, hist_counts, hist_bin_centers)
                        
                        else:
                            for line in background_lines:
                                line.remove()
                            background_lines.clear()

                            hist_counts_subtracted, background_result, background_line = fit_background(background_markers, hist_counts, hist_bin_centers)
     
                        model= GaussianModel()
                
                        model.set_param_hint('sigma', value=np.std(hist_counts_subtracted[fit_range]))
                        model.set_param_hint('center', value=np.mean(hist_counts[fit_range]))
                        model.set_param_hint('height', value=np.max(hist_counts_subtracted[fit_range]))
                        model.set_param_hint('amplitude', value=np.sum(hist_counts_subtracted[fit_range]))
                        
                        for marker in peak_markers:
                            if marker.get_xdata()[0] > region_markers_pos[0] and marker.get_xdata()[0] < region_markers_pos[1]:
                                model.set_param_hint('center', value=marker.get_xdata()[0])
                                break
                    
                        result = model.fit(hist_counts_subtracted[fit_range], x=hist_bin_centers[fit_range])

                        # # Print the fit results
                        # for param_name, param_value in result.params.items():
                        #     rounded_value = round(param_value.value, 2)
                        #     rounded_stderr = round(param_value.stderr, 2)
                        #     print(f"{param_name}: {rounded_value} +/- {rounded_stderr}")
                            
                        x_range = hist_bin_centers[fit_range]                    
                        area_bin_width = x_range[1] - x_range[0]
                        
                        area = result.params['amplitude'].value / area_bin_width
                        area_uncertainty = result.params['amplitude'].stderr / area_bin_width
                        
                        print(f"Mean: {round(result.params['center'].value,3)} +/- {round(result.params['center'].stderr,3)}")
                        print(f"FWHM: {round(result.params['fwhm'].value,3)} +/- {round(result.params['fwhm'].stderr,3)}")
                        print(f"Area: {round(area)} +/- {round(area_uncertainty)}\n")
                        
                        # plt.plot(hist_bin_centers[fit_range], result.best_fit, color='red', linewidth=0.5) #Just the gaussian
                        # plt.plot(hist_bin_centers[fit_range], result.best_fit + background_values[fit_range], 'r', linewidth=0.5) #gaussian + background
                                    
                        x = np.linspace(result.params['center'].value - 4*result.params['sigma'].value, result.params['center'].value + 4*result.params['sigma'].value, 1000)
            
                        gaus = plt.plot(x, result.eval(x=x), color='red', linewidth=0.5) #Just the gaussian
                        gaus_p_background = plt.plot(x, result.eval(x=x) + background_result.eval(x=x), 'red', linewidth=0.5) #gaussian + background
                        
                        temp_fits.append( [gaus[0], gaus_p_background[0], background_line] )
                        
                        fig.canvas.draw()
                        
                if event.key == 'F': # store the fits
                    
                    for fit in temp_fits:
                        stored_fits.append(fit)
                    # change the color of the lines to purple
                    
                    for fit in temp_fits:
                        for line in fit:
                            line.set_color('purple')
                    fig.canvas.draw()
                    
                    temp_fits.clear()
                        
                if event.key == 'g': # snake game

                    self.snake()

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
            
        # draw a 2d histogram and add a similar stat bar to the 2d histogram, just the intergral though
        ax.hist2d(x_data, y_data, bins=bins, range=range, norm=colors.LogNorm())
        
        
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
 
    def general_xml(self, file):
        
        import xml.etree.ElementTree as ET
        
        #Wrote by Bryan Kelly to extract the xml format fit file from HDTV to a easily readable table if wanted
        mytree = ET.parse(file)
        myroot = mytree.getroot()
    
        uncal_fit_list = []
        uncal_fit_err_list = []
        uncal_width_list = []
        uncal_width_err_list = []
        uncal_volume_list = []
        uncal_volume_err_list = []

        cal_fit_list = []
        cal_fit_err_list = []
        cal_width_list = []
        cal_width_err_list = []
        cal_volume_list = []
        cal_volume_err_list = []

        for fit in myroot:
            for i in fit:
                if i.tag == 'peak':
                    for child in i.iter():
                        if child.tag == 'uncal':
                            for j in child.iter():
                                if j.tag == 'pos':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            fit_value = newchild.text
                                            uncal_fit_list.append(round(float(fit_value), 4))
                                        elif newchild.tag == 'error':
                                            fit_err = newchild.text
                                            uncal_fit_err_list.append(round(float(fit_err),4))
                                elif j.tag == 'vol':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            vol_value = newchild.text
                                            uncal_volume_list.append(round(float(vol_value),4))
                                        elif newchild.tag == 'error':
                                            vol_err = newchild.text
                                            uncal_volume_err_list.append(round(float(vol_err),4))
                                elif j.tag == 'width':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            width_value = newchild.text
                                            uncal_width_list.append(round(float(width_value),4))
                                        elif newchild.tag == 'error':
                                            width_err = newchild.text
                                            uncal_width_err_list.append(round(float(width_err),4))

                        #gets the calibrated data information                
                        if child.tag == 'cal':
                            for j in child.iter():
                                if j.tag == 'pos':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            fit_value = newchild.text
                                            cal_fit_list.append(round(float(fit_value), 4))
                                        elif newchild.tag == 'error':
                                            fit_err = newchild.text
                                            cal_fit_err_list.append(round(float(fit_err),4))
                                elif j.tag == 'vol':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            vol_value = newchild.text
                                            cal_volume_list.append(round(float(vol_value),4))
                                        elif newchild.tag == 'error':
                                            vol_err = newchild.text
                                            cal_volume_err_list.append(round(float(vol_err),4))
                                elif j.tag == 'width':
                                    for newchild in j.iter():
                                        if newchild.tag == 'value':
                                            width_value = newchild.text
                                            cal_width_list.append(round(float(width_value),4))
                                        elif newchild.tag == 'error':
                                            width_err = newchild.text
                                            cal_width_err_list.append(round(float(width_err),4))
        
        cal_data = list(zip(cal_fit_list, cal_fit_err_list, cal_width_list, cal_width_err_list, cal_volume_list, cal_volume_err_list))

        return cal_data

    def snake(self):
        import pygame
        import time
        import random
        
        snake_speed = 15
        
        # Window size
        window_x = 720
        window_y = 480
        
        # defining colors
        black = pygame.Color(0, 0, 0)
        white = pygame.Color(255, 255, 255)
        red = pygame.Color(255, 0, 0)
        green = pygame.Color(0, 255, 0)
        blue = pygame.Color(0, 0, 255)
        
        garnet = pygame.Color(120, 47, 64)
        gold = pygame.Color(206, 184, 136)
        
        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('Snake')
        game_window = pygame.display.set_mode((window_x, window_y))
        
        # FPS (frames per second) controller
        fps = pygame.time.Clock()
        
        # defining snake default position
        snake_position = [100, 50]
        
        # defining first 4 blocks of snake body
        snake_body = [[100, 50],
                    [90, 50],
                    [80, 50],
                    [70, 50]
                    ]
        # fruit position
        fruit_position = [random.randrange(1, (window_x//10)) * 10,
                        random.randrange(1, (window_y//10)) * 10]
        
        fruit_spawn = True
        
        # setting default snake direction towards
        # right
        direction = 'RIGHT'
        change_to = direction
        
        # initial score
        score = 0
        
        # displaying Score function
        def show_score(choice, color, font, size):
        
            # creating font object score_font
            score_font = pygame.font.SysFont(font, size)
            
            # create the display surface object
            # score_surface
            score_surface = score_font.render('Score : ' + str(score), True, color)
            
            # create a rectangular object for the text
            # surface object
            score_rect = score_surface.get_rect()
            
            # displaying text
            game_window.blit(score_surface, score_rect)
        
        # game over function
        def game_over():
        
            # creating font object my_font
            my_font = pygame.font.SysFont('times new roman', 50)
            
            # creating a text surface on which text
            # will be drawn
            game_over_surface = my_font.render(
                'Your Score is : ' + str(score), True, red)
            
            # create a rectangular object for the text
            # surface object
            game_over_rect = game_over_surface.get_rect()
            
            # setting position of the text
            game_over_rect.midtop = (window_x/2, window_y/4)
            
            # blit will draw the text on screen
            game_window.blit(game_over_surface, game_over_rect)
            pygame.display.flip()
            
            # after 2 seconds we will quit the program
            time.sleep(2)
            
            # deactivating pygame library
            pygame.quit()
            
            # quit the program
            quit()
        
        
        # Main Function
        while True:
            
            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        change_to = 'RIGHT'
        
            # If two keys pressed simultaneously
            # we don't want snake to move into two
            # directions simultaneously
            if change_to == 'UP' and direction != 'DOWN':
                direction = 'UP'
            if change_to == 'DOWN' and direction != 'UP':
                direction = 'DOWN'
            if change_to == 'LEFT' and direction != 'RIGHT':
                direction = 'LEFT'
            if change_to == 'RIGHT' and direction != 'LEFT':
                direction = 'RIGHT'
        
            # Moving the snake
            if direction == 'UP':
                snake_position[1] -= 10
            if direction == 'DOWN':
                snake_position[1] += 10
            if direction == 'LEFT':
                snake_position[0] -= 10
            if direction == 'RIGHT':
                snake_position[0] += 10
        
            # Snake body growing mechanism
            # if fruits and snakes collide then scores
            # will be incremented by 10
            snake_body.insert(0, list(snake_position))
            if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
                score += 10
                fruit_spawn = False
            else:
                snake_body.pop()
                
            if not fruit_spawn:
                fruit_position = [random.randrange(1, (window_x//10)) * 10,
                                random.randrange(1, (window_y//10)) * 10]
                
            fruit_spawn = True
            game_window.fill(gold)
            
            for pos in snake_body:
                pygame.draw.rect(game_window, garnet,
                                pygame.Rect(pos[0], pos[1], 10, 10))
            pygame.draw.rect(game_window, black, pygame.Rect(
                fruit_position[0], fruit_position[1], 10, 10))
        
            # Game Over conditions
            if snake_position[0] < 0 or snake_position[0] > window_x-10:
                game_over()
            if snake_position[1] < 0 or snake_position[1] > window_y-10:
                game_over()
        
            # Touching the snake body
            for block in snake_body[1:]:
                if snake_position[0] == block[0] and snake_position[1] == block[1]:
                    game_over()
        
            # displaying score continuously
            show_score(1, black, 'times new roman', 20)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            fps.tick(snake_speed)
            
            pass

# df = pl.read_parquet("~/SanDisk/Projects/52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")
df = pl.read_parquet("~/Projects/52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")

#52Cr(d,p)53Cr, 8.3 kG field 
xavg_ECal_Para = [-0.0023904378617156377,-18.49776562220117, 1457.4874219091237]

a, b, c = xavg_ECal_Para
df = df.with_columns( ( (a*pl.col(f"Xavg")*pl.col(f"Xavg")) + (b*pl.col(f"Xavg")) + (c) ).alias(f"XavgEnergyCalibrated"))


# Protons coincidence of the 52Cr at the 8.3 kG field (run 82 through 112)
cebraTimeGate = [ [0,-1158,-1152],
                  [1,-1157,-1151],
                  [2,-1157,-1151],
                  [3,-1156,-1150],
                  [4,-1126,-1120] ]

ECal_cebraE0_Para = [0, 1.7551059351549314, -12.273506897222896]
ECal_cebraE1_Para = [0, 1.9510278378962256, -16.0245754973971]
ECal_cebraE2_Para = [0, 1.917190081718234, 16.430212777833802]
ECal_cebraE3_Para = [0, 1.6931918955746692, 12.021258506937766]
ECal_cebraE4_Para = [0, 1.6373533248536343, 13.091030061910748]

CeBrA_ECal = [ECal_cebraE0_Para, ECal_cebraE1_Para, ECal_cebraE2_Para, ECal_cebraE3_Para, ECal_cebraE4_Para]

det_dfs = []
NumDetectors = 5

pg_data = []
for det in range(NumDetectors):
    
    a,b,c = CeBrA_ECal[det]
    df = df.with_columns( ( a*pl.col(f"Cebra{det}Energy")*pl.col(f"Cebra{det}Energy") + b*pl.col(f"Cebra{det}Energy") + c).alias(f"Cebra{det}EnergyCalibrated"))
    
    det_df = df.filter( pl.col(f"Cebra{det}TimetoScint") > cebraTimeGate[det][1] ).filter( pl.col(f"Cebra{det}TimetoScint") < cebraTimeGate[det][2] )
    det_dfs.append(det_df)

    pg_data.append( (det_df["XavgEnergyCalibrated"], det_df[f"Cebra{det}EnergyCalibrated"]) )

h = Histogrammer()

# fig, ax = plt.subplots(2,2, figsize=(10,10))
# ax = ax.flatten()

# h.histo1d(xdata=df["XavgEnergyCalibrated"], bins=500, range=(0,6000))
h.histo1d(xdata=[df["Cebra0Energy"]], bins=512, range=(0,4096))

# h.histo2d(data=[ (df["ScintLeftEnergy"],df["AnodeBackEnergy"])], bins=[512,512], range=[ [0,2048], [0,2048]])

# for i in range(5):
#     h.histo2d(data=[[det_dfs[i]["Xavg"],det_dfs[i][f"Cebra{i}EnergyCalibrated"]]] , bins=[600,250], range=[[-300,300], [0,6000]], xlabel="Xavg", ylabel=f"Cebra{i}Energy")


# fig, axs = plt.subplots(1,1, figsize=(5,5))
# fig1, axs1 = plt.subplots(1,1, figsize=(10,5))

# h.histo1d(xdata=df["XavgEnergyCalibrated"], bins=500, range=(0,6000), subplots=(fig1,axs1))

# h.histo2d(data=pg_data, bins=[400,400], range=[[0,5200], [0,5200]], subplots=(fig,axs), xlabel=r"$^{53}$Cr Energy [keV]",ylabel=r"$\gamma$-ray Energy [keV]",display_stats=False)

# x = np.linspace(0,5200,5200)
# gs = x 
# first_excited = x - 564
# second_excited = x - 1006
# third_excited = x - 1289

# axs.plot(x,gs, color='#17a657',             linewidth=1, label=r'$\frac{3}{2}^{-}$ Band', alpha=0.8, linestyle='-')
# axs.plot(x,first_excited, color='#751a9c',  linewidth=1, label=r'$\frac{1}{2}^{-}$ Band', alpha=0.8, linestyle='--')
# axs.plot(x,second_excited, color='#a61753', linewidth=1, label=r'$\frac{5}{2}^{-}$ Band', alpha=0.8, linestyle='-.')
# axs.plot(x,third_excited, color='#251a9c',  linewidth=2, label=r'$\frac{7}{2}^{-}$ Band', alpha=0.8, linestyle=':')

# axs.legend(loc='upper left',shadow=False, frameon=True, fancybox=False, edgecolor='none', facecolor='none')

# fig.subplots_adjust(top=0.985,
# bottom=0.097,
# left=0.137,
# right=0.983,
# hspace=0.2,
# wspace=0.2)

# plt.savefig(f"53Cr_pg_matrix.pdf",format='pdf')
# plt.savefig(f"53Cr_pg_matrix.png",format='png',dpi=900)

plt.show()



# '''

