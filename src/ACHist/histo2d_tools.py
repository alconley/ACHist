import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from matplotlib.widgets import PolygonSelector
import matplotlib.colors as colors
from colorama import Fore, Style
from .cut import CutHandler, write_cut_json
from .histo1d_tools import histo1d

def histo2d(
        xdata: list,
        ydata: list,
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

        if isinstance(xdata, pl.Series): # checks if xdata is a polars series
            x_data = xdata.to_numpy()
            xcolumn_name = ydata.name
            
        if isinstance(ydata, pl.Series): # checks if xdata is a polars series
            y_data = ydata.to_numpy()
            ycolumn_name = ydata.name
        
        if isinstance(xdata, list): # if xdata is a list of polars series
            # Concatenate the arrays horizontally to get the final result
            x_data = np.concatenate([data.to_numpy() for data in xdata])
            xcolumn_name = '_'.join([item.name for item in xdata])
            
        if isinstance(ydata, list): # if xdata is a list of polars series
            # Concatenate the arrays horizontally to get the final result
            y_data = np.concatenate([data.to_numpy() for data in ydata])
            ycolumn_name = '_'.join([item.name for item in ydata])
            
        # x_data = np.hstack([column for column in xdata])
        # y_data = np.hstack([column for column in ydata])
        
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins, range=range)

        fig, ax = (plt.subplots() if subplots is None else subplots)
                    
        handler = CutHandler()
        
        selector = PolygonSelector(ax, onselect=handler.onselect)
        selector.set_active(False)
        
        if cmap is None: "viridis"
        
        h = ax.hist2d(x_data, y_data, bins=bins, range=range, cmap=cmap, norm=colors.LogNorm())
        
        if cbar: fig.colorbar(h[3], ax=ax)
            
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
        # ax.set_xlabel(xlabel)
        
        ax.set_xlabel(xlabel if xlabel is not None else xcolumn_name)
        ax.set_xlabel(ylabel if ylabel is not None else ycolumn_name)
        
        # ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        ax.minorticks_on()
        ax.tick_params(axis='both',which='minor',direction='in',top=True,right=True,left=True,bottom=True,length=2)
        ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,left=True,bottom=True,length=4)
        
        if display_stats: matplotlib_2DHistogram_stats(ax=ax, xdata=x_data, ydata=y_data, bins=bins)

        interactive_2DHistogram(subplot=(fig,ax), x_data=x_data, y_data=y_data, bins=bins, range=range, selector=selector, handler=handler)
        
        fig.tight_layout()
                
        return hist, x_edges, y_edges
 
def matplotlib_2DHistogram_stats(ax, xdata, ydata, bins):
    
    x_lims = ax.get_xlim()
    on_screen_x_data = xdata[(xdata >= x_lims[0]) & (xdata <= x_lims[1])] # Filter the data based on the new x-axis limits
    
    y_lims = ax.get_ylim()
    on_screen_y_data = ydata[(ydata >= y_lims[0]) & (ydata <= y_lims[1])] # Filter the data based on the new x-axis limits
    
    summed_data = np.sum(on_screen_x_data) + np.sum(on_screen_y_data)

    # Create the stats box
    stats = f"Integral: {summed_data:.0f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='black')
    text_box = ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    def on_lims_change(ax):
        # Get the new x-axis limits
        x_lims = ax.get_xlim()

        # Get the new y-axis limits
        y_lims = ax.get_ylim()
        
        # Create a mask to filter both x_data and y_data
        mask = (xdata >= x_lims[0]) & (xdata <= x_lims[1]) & (ydata >= y_lims[0]) & (ydata <= y_lims[1])

        filtered_x_data = xdata[mask]
        filtered_y_data = ydata[mask]
    
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
  
def get_x_projection_data(xdata, ydata, xmarkers):
    if len(xmarkers) < 2: 
        return print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")
    else:
        y_coordinates = []
        for line in xmarkers:
            y_coordinate = line.get_ydata()[0] 
            y_coordinates.append(y_coordinate)
        y_coordinates.sort() 
        
        y_mask = (ydata >= y_coordinates[0]) & (ydata <= y_coordinates[1])
            
        x_projection_data = xdata[y_mask]
        
        return x_projection_data, y_coordinates
    
def get_y_projection_data(xdata, ydata, ymarkers):
    if len(ymarkers) < 2: 
        print(f"{Fore.RED}{Style.BRIGHT}Must have two lines!{Style.RESET_ALL}")
        
    else:
        x_coordinates = []
        for line in ymarkers:
            x_coordinate = line.get_xdata()[0]  # Assuming the line is vertical and has only one x-coordinate
            x_coordinates.append(x_coordinate)
        x_coordinates.sort() 
        
        x_mask = (xdata >= x_coordinates[0]) & (xdata <= x_coordinates[1])
            
        y_projection_data = ydata[x_mask]
        
        return y_projection_data, x_coordinates
        
def interactive_2DHistogram(subplot: (plt.figure, plt.Axes), x_data, y_data, bins, range, selector, handler):

    fig, ax = subplot

    # Keep track of the added lines for the x and y projections
    y_markers = []
    x_markers = []
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
                if len(y_markers) >= 2:
                    # If two lines are present, remove them from the plot and the list
                    for line in y_markers:
                        line.remove()
                    y_markers.clear()
                
                x_coord = event.xdata
                line = ax.axvline(x_coord, color='red')
                y_markers.append(line)

                fig.canvas.draw()
                
            if event.key == 'Y': # For showing the y-projection
                
                y_projection_data, x_coordinates = get_y_projection_data(xdata=x_data, ydata=y_data, ymarkers=y_markers)
                histo1d(xdata=y_projection_data, bins=bins[1], range=range[1], title=f"Y-Projection: {round(x_coordinates[0], 2)} to {round(x_coordinates[1], 2)}")
                plt.show()
                    
            if event.key == 'x': # For drawing lines to do a x-projection
                # Check if there are already two lines present
                if len(x_markers) >= 2:
                    # If two lines are present, remove them from the plot and the list
                    for line in x_markers:
                        line.remove()
                    x_markers.clear()
                
                y_coord = event.ydata
                line = ax.axhline(y_coord, color='green')
                x_markers.append(line)

                fig.canvas.draw()
                
            if event.key == 'X': # For showing the X-projection
                
                x_projection_data, y_coordinates = get_x_projection_data(xdata=x_data, ydata=y_data, xmarkers=x_markers)
                histo1d(xdata=x_projection_data, bins=bins[0], range=range[0], title=f"X-Projection: {round(y_coordinates[0], 2)} to {round(y_coordinates[1], 2)}")
                plt.show()
                    
            if event.key == 'c': # create a cut

                print(f"{Fore.YELLOW}Activating the polygon selector tool:\n\tPress 'C' to save the cut (must enter cut name e.g. cut.json){Style.RESET_ALL}")
                
                selector.set_active(True)
                plt.show()
                
            if event.key == 'C': # save the cut to a file name that the user must enter
                selector.set_active(False)
                plt.show()
                
                handler.cuts["cut_0"].name = "cut"
                print(handler.cuts["cut_0"])
                
                # Prompt the user for the output file name
                output_file = input(f"{Fore.YELLOW}Enter a name for the output file (e.g., cut.json): {Style.RESET_ALL}")

                # Write the cut to the specified output file
                try:
                    write_cut_json(cut=handler.cuts["cut_0"], filepath=output_file)
                    print(f"{Fore.GREEN}Cut saved to '{output_file}' successfully.{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error: {e}. Failed to save the cut to '{Style.RESET_ALL}'.")
                    
    ax.figure.canvas.mpl_connect('key_press_event', on_press)