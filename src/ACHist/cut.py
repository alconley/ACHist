import polars as pl
from matplotlib.path import Path
import numpy as np
import json
import os

# CutHandler and Cut2D classes were create by Gordon McCann

""" 
Handler to recieve vertices from a matplotlib selector (i.e. PolygonSelector).
Typically will be used interactively, most likely via cmd line interpreter. The onselect
method should be passed to the selector object at construction. CutHandler can also be used in analysis
applications to store cuts.
"""
class CutHandler:
    def __init__(self):
        self.cuts: dict[str, Cut2D] = {}

    def onselect(self, vertices: list[tuple[float, float]]):
        cut_default_name = f"cut_{len(self.cuts)}"
        self.cuts[cut_default_name] = Cut2D(cut_default_name, vertices)

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
    
    
def write_cut_json(cut: Cut2D, filepath):
    json_str = cut.to_json_str()
    try:
        with open(filepath, "w") as output:
            output.write(json_str)
            return True
    except OSError as error:
        print(f"An error occurred writing cut {cut.name} to file {filepath}: {error}")
        return False

def load_cut_json(filepath: str):
    try:
        with open(filepath, "r") as input:
            buffer = input.read()
            cut_dict = json.loads(buffer)
            if not "name" in cut_dict or not "vertices" in cut_dict:
                print(f"Data in file {filepath} is not the right format for Cut2D, could not load")
                return None
            return Cut2D(cut_dict["name"], cut_dict["vertices"])
    except OSError as error:
        print(f"An error occurred reading trying to read a cut from file {filepath}: {error}")
        return None

def reduce_df_with_cut(df:pl.DataFrame, CutFile: str, XColumn: str, YColumn: str):
    if os.path.exists(CutFile):
        
        cut = load_cut_json(CutFile)
        
        if XColumn in df.columns and YColumn in df.columns: # Check if XColumn and YColumn exist in the DataFrame
            df = df.filter(pl.col(XColumn).list.concat(YColumn).map(cut.is_cols_inside))   
            return df
        else:
            raise ValueError(f"'{XColumn}' and/or '{YColumn}' do not exist in the DataFrame columns.")
    else:
        raise FileNotFoundError(f"The file '{CutFile}' does not exist.")
    
