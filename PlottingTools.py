# PlottingTools.py

""" Set of routines for generating plots,
    Kelvin style.
"""

from plotly.offline import *
import plotly.graph_objs as go
import colorlover as cl
import NotebookTools as NT
import numpy as np

###############################################################################

""" Initialisation """

init_notebook_mode(connected=True) 

###############################################################################

""" Common routines """

def GenerateRandomPlot(NPoints, Settings):
    x = np.random.rand(NPoints)
    y = np.random.rand(NPoints)
    return go.Scatter(x=x, y=y, **Settings)

def GenerateColours(Columns, Colormap=["div", "Spectral"]):
    """ Generates a set of colours for Plotly 

        Takes input as number of plots (integer)
        and a list containing the type of colourmap
        and the specific colourmap.

        Choices are:

        "div" - Diverging
                When plotting data that is supposed to
                highlight a middle ground; i.e.
                contrast between max and min data,
                effectively to show how the data
                deviates from the norm.
        ['RdYlBu','Spectral','RdYlGn','PiYG','PuOr','PRGn','BrBG','RdBu','RdGy']

        "seq" - Sequential
                When plotting data that is gradually
                changing smoothly. Should be used
                to map quanitative data.
        ['Reds','YlOrRd','RdPu','YlOrBr','Greens','YlGnBu','GnBu','BuPu','Greys',
         'Oranges','OrRd','BuGn','PuBu','PuRd','Blues','PuBuGn','YlGn','Purples']

        "qual" - Qualitative
                 When plotting data that does not
                 depend on intensity, rather to
                 highlight the different types of
                 data.

        ['Pastel2', 'Paired', 'Pastel1', 'Set1', 'Set2', 'Set3', 'Dark2', 'Accent']

        Returns a dictionary with keys as columns.
    """
    NPlots = len(Columns)             # Get the number plots to make
    
    """ Three cases depending on the number of plots """
    if NPlots == 1:
        Colors = ['rgb(252,141,89)']                            # Just red for one plot
    elif NPlots == 2:
        Colors = ['rgb(252,141,89)', 'rgb(153,213,148)']        # Red and green for two plots
    else:
        import colorlover as cl
        Colors = cl.scales[str(NPlots)][Colormap[0]][Colormap[1]]  # For n > 2, moar colours
    
    OutColors = dict()
    for Index, Column in enumerate(Columns):         # Set colours
        OutColors[Column] = Colors[Index]
    return OutColors

def GenerateColourMap(Data, Colormap=["div", "Spectral"]):
    """ Generate a linearly spaced colourmap """
    IntensityValues = np.linspace(0., 1., 5)        # Z scale cmap is normalised
    Colors = cl.scales["5"][Colormap[0]][Colormap[1]]
    Map = []
    for Index, Value in enumerate(IntensityValues):
        Map.append([Value, Colors[Index]])
    return Map

###############################################################################

""" Default/Initialisation routines """

def DefaultPlotSettings(PlotType="markers"):
    PlotSettings = dict()
    if PlotType == "line":                               # line plot defaults
        PlotSettings = {"mode": PlotType,
                        "line": {"width": 2.,
                                 "color": "blue",
                                },
                        }
    elif PlotType == "markers" or "marker":              # marker plot defaults
        PlotSettings = {"mode": "markers",
                        "marker": {"size": 12.,
                                   "color": "blue"
                                  },
                        }
    return PlotSettings

def DefaultLayoutSettings():
    """ The default layout settings, such as Axis
        labels, plot background colour and size
    """
    LayoutSettings = {"xaxis":{"title": "X Axis",             # x axis
                               "titlefont": {"size": 18.},
                               "tickfont": {"size": 14.}
                              },
                      "yaxis":{"title": "Y Axis",             # y axis
                               "titlefont": {"size": 18.},
                               "tickfont": {"size": 14.}
                              },
                      "plot_bgcolor": 'rgb(229,229,229)',     # bkg colour
                      "autosize": False,                      # set manual size
                      "width": 950,
                      "height": 500,
                      
                     }
    return LayoutSettings

def FilterPlotTypes(Columns, PlotTypes, CustomPlotTypes):
    CheckList = ["Model", "Regression", "Fit", "Smoothed", "Stick Spectrum"]
    for Key in Columns:
        if (type(Key) is str) == True:
            Exists = NT.CheckString(Key, CheckList)
            if Exists is True:                            # If the column is any of the above types
                PlotTypes[Key] = "line"
            else:
                PlotTypes[Key] = "markers"                # set all default plots to markers
        else:
            for Key in Columns:
                PlotTypes[Key] = "markers"
        if CustomPlotTypes is not None:               # if we specify the plot type,
            try:
                PlotTypes[Key] = CustomPlotTypes[Key] # then make it so.
            except KeyError:
                pass
    return PlotTypes

###############################################################################

""" Plotting Routines """

def XYPlot(DataFrame, Columns=None, CustomPlotTypes=None, Labels=None):
    """ Use Plotly to plot a dataframe using XY markers.
        Some optional arguments are:

        Columns as a list of dataframe keys to choose which
        series to plot

        CustomPlotTypes will change the type (line or marker)
        of plot to be made for that series. Input as a dictionary
        with the dataframe keys and specify the plot type by a string.

        e.g. {"Y Range": "line"}

        Labels will let you change the title and axis labels. 

    """
    
    if Columns is None:
        Columns = list(DataFrame.keys())              # if nothing specified, plot them all
    
    PlotTypes = dict()
    PlotSettings = dict()                             # Stores the plot settings for each plot
    Plots = []                                        # list of instances of plotly plots

    """ Initialise plotting settings """
    Colors = GenerateColours(Columns)                 # Set up colours based on colourmap
    PlotTypes = FilterPlotTypes(Columns, PlotTypes, CustomPlotTypes)# Check for keys to set default plots
    
    Layout = DefaultLayoutSettings()                  # Generate the default layout
    if Labels is not None:
        for Key in Labels:
            if Key == "X Label":
                Layout["xaxis"]["title"] = Labels["X Label"]
            if Key == "Y Label":
                Layout["yaxis"]["title"] = Labels["Y Label"]
            if Key == "Title":
                Layout["title"] = Labels["Title"]
    
    for Plot in Columns:
        PlotSettings[Plot] = DefaultPlotSettings(PlotTypes[Plot])    # Copy default settings
        if PlotTypes[Plot] is "markers":
            PlotFunction = go.Scatter
            PlotSettings[Plot]["marker"]["color"] = Colors[Plot]     # set marker colour
        elif PlotTypes[Plot] is "line":
            PlotFunction = go.Scatter
            PlotSettings[Plot]["line"]["color"] = Colors[Plot]       # set line colour
            if Plot == "Stick Spectrum":
                PlotSettings[Plot]["opacity"] = 0.5                  # make sticks more transparent
        else:
            PlotFunction = go.Scatter
            PlotSettings[Plot]["marker"]["color"] = Colors[Plot]
        PlotSettings[Plot]["name"] = Plot                            # set legend caption
        Plots.append(PlotFunction(x = DataFrame.index, 
                                  y = DataFrame[Plot], 
                                  **PlotSettings[Plot])
                    )
    iplot(dict(data=Plots, layout=Layout))

def XYZPlot(DataFrame, Type="Heatmap", Colourmap=["div","Spectral"], Labels=None):
    """ A general wrapper for three dimensional data.
        Takes a dataframe and plots the data as xyz.

        The X data is taken from the dataframe index,
        while the Y data is taken as the column headings.

        Choices are:
        Heatmap, Contour, Surface and Scatter.

        The others are more obvious, but for Scatter the Z
        dimension is given as colour intensity.
    """
    """ Assign data to variables """
    ZData = DataFrame.as_matrix()
    XData = DataFrame.index
    YData = list(DataFrame.keys())

    """ Generate colour map and layout settings """
    if len(Colourmap) == 2:                             # this case depends on what's passed as Colourmap
        Map = GenerateColourMap(ZData, Colourmap)       # if it's a 2-tuple string, it's specifying colorlover
    else:
        Map = Colourmap                                 # otherwise it's specifying the name of the colourmap

    LayoutSettings = DefaultLayoutSettings()                         # initialise layout
    if Labels is not None:                                           # change to specified labels
        for Key in Labels:
            if Key == "Title":
                LayoutSettings["title"] = Labels[Key]
            if Key == "X Label":
                LayoutSettings["xaxis"]["title"] = Labels[Key]
            if Key == "Y Label":
                LayoutSettings["yaxis"]["title"] = Labels[Key]

    """ Set up the plot parameters """
    Parameters = {"x": XData,
                  "y": YData,
                  "z": ZData,
                  "colorscale": Map,
                  }

    """ Now set up the plot type specific settings """
    if Type == "Heatmap":
        Function = go.Heatmap
    elif Type == "Contour":
        Function = go.Contour
        Parameters["line"] = {"smoothing": 0.7}        # smoothes the contour lines
    elif Type == "Surface":
        Function = go.Surface
    elif Type == "Scatter":
        Function = go.Scatter
        del Parameters["z"]
        del Parameters["colorscale"]
        Parameters["marker"] = {"size": 12.,
                                "color": ZData,
                                "colorscale": Map,
                                "showscale": True
                                }

    Trace = [Function(**Parameters)]
    iplot(dict(data=Trace, layout=LayoutSettings))