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

        "seq" - Sequential
                When plotting data that is gradually
                changing smoothly. Should be used
                to map quanitative data.

        "qual" - Qualitative
                 When plotting data that does not
                 depend on intensity, rather to
                 highlight the different types of
                 data.

        Returns a dictionary with keys as columns.
    """
    NPlots = len(Columns)             # Get the number plots to make
    
    """ Three cases depending on the number of plots """
    if NPlots == 1:
        Colors = ['rgb(252,141,89)']
    elif NPlots == 2:
        Colors = ['rgb(252,141,89)', 'rgb(153,213,148)']
    else:
        import colorlover as cl
        Colors = cl.scales[str(NPlots)][Colormap[0]][Colormap[1]]
    
    OutColors = dict()
    for Index, Column in enumerate(Columns):         # Set colours
        OutColors[Column] = Colors[Index]
    return OutColors

def GenerateColourMap(Data, Colormap=["div", "Spectral"]):
    """ Generate a linearly spaced colourmap """
    IntensityValues = np.linspace(0., 1., 10)
    Colors = cl.scales["10"][Colormap[0]][Colormap[1]]
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
                      "width": 800,
                      "height": 500,
                      
                     }
    return LayoutSettings

###############################################################################

""" Plotting Routines """

def PlotMarkersDataFrame(DataFrame, Columns=None, CustomPlotTypes=None, Labels=None):
    NPlots = len(DataFrame.keys())
    
    if Columns is None:
        Columns = list(DataFrame.keys())              # if nothing specified, plot them all
    
    """ Initialise plotting settings """
    Colors = GenerateColours(Columns)
    PlotTypes = dict()
    for Key in Columns:
        Exists = NT.CheckString(Key, ["Model", "Regression", "Fit", "Smoothed"])
        if Exists is True:                            # If the column is any of the above types
            Plots[Key] = "line"
        else:
            PlotTypes[Key] = "markers"                # set all default plots to markers
        if CustomPlotTypes is not None:               # if we specify the plot type,
            try:
                PlotTypes[Key] = CustomPlotTypes[Key] # then make it so.
            except KeyError:
                pass
    PlotSettings = dict()                             # Stores the plot settings for each plot
    Plots = []                                        # list of instances of plotly plots
    
    Layout = DefaultLayoutSettings()                  # Generate the default layout
    if Labels is not None:
        for Key in Labels:
            if Key == "X Label":
                Layout["xaxis"]["title"] = Labels["X Label"]
            if Key == "Y Label":
                Layout["yaxis"]["title"] = Labels["Y Label"]
    
    for Plot in Columns:
        PlotSettings[Plot] = DefaultPlotSettings(PlotTypes[Plot])    # Copy default settings
        if PlotTypes[Plot] is "markers":
            PlotFunction = go.Scatter
            PlotSettings[Plot]["marker"]["color"] = Colors[Plot]     # set marker colour
        elif PlotTypes[Plot] is "line":
            PlotFunction = go.Scatter
            PlotSettings[Plot]["line"]["color"] = Colors[Plot]       # set line colour
        else:
            PlotFunction = go.Scatter
            PlotSettings[Plot]["marker"]["color"] = Colors[Plot]
        PlotSettings[Plot]["name"] = Plot                            # set legend caption
        Plots.append(PlotFunction(x = DataFrame.index, 
                                  y = DataFrame[Plot], 
                                  **PlotSettings[Plot])
                    )
    iplot(dict(data=Plots, layout=Layout))
    return Plots, Layout

def SurfacePlot(DataFrame, Colourmap=["div", "Spectral"], Title=None):
    ZData = DataFrame.as_matrix()
    Map = GenerateColourMap(ZData)
    LayoutSettings = {"title": Title,
                      "autosize": False,
                      "width": 600,
                      "height": 600,
                      "plot_bgcolor":"rgb(230,230,230)"
                      }
    Trace = [go.Surface(z=ZData,
                        colorscale=Map
                        ),
            ]
    iplot(dict(data=Trace, layout=LayoutSettings))