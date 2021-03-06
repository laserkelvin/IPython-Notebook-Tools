# InteractiveWidgets.py

#!/bin/python

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from itertools import cycle

plt.style.use("fivethirtyeight")
class PlotContainerGUI:
    """ PlotContainerGUI

    An object that contains meta-objects that cam
        be used as a general pyplot interface with 
        interactive widgets to control the plotting 
        styles.

        Only requires initial input of a Pandas dataframe,
        everything should be self-explanatory from then
        onwards.

        Two sub-classes are used in PlotContainerGUI:
        1. FigureSetup
            Container with the general figure settings
            like style sheet and axis labels.
        2. ClassReferences
            Objects that will control the plot settings
            for individual plots. This is a child of the
            object MainTabs, which houses each ClassReference 
            object as tabs.

        General way this was written is 
    """
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame
        # Generate a list of what the tabs will be called
        self.PlotTabs = [str(Key) for Key in DataFrame.keys()]
        # For each tab, generate a PlotSettings object (tab menu objects)
        self.ClassReferences = [self.PlotSettings(Key) for Key in self.PlotTabs]
        # Generate a list of references to the tabs
        self.TabReferences = [Key.SettingsContainer for Key in self.ClassReferences]
        # Populate the tab menu with tabs
        self.MainTabs = widgets.Tab(children=self.TabReferences)
        self.FigureSetup = self.FigureSettings()
        self.PlotSettings = dict()                # Needs and wants of all each plot 
        self.DatatoPlot = dict()                  # Dictionary of instances of plots
        display(self.MainTabs)
        display(self.FigureSetup.Container)
        self.InitialisePlots()
        plt.show()
        self.FigureSetup.UpdateFigure.on_click(self.UpdatePlot)
        self.UpdateNames()
    
    def UpdateNames(self):
        for Index, Key in enumerate(self.PlotTabs):
            self.MainTabs.set_title(Index, Key)
            
    def UpdateFigureSettings(self):
        self.Settings = dict()
        self.Settings = self.FigureSetup.GetSettings()
        plt.xlabel(self.Settings["XLabel"])
        plt.ylabel(self.Settings["YLabel"])
        plt.title(self.Settings["PlotTitle"])
        plt.style.use(self.Settings["Style"])
        
    def DefineColours(self):
        """ Method for generating a colour palette. 
            This is done by checking how many plots are
            actually going to be plotted by their Booleans,
            then generating a 1D array with the RGB values
            for a specific colourmap.

        """
        # Count the total number of plots we're going to make; sum of boolean values
        PlotList = []
        for Key in self.ClassReferences:
            self.PlotSettings[Key.Name.value] = Key.GetSettings()
            PlotList.append(self.PlotSettings[Key.Name.value]["PlotBoolean"])
        self.PlotCount = np.sum(PlotList)
        try:
            ColourMap = cm.__dict__[self.FigureSetup.PlotColours.value]
        except KeyError:
            ColourMap = cm.Spectral                                   # Default to spectral, good for several plots
        if self.PlotCount <= 2:
            ColourGenerator = cycle(["red", "green", "blue"])
            self.Colours = [Colour for Colour, Key in zip(ColourGenerator,
                                                          self.DataFrame.keys())]  # if there's only one plot, make it red.
        elif self.PlotCount > 2:
            self.Colours = ColourMap(np.linspace(0, 1, self.PlotCount))   # This generates enough colours

    def InitialisePlots(self):
        """ Method for plotting data. This will reference a figure
            called "Main", and generate plots in that figure.

            Before the plotting is done, the figure settings are
            retrieved from FigureSetup, called in UpdateFigureSettings.
        """
        plt.figure("Main", figsize=(12,6))
        self.UpdateFigureSettings()
        self.DefineColours()          # Generates colour palette as well as retrieves plot settings
        for Key, Colour in zip(self.ClassReferences, self.Colours):
            if self.PlotSettings[Key.Name.value]["PlotColour"] == "Default":
                self.DatatoPlot[Key.Name.value] = plt.plot(self.DataFrame.index,
                                                           self.DataFrame[Key.DataReference],
                                                           label=Key.Name.value,
                                                           marker=self.PlotSettings[Key.Name.value]["PlotType"],
                                                           alpha=0.8,
                                                           linestyle=":",
                                                           markersize=10,
                                                          )
            else:
                self.DatatoPlot[Key.Name.value] = plt.plot(self.DataFrame.index,
                                                           self.DataFrame[Key.DataReference],
                                                           label=Key.Name.value,
                                                           marker=self.PlotSettings[Key.Name.value]["PlotType"],
                                                           alpha=0.8,
                                                           linestyle=":",
                                                           markersize=10,
                                                           c=Colour
                                                          )
        plt.legend()

    def UpdatePlot(self, Blank):
        """ This method is called each time
            the Update Plot button is clicked.

            It sets the current figure to "Main",
            clears it and re-plots with the latest
            figure settings.

            After plotting, it checks if the PlotBoolean
            checkbox is clicked; if it is then we make the
            plot visible. Otherwise, we make it invisible.
        """
        plt.figure("Main")
        plt.clf()
        self.InitialisePlots()
        for Key in self.ClassReferences:
            if not self.PlotSettings[Key.Name.value]["PlotBoolean"]:
                plt.setp(self.DatatoPlot[Key.Name.value], visible=False)
            else:
                plt.setp(self.DatatoPlot[Key.Name.value], visible=True)
        plt.legend()
        plt.draw()
    
    class FigureSettings:
        """ General plotting settings, such as colours and whatnot """
        def __init__(self):
            ColourMaps = ["Default", "viridis", "inferno", "magma",
                          "Spectral", "Pastel1", "coolwarm"]
            self.Style = widgets.Dropdown(description="Plot Style Sheet",
                                          value="seaborn-pastel",
                                          options=plt.style.available)
            self.PlotColours = widgets.Dropdown(description="Plot Colours",
                                                options=ColourMaps,
                                                value="Default")
            self.UpdateFigure = widgets.Button(description="Update plot")
            self.XLabel = widgets.Text(description="X Axis Label",
                                       value="X Axis",
                                       width=120)
            self.YLabel = widgets.Text(description="Y Axis Label",
                                       value="Y Axis",
                                       width=120)
            self.PlotTitle = widgets.Text(description="Plot Title",
                                          value="Main",
                                          width=120)
            self.Container = widgets.HBox(children=[self.UpdateFigure,
                                                    self.Style,
                                                    self.PlotColours,
                                                    self.XLabel,
                                                    self.YLabel,
                                                    self.PlotTitle
                                                   ],
                                          padding=0)

        def GetSettings(self):
            """ Method for retrieving a dictionary
                of all of the plot settings.
            """
            self.Settings = dict()
            for Setting in self.__dict__:
                try:
                    self.Settings[Setting] = self.__dict__[Setting].value
                except AttributeError:         # Ignore what we can't get!
                    pass
            return self.Settings
    
    class PlotSettings:
        """ Interactive widgets for each key in a dataframe to plot """
        def __init__(self, Key):
            Types = Line2D.filled_markers
            self.DataReference = Key
            self.Name = widgets.Text(description="Label",         # This only changes
                                     value=Key)                   # the plot label!
            self.PlotColour = widgets.Dropdown(description="Plot Colour",
                                               options=["red",
                                                        "blue",
                                                        "green",
                                                        "cyan",
                                                        "magenta",
                                                        "black"])
            self.PlotType = widgets.Dropdown(description="Plot Type",
                                             options=Types,
                                             value="o"
                                             )
            self.PlotBoolean = widgets.Checkbox(description="Show Plot?",
                                                value=False)
            self.SettingsContainer = widgets.HBox(children=[self.PlotBoolean,
                                                            self.PlotType,
                                                            #self.PlotColour,
                                                            self.Name
                                                            ],
                                                  padding=20)
        def GetSettings(self):
            """ Returns a dictionary of the plot settings """
            self.Settings = dict()
            for Setting in self.__dict__:
                try:
                    self.Settings[Setting] = self.__dict__[Setting].value
                except AttributeError:
                    pass
            return self.Settings