# InteractiveWidgets.py

#!/bin/python

import matplotlib.pyplot as plt
from ipywidgets import *
from IPython.display import display  
from IPython.html import widgets
plt.style.use("fivethirtyeight")

class PyplotWidget:
	"""
	Class for a Pyplot widget
	The general idea behind this is to have
	LabView/Origin plotting flexibility
	without the suck of either of those
	programs
	
	Will initialise with some DataFrame
	with a pyplot figure

	Planned functionality:
	1. Have buttons for each data key and within that
	   display plotting options for each key.
	
	Attributes:
	PlotCheckContainer - A container for holding the
	                     checkboxes for selecting what
	                     plots to show. Takes PlotCheckboxes
	                     as a list of children to the container

	PlotTypes - Dictionary containing keys corresponding
	            to the dataframe columns for selecting
	            what kind of plot is shown.
	            
	DatatoPlot - Dictionary of plots that will be shown
	             and updated dynamically based on boolean
	             value.
	"""
	def __init__(self, DataFrame):
		self.Figure, self.Axes = plt.subplots(figsize=(12,5))
		self.DataFrame = DataFrame
		self.PlotCheckContainer = widgets.HBox()
		self.UpdatePlotButton = widgets.Button(description="Update plot")
		self.FigureOptionsButton = widgets.Button(description="Figure Options")
		self.Title = widgets.Text(description="Plot Title")
		self.XLabel = widgets.Text(description="X Axis Label")
		self.YLabel = widgets.Text(description="Y Axis Label")
		self.DatatoPlot = dict()
		self.PlotTypes = dict()
		self.PlotCheckboxes = []
		self.InitialisePlotTypes()                # Set up line plots
		self.InitialisePlots()
		self.DisplayWidgets()                     # Display the widgets in notebook
		self.UpdatePlotButton.on_click(self.UpdatePlot)
		self.FigureOptionsButton.on_click(self.DisplayFigureOptions)
		plt.legend()
		plt.show()

	def InitialisePlotTypes(self):
		""" Initial plots will all be lines
		"""
		for Key in self.DataFrame.keys():
			self.PlotTypes[Key] = "Line"

	def InitialisePlots(self):
		""" Figure out what kind of plot is requested
		    and then initialise it as scatter or line
		    plot
		"""
		for Key in self.DataFrame.keys():
			if self.PlotTypes[Key] == "Line":
				self.DatatoPlot[Key] = self.Axes.plot(self.DataFrame.index,
					                                  self.DataFrame[Key],
				    	                              label=Key)
			elif self.PlotTypes[Key] == "Scatter":
				self.DatatoPlot[Key] = self.Axes.scatter(self.DataFrame.index,
					                                     self.DataFrame[Key],
					                                     label=Key)

	def DisplayWidgets(self):
		self.PlotSelector()
		display(self.PlotCheckContainer)
		display(self.UpdatePlotButton)
		display(self.FigureOptionsButton)

	def DisplayFigureOptions(self, Blank):
		display(self.Title)
		display(self.XLabel)
		display(self.YLabel)

	def UpdatePlot(self, Blank):
		""" Function called by Update Button press
		    to set which plots are plotted
		"""
		plt.xlabel(self.XLabel.value)
		plt.ylabel(self.YLabel.value)
		plt.title(self.Title.value)
		#self.InitialisePlots()
		for Checkbox in self.PlotCheckContainer.children:
			if not Checkbox.value:
				plt.setp(self.DatatoPlot[Checkbox.description], visible=False)
			else:
				plt.setp(self.DatatoPlot[Checkbox.description], visible=True)
		plt.legend()
		plt.draw()

	def PlotSelector(self):
		""" Generate tickboxes for which
	    	Plots to display
		"""
		for Key in self.DataFrame.keys():
			self.PlotCheckboxes.append(widgets.Checkbox(description = Key,
			                              				value=True,
			                              				width=90
			                              				)
			                          )
		self.PlotCheckContainer.children = [Plot for Plot in self.PlotCheckboxes]

	def AddFunction(self):
		""" Ability to add a function
	    	That can be adjusted with sliders
	    	Planned functionality
	    
	    	Will need a way to specify which variable
	    	Is adjusted
		"""
	
	def FunctionSlider(self):
		""" Use self.function to get
	    	Variables and choose somehow what
	    	Variable to adjust with the slider
		"""
		