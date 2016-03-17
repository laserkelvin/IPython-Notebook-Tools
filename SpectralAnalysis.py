#!/bin/python

import numpy as np
import pandas as pd
import NotebookTools as NT
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from scipy import constants
from scipy import signal
from scipy.optimize import curve_fit
from scipy import interpolate
from bokeh.palettes import brewer
from bokeh.plotting import figure, show

# SpectralAnalysis.py

# Module containing functions for day-to-day spectral analysis, including re-packaged
# routines from FittingRoutines.py
# My plan is to make FittingRoutines obsolete, as it has A LOT of poorly written
# routines.

###################################################################################################

""" Classes """

class Spectrum:
	def __init__(self, File, Reference=None):
		self.Data = LoadSpectrum(File)
		self.PumpWavelength = 0.
		self.ProbeWavelength = 0.
		if Reference != None:                       # If we give it a logbook reference
			self.Reference = Reference
	def AddData(self, NewData, Name):
		self.Data[Name] = NewData
	def DeleteData(self, Name):
		del self.Data[Name]
	def PlotAll(self, Labels=None, Interface="pyplot"):
		self.PlotLabels(Labels)                     # Initialise labels
		try:
			self.Labels["Title"] = self.Reference
		except KeyError:                            # Give up if we never gave it a logbook
			pass
		PlotData(self.Data, self.Labels, Interface)
	def PlotLabels(self, Labels=None):
		if Labels == None:                          # use default labels
			Labels = {"X Label": "X Axis",
					  "Y Label": "Y Axis",
					  "Title": " ",
					  "X Limits": [0, 100],
					  "Y Limits": [0, 1.],
					 }
		self.Labels = Labels

class Model:
	""" Class for fitting models and functions. Ideally, I'll have a
	pickle file containing a lot of these model instances so that
	they'll be out of the way

	Input:
	Variables - Dictionary containing all of the variables required,
	with the keys named exactly as the function requests.

	ObjectiveFunction - Reference to a model function
	"""
	def __init__(self, FunctionName):
		self.FunctionName = FunctionName
		self.Variables = {}
		print " Please call SetFunction(Function)"
	def SetFunction(self, ObjectiveFunction):
		""" Sets what the model functionw will be. Will also automagically
		initialise a dictionary that will hold all of the variables required,
		as well as the boundary conditions
		"""
		self.Function = ObjectiveFunction
		self.Variables = dict.fromkeys(ObjectiveFunction.__code__.co_varnames)
		try:
			del self.Variables["x"]                  # X keeps getting picked up, get rid of it
		except KeyError:
			pass
		self.BoundaryConditions = ([-np.inf for Variable in self.Variables],
			                       [np.inf for Varaible in self.Variables])
		print " Initialised variable dictionary:"
		print self.Variables
	def SetVariables(self, Variables):
		self.Variables = Variables
		print " Variables set to:\t" + str(self.Variables)
	def SetBounds(self, Bounds):
		""" Method for setting the boundary conditions for curve_fit,
		requires input as 2-tuple list ([], [])
		"""
		self.BoundaryConditions = Bounds
		print " Boundary conditions set to:\t" + str(self.BoundaryConditions)
	def ResetAttributes(self):
		""" Wipes the variable and boundary conditions
		"""
		self.SetFunction(self.Function)

###################################################################################################

""" Formatting and Fitting """

def FormatData(X, Y):
	""" Function to format data into a pandas data frame for
	fitting. In case I'm too lazy to set it up myself.
	"""
	return pd.DataFrame(data=Y, columns=["Y Range"], index=X)

def FitModel(DataFrame, Model):
	""" Uses an instance of the Model class to fit data contained
	in the pandas dataframe. Dataframe should have indices of the X-range
	and column "Y Range" as the Y data to be fit to

	Requires input of a DataFrame formatted using FormatData, and a reference
	to an instance of Model.

	Returns the optimised parameters, a fitting report as a dataframe,
	the fitted curves for easy plotting, and the covariance matrix.
	"""
	print " Curve fitting with:\t" + Model.FunctionName
	print " Boundary Conditions:"
	print str(Model.BoundaryConditions)
	print " Initial parameters:"
	OptimisedParameters, CovarianceMatrix = curve_fit(Model.Function,
														  DataFrame.index,
														  DataFrame["Y Range"], 
														  UnpackDict(**Model.Variables),
														  bounds=Model.BoundaryConditions)
	ParameterReport = pd.DataFrame(data=OptimisedParameters,
			                           index=Model.Variables.keys())
	ModelFit = Model.Function(DataFrame.index, *OptimisedParameters)
	FittedCurves = pd.DataFrame(data=zip(DataFrame["Y Range"], ModelFit),
			                        columns=["Data", "Model Fit"],
			                        index=DataFrame.index)
	print " ------------------------------------------------------"
	print " Parameter Report:"
	print ParameterReport
	return OptimisedParameters, ParameterReport, FittedCurves, CovarianceMatrix

###################################################################################################

""" Commonly used base functions """

def BaseGaussian(x, x0):
	""" The most vanilla of Gaussians, wrote it when I was debugging
	"""
	return np.exp(-np.square(x-x0))

def GaussianFunction(x, Amplitude, Centre, Width):
	return Amplitude * (1 / (Width * np.sqrt(2 * np.pi))) * np.exp(-np.square(x - Centre) / (2 * Width**2))

# Unit conversion, kB from J/mol to 1/(cm mol)
kcm = constants.physical_constants["Boltzmann constant in inverse meters per kelvin"][0] / 100.0

def BoltzmannFunction(x, Amplitude, Temperature):
	#return Amplitude * np.exp(1) / (kcm * Temperature) * x * np.exp(- x / (kcm * Temperature))
	return Amplitude * np.sqrt(1 / (2 * np.pi * kcm * Temperature)**3) * 4 * np.pi * x * np.exp(-(x) / (kcm * Temperature))

def Linear(x, Gradient, Offset):
	return x * Gradient + Offset

def ConvolveArrays(A, B, X=None):
	""" Special function that will return the convolution of two arrays 
	in the same length, rather than 2N - 1

	Requires input of two same-dimension arrays A and B, and an optional
	1-D array that holds the X dimensions.
	"""
	BinSize = len(A)
	ConvolutionResult = signal.fftconvolve(A, B, mode="full")
	ReshapedConvolution = np.resize(ConvolutionResult, BinSize)
	return ReshapedConvolution                    # Return same length as input X

###################################################################################################

""" File I/O and plotting functions """

def UnpackDict(**args):
	""" I don't know if there's a better way of doing this,
	but I wrote this to upack a dictionary so we can parse 
	a class dictionary and unpack it into curve_fit
	"""
	print args

def LoadSpectrum(File):
	""" Function for generating a pandas dataframe from a csv,
	involves smart sniffing of delimiter, and returns a dataframe where
	the index is the X Range of the spectrum.

	Assumes two-column data.
	"""
	Delimiter = NT.DetectDelimiter(File)                          # detect what delimiter
	DataFrame = pd.read_csv(File, delimiter=Delimiter,
					 header=None, names=["Y Range"],
					 index_col=0)     							  # read file from csv
	DataFrame = DataFrame.dropna(axis=0)                          # removes all NaN values
	return DataFrame

def PlotData(DataFrame, Labels=None, Interface="pyplot"):
	""" A themed data plotting routine. Will use either matplotlib or
	bokeh to plot everything in an input dataframe, where the index is
	the X axis.
	"""
	NCols = len(DataFrame.columns)                            # Get the number of columns
	if NCols <= 2:
		Colours = ["blue", "red"]
	else:
		Colours = brewer["Spectral"][NCols]                   # Set colours depending on how many
	Headers = list(DataFrame.columns.values)                  # Get the column heads
	if Interface == "pyplot":                                 # Use matplotlib library
		plt.figure(figsize=(10,5))
		if Labels != None:
			try:                                              # Unpack whatever we can from Labels
				plt.xlabel(Labels["X Label"], fontsize=14.)
				plt.ylabel(Labels["Y Label"], fontsize=14.)
				plt.title(Labels["Title"])
				plt.xlim(Labels["X Limits"])
				plt.ylim(Labels["Y Limits"])
			except KeyError:                    # Will ignore whatever isn't given in dictionary
				pass
		for Data in enumerate(DataFrame):
			plt.plot(DataFrame.index, DataFrame[Data[1]],           # Plots with direct reference
			         color=Colours[Data[0]], label=Headers[Data[0]])     # Aesthetics with index
		plt.legend(ncol=2, loc=9)
		plt.show()
	elif Interface == "bokeh":                                # Use bokeh library
		if Labels != None:
			try:                                              # Unpack whatever we can from Labels
				XLabel = Labels["X Label"]
				YLabel = Labels["Y Label"]
				Title = Labels["Title"]
				XRange = Labels["X Limits"]
				YRange = Labels["Y Limits"]
			except KeyError:
				pass
			plot = figure(width=500, height=400,                        # set up the labels
				          x_axis_label=XLabel, y_axis_label=YLabel,
				          title=Title,
				          x_range=XRange, y_range=YRange)
		else:
			plot = figure(width=500, height=400)                        # if we have no labels
		for Data in enumerate(DataFrame):
			plot.line(x=DataFrame.index, y=DataFrame[Data[1]],
				      line_width=2, color=Colours[Data[0]],
				      legend=Headers[Data[0]])
		show(plot)
