#!/bin/python

from collections import OrderedDict
import inspect
import numpy as np
import pandas as pd
import NotebookTools as NT
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from scipy import constants
import peakutils
from scipy import fftpack
from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit
from bokeh.palettes import brewer
from bokeh.plotting import figure, show

# SpectralAnalysis.py

# Module containing functions for day-to-day spectral analysis, including re-packaged
# routines from FittingRoutines.py
# My plan is to make FittingRoutines obsolete, as it has A LOT of poorly written
# routines.

###################################################################################################

""" Constants and small persistent dictionaries """

""" Values for oxygen calibration as a dictionary
OxygenKER is the total translational energy release for the molecule,
while OxygenAtomSpeed is self explanatory.
 """
OxygenKER = {"3P": 0.21585*2,
             "5P": 0.33985*2,
             "3S": 0.95*2,
             "5S": 1.14*2}

OxygenAtomSpeed = {"3P": 1141.016103985279,
                   "5P": 1431.7242621001822,
                   "3S": 2393.7431587974975,
                   "5S": 2622.2142498941203}

# Unit conversion, kB from J/mol to 1/(cm mol)
kcm = constants.physical_constants["Boltzmann constant in inverse meters per kelvin"][0] / 100.0

###################################################################################################

""" Classes """

class Spectrum:
	instances = []
	def __init__(self, File=None, CalConstant=1., Reference=None):
		""" If nothing is supplied, we'll just initialise a new 
		instance of Spectrum without giving it any data.
		That way I don't HAVE to load a file.
		"""
		self.CalibrationConstant = CalConstant
		self.CalibratedWavelengths = [0., 0.,]
		self.PumpWavelength = 0.
		self.ProbeWavelength = 0.
		if File != None:
			self.Data = LoadSpectrum(File, CalConstant)
		if Reference != None:                       # If we give it a logbook reference
			self.Reference = Reference
		Spectrum.instances.append(self)
	def CalibrateWavelengths(self, Wavelengths):
		""" Wavelengths given in as 2-tuple list
		and sets the Dataframe index to the calibrated
		wavelengths
		"""
		NData = len(self.Data.index)
		NewAxis = np.linspace(num=NData, *Wavelengths)
		self.Data.index = NewAxis
	def AddData(self, NewData, Name):
		self.Data[Name] = NewData
	def DeleteData(self, Name):
		del self.Data[Name]
	def ReadBogScan(self, File):
		""" Special function for reading data files from
		BogScan
		"""
		DataFrame = pd.read_csv(File, delimiter="\t", header=None)
		if DataFrame[1].sum > 0. == True:                   # If we've got calibrated wavelengths
			X = DataFrame.as_matrix([1])
			print " Using calibrated wavelengths"
		else:
			X = DataFrame.as_matrix([0])
			print " Using bogscan wavelengths"
		Y = -DataFrame.as_matrix([2])
		NewDataFrame = FormatData(X, Y)
		self.Data = NewDataFrame
	def PlotAll(self, Labels=None, Interface="pyplot"):
		self.PlotLabels(Labels)                     # Initialise labels
		try:
			self.Labels["Title"] = self.Reference
		except AttributeError:                            # Give up if we never gave it a logbook
			pass
		PlotData(self.Data, self.Labels, Interface)
	def PlotLabels(self, Labels=None):
		if Labels == None:                          # use default labels
			Labels = {"X Label": "X Axis",
					  "Y Label": "Y Axis",
					  "Title": " ",
					  "X Limits": [min(self.Data.index), max(self.Data.index)],
					  "Y Limits": [min(self.Data["Y Range"]), max(self.Data["Y Range"])],
					 }
		self.Labels = Labels
	def ExportData(self, CustomSuffix=False, Suffix=None):
		if CustomSuffix == False:
			try:
				FilePath = self.Reference + "_export.csv"
			except AttributeError:
				FilePath = raw_input(" No reference found, please specify file.")
		elif CustomSuffix == True and Suffix == None:
			try:
				Suffix = raw_input("Please enter a suffix to be used for export, e.g. _export")
				FilePath = self.Reference + Suffix
			except AttributeError:
				Reference = raw_input("No reference found, please provide one.")
				FilePath = Reference + Suffix
		elif Suffix != None:
			try:
				FilePath = self.Reference + Suffix
			except AttributeError:
				Reference = raw_input("No reference found, please provide one.")
				FilePath = Reference + Suffix
		self.Data.to_csv(FilePath, header=False)
		print " File saved to:\t" + FilePath
	def Fit(self, Model, Interface="pyplot"):
		""" Calls the FitModel function to fit the Data contained in this
		instance.

		Requires Model instance reference as input, and will only fit the
		column labelled "Y Range" in data (i.e. the initially loaded data)

		Sets attributes of instance corresponding to the optimised parameters,
		the fit report, the fitted curves dataframe and covariance matrix
		"""
		try:
			FittingData = FormatData(self.Data.index, self.Data["Y Range"])
			self.Opt, self.Report, self.FitResults, self.Cov = FitModel(FittingData, Model)
			PlotData(self.FitResults, Labels=self.Labels, Interface=Interface)
		except KeyError:
			print ''' No data column labelled "Y Range" '''
			print ''' You may need to repack the data manually using FormatData. '''
	def DetectPeaks(self, Threshold=0.3, MinimumDistance=30.):
		""" Calls the peak finding function from peakutils that will
		sniff up peaks in a spectrum. This class method will then
		store that information as an attribute
		"""
		PeakIndices = PeakFinding(self.Data, Threshold, MinimumDistance)
		self.Peaks = {"Peak Indices": PeakIndices,
		              "Threshold": Threshold,
		              "Minimum Distance": MinimumDistance}

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
		self.Variables = OrderedDict.fromkeys(inspect.getargspec(ObjectiveFunction)[0])
		try:
			del self.Variables["x"]                  # X keeps getting picked up, get rid of it
		except KeyError:
			pass
		self.BoundaryConditions = ([-np.inf for Variable in self.Variables],
			                       [np.inf for Varaible in self.Variables])
		print " Initialised variable dictionary:"
		print self.Variables
	def SetVariables(self, Variables):
		self.Variables = UpdateDictionary(self.Variables, Variables)
		print " Variables set to:"
		print str(self.Variables)
	def SetBounds(self, Bounds):
		""" Method for setting the boundary conditions for curve_fit,
		requires input as 2-tuple list ([], [])
		"""
		self.BoundaryConditions = Bounds
		print " Boundary conditions set to:" 
		print str(self.BoundaryConditions)
	def ResetAttributes(self):
		""" Wipes the variable and boundary conditions
		"""
		self.SetFunction(self.Function)

###################################################################################################

""" Data formatting and comprehension """

def ConvertSpeedDistribution(DataFrame, Mass, Units="cm"):
	""" Function to convert a data frame of speed distribution 
	into a kinetic energy data frame. Requires the mass in amu 
	as input, and returns the kinetic energy dataframe with the
	index as the KER, and "Y Range" as P(E).
	"""
	if Units == "cm":        # 1/cm
		Conversion = 83.59
	if Units == "kJ":        # kJ/mol
		Conversion = 1.
	if Units == "eV":        # eV
		Conversion = .0103636
	KER = (np.square(DataFrame.index) * (Mass / 1000.) / 2000.) * Conversion
	PE = DataFrame["Y Range"].values / (Mass / 1000. * DataFrame.index)
	KERDataFrame = pd.DataFrame(data=PE, index=KER, columns=["Y Range"])
	KERDataFrame = KERDataFrame.dropna(axis=0)
	#for index, value in enumerate(KERDataFrame)
	return KERDataFrame

def NormaliseColumn(DataFrame, Column="Y Range"):
	""" Routine to normalise a column in pandas dataframes
	"""
	DataFrame[Column] = DataFrame[Column] / np.max(DataFrame[Column])

def Dict2List(Dictionary):
	List = [Dictionary[Item] for Item in Dictionary]
	return List

def UnpackDict(**args):
	""" I don't know if there's a better way of doing this,
	but I wrote this to upack a dictionary so we can parse 
	a class dictionary and unpack it into curve_fit
	"""
	print args	

def FormatData(X, Y):
	""" Function to format data into a pandas data frame for
	fitting. In case I'm too lazy to set it up myself.
	"""
	return pd.DataFrame(data=Y, columns=["Y Range"], index=X)

def SubtractSpectra(A, B):
	""" Takes input as two instances of Spectrum class, and does
	a nifty subtraction of the spectra A - B by interpolating
	B into the X axis range of A
	"""
	XA = A.Data.index
	YA = A.Data.as_matrix(["Y Range"])
	XB = B.Data.index
	YB = B.Data.as_matrix(["Y Range"])
	Interpolation = interpolate.interp1d(XB, YB)
	RecastYB = Interpolation(XA)
	Subtraction = YA - RecastYB
	return FormatData(XA, Subtraction)

def UpdateDictionary(OldDictionary, NewValues):
	""" Will loop over keys in new dictionary and set
	them to the old dictionary
	"""
	for Key in NewValues:
		OldDictionary[Key] = NewValues[Key]
	return OldDictionary

def ConvertOrderedDict(Dictionary):
	Keys = [Key for Key in Dictionary]
	Items = [Dictionary[Key] for Key in Dictionary]
	return OrderedDict(zip(Keys, Items))

###################################################################################################

""" Fitting functions """

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
	if type(Model.BoundaryConditions) == "dict":
		Bounds = (Dict2List(Model.BoundaryConditions[0]), Dict2List(Model.BoundaryConditions[1]))
	else:
		Bounds = Model.BoundaryConditions
	print " Boundary Conditions:"
	print str(Bounds)
	print " Initial parameters:"
	OptimisedParameters, CovarianceMatrix = curve_fit(Model.Function,
												      DataFrame.index,
													  DataFrame["Y Range"], 
													  UnpackDict(**Model.Variables),
													  bounds=Bounds,
													  method="trf")
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

def PeakFinding(DataFrame, Threshold=0.3, MinimumDistance=30.):
	""" Routine that will sniff out peaks in a spectrum, and fit them with Gaussian functions
	I have no idea how well this will work, but it feels like it's gonna get pretty
	complicated pretty quickly
	"""
	PeakIndices = peakutils.indexes(DataFrame["Y Range"], thres=Threshold, min_dist=MinimumDistance)
	NPeaks = len(PeakIndices)
	print " Found \t" + str(NPeaks) + "\t peaks."
	StickSpectrum = np.zeros((len(DataFrame["Y Range"])), dtype=float)
	for Index in PeakIndices:
		StickSpectrum[Index] = 1.
	DataFrame["Stick Spectrum"] = StickSpectrum
	return PeakIndices

###################################################################################################

""" Commonly used base functions """

def BaseGaussian(x, x0):
	""" The most vanilla of Gaussians, wrote it when I was debugging
	"""
	return np.exp(-np.square(x-x0))

def GaussianFunction(x, Amplitude, Centre, Width):
	return Amplitude * (1 / (Width * np.sqrt(2 * np.pi))) * np.exp(-np.square(x - Centre) / (2 * Width**2))

def BoltzmannFunction(x, Amplitude, Temperature):
	#return Amplitude * np.exp(1) / (kcm * Temperature) * x * np.exp(- x / (kcm * Temperature))
	return Amplitude * np.sqrt(1 / (2 * np.pi * kcm * Temperature)**3) * 4 * np.pi * x * np.exp(-(x) / (kcm * Temperature))

def Linear(x, Gradient, Offset):
	return x * Gradient + Offset

def ConvolveArrays(A, B, method="new"):
	""" Function I wrote to compute the convolution of two arrays.
	The new method is written as a manual convolution calculated by
	taking the inverse Fourier transform of the Fourier product of the
	two arrays.

	Returns only the real values of the transform.

	Old function uses the scipy function, which I had issues with the
	length of array returned.

	Requires input of two same-dimension arrays A and B, and an optional
	1-D array that holds the X dimensions.
	"""
	if method == "old":
		BinSize = len(A)
		ConvolutionResult = signal.fftconvolve(A, B, mode="full")
		ReshapedConvolution = ConvolutionResult[0:BinSize]
		return ReshapedConvolution                                 # Return same length as input X
	elif method == "new":
		FTA = fftpack.fft(A)
		FTB = fftpack.fft(B)
		FTProduct = FTA * FTB
		return np.real(fftpack.ifft(FTProduct))

###################################################################################################

""" File I/O and plotting functions """

def LoadSpectrum(File, CalConstant=1.):
	""" Function for generating a pandas dataframe from a csv,
	involves smart sniffing of delimiter, and returns a dataframe where
	the index is the X Range of the spectrum.

	Assumes two-column data.
	"""
	Delimiter = NT.DetectDelimiter(File)                          # detect what delimiter
	DataFrame = pd.read_csv(File, delimiter=Delimiter,
					 header=None, names=["Y Range"],
					 index_col=0)     							  # read file from csv
	DataFrame.set_index(DataFrame.index * CalConstant, inplace=True) # Modify index
	DataFrame = DataFrame.dropna(axis=0)                          # removes all NaN values
	return DataFrame

def GenerateJComb(DataFrame, TransitionEnergies, Offset=1.3, Teeth=0.2, SelectJ=10):
	""" Function for generating a rotational comb spectrum as an annotation
	Not elegant, but it works!
	The required input:
	DataFrame - pandas dataframe containing X axis of target spectrum
	TransitionEnergies - A list of transition energies in some (e.g. J) order
	Offset - The height of the comb 
	Teeth - The length of the comb's teeth (lol)
	SelectJ - Integer for selecting out multiples of J so it's not insane

	Sets the input dataframe key "Comb" with the comb spectrum
	"""
	# Find the indices closest to where we can put our comb teeth on
	#Indices = [NT.find_nearest(DataFrame.index, Energy) for Energy in TransitionEnergies if TransitionEnergies[Energy] % SelectJ == 0]
	Indices = []
	for index, Energy in enumerate(TransitionEnergies):
		if index % SelectJ == 0:
			Indices.append(NT.find_nearest(DataFrame.index, Energy))
	Comb = [Offset for value in DataFrame.index]
	print " Adding\t" + str(len(Indices)) + "\t teeth to the comb."
	for index in Indices:
		Comb[index] = Comb[index] - Teeth
	DataFrame["Comb"] = Comb

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
		for index, Data in enumerate(DataFrame):
			plt.scatter(DataFrame.index, DataFrame[Data],           # Plots with direct reference
			         color=Colours[index])     # Aesthetics with index
			plt.plot(DataFrame.index, DataFrame[Data],           # Plots with direct reference
				     antialiased=True,
			         color=Colours[index], label=Headers[index])     # Aesthetics with index
		plt.legend(ncol=2, loc=9)
		plt.show()
	elif Interface == "bokeh":                                # Use bokeh library
		tools = "pan, wheel_zoom, box_zoom, reset, resize, hover"
		if Labels != None:
			try:                                              # Unpack whatever we can from Labels
				XLabel = Labels["X Label"]
				YLabel = Labels["Y Label"]
				Title = Labels["Title"]
				XRange = Labels["X Limits"]
				YRange = Labels["Y Limits"]
				plot = figure(width=700, height=400,                        # set up the labels
				          	  x_axis_label=XLabel, y_axis_label=YLabel,
				          	  title=Title, tools=tools,
				          	  x_range=XRange, y_range=YRange)
			except KeyError:
				print " Not using labels"
				pass
		else:
			plot = figure(width=700, height=400, tools=tools)               # if we have no labels
		plot.background_fill_color="gray"
		for index, Data in enumerate(DataFrame):
			plot.scatter(x=DataFrame.index, y=DataFrame[Data],
				      line_width=2, color=Colours[index],
				      legend=Headers[index])
			plot.line(x=DataFrame.index, y=DataFrame[Data],
				      line_width=2, color=Colours[index],
				      legend=Headers[index])
		show(plot)
