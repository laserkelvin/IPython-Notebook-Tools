#!/bin/python

import numpy as np
import pandas as pd
import NotebookTools as NT

# SpectralAnalysis.py

# Module containing functions for day-to-day spectral analysis, including re-packaged
# routines from FittingRoutines.py
# My plan is to make FittingRoutines obsolete, as it has A LOT of poorly written
# routines.

class Spectrum:
	def __init__(self, File):
		self.RawData = LoadSpectrum(File)

def LoadSpectrum(File):
	""" Function for generating a pandas dataframe from a csv,
	involves smart sniffing of delimiter, and returns a dataframe where
	the index is the X Range of the spectrum.

	Assumes two-column data
	"""
	Delimiter = NT.DetectDelimiter(File)                         # detect what delimiter
	Original = pd.read_csv(File, delimiter=delimiter, header=None)     # read file from csv
	Original = df.dropna(axis=0)                                       # removes all NaN values
	Packaged = pd.DataFrame(data=Original[1],
		                    index=Original[0],
		                    header=["Y Range"])
	return Packaged