#!/bin/python

# fragment_supp.py

# Supplementary routines for the fragment analysis python script

import numpy as np         # for everything
import scipy.constants as spc
import os
import scipy.interpolate
import scipy.ndimage
from pylab import *        # for plotting

def ReadFromFile(file):       # function for reading files
	f = open(file,'r')
	fc = f.readlines()
	f.close()
	nlines = len(fc)          # get the number of lines 
	xvector = np.zeros((nlines),dtype=float)    # initialize arrays
	yvector = np.zeros((nlines),dtype=float)    # for holding two column data
	for line in range(len(fc)):
		currline = fc[line].split()
		xvector[line] = currline[0]
		yvector[line] = currline[1]
	return nlines, xvector, yvector

def BuildDistribution(vectora,vectorb):     # Combine two 1D arrays to make one 2D array
	combined = np.transpose(np.array([vectora,vectorb]))   # Transpose to get into two column
	return combined

def WriteArrayToFile(savepath,array):       # Routine for saving files
	np.savetxt(savepath,array,fmt='%1.3E')

def OldFashioned(savepath,array):
	return savepath

def SimulateTriplet(wavelength):           # Routine for simulating the pure triplet distribution in speed
	speedvector = np.zeros((4000),dtype=int)
	for i in range(len(speedvector)):
		speedvector[i] = i
	#speedvector = np.transpose(speedvector)
	#psvector = np.transpose(psvector)
	photolysisenergy = 1E7 / wavelength   # convert to cm-1
#    centre = 2550
	centre = -437.79498039 + 0.0788577706908 * photolysisenergy
	# the width we extrapolate using fitted data in wavenumbers. 
	#width = -1253.06754 + 0.03879 * photolysisenergy
	# width = np.sqrt(((width / 83.59) * 1000.0 * 2) / (15.0/1000.0)) * 0.5   # convert to speed width
	# width in speed
	width = -2020.39361 + 0.06934 * photolysisenergy
	sigma = width / 2.0
	gauss, bins = CalculateGaussian(speedvector,centre,sigma)
	return gauss, bins

def MatchMomentum(massa,massb,speedvector):     # Momentum matches the co-fragment
	dimension = len(speedvector)
	cofrag_speed = np.zeros((dimension),dtype=float)
	for i in range(dimension):      # mv = mv lol
		cofrag_speed[i] = (massa * speedvector[i]) / massb
	return cofrag_speed

def Speed2Energy(speedvector,mass):      # routine for converting speed to energy
	dimension = len(speedvector)
	energyvector = np.zeros((dimension),dtype=float)   # initialize array 
	for i in range(dimension):
		energyvector[i] = (speedvector[i]**2) * mass * (1/2000.0)   
	return energyvector

def Energy2Speed(energyvector,mass):    # Routine for converting energy to speed
# v = sqrt(T * 2m)
	dimension = len(energyvector)
	speedvector = np.zeros((dimension),dtype=float)
	for i in range(dimension):
		speedvector[i] = np.sqrt((2.0 * energyvector[i]) / mass * 1000.0)
	return speedvector

def PS2PE(speedvector,mass,psvector):    # The Jacobian to go from P(s) to P(E)
# P(E) = P(s) / mv
	dimension = len(speedvector)
	pevector = np.zeros((dimension),dtype=float)
	for i in range(1,dimension):           # P(E) = P(s)/mv   escape first number because it's zero
		pevector[i] = psvector[i] / (mass * speedvector[i])
	pevector = NormaliseVector(pevector)
	return pevector

def PE2PS(pevector,speedvector,mass):     # Routine for converting P(E) to P(s)
# P(s) = P(E) * mv
	dimension = len(pevector)
	psvector = np.zeros((dimension),dtype=float)
	for i in range(dimension):
		psvector[i] = pevector[i] * mass * speedvector[i]
	psvector = NormaliseVector(psvector)
	return psvector

def CalculateCOMSpeed(vectora,vectorb,massa,massb):      # Calculate the centre-of-mass speed
# What the shit is vectora/b?
	dimension = len(vectora)
	comspeed = np.zeros((dimension),dtype=float)
	fullmass = massa + massb
	for i in range(dimension):
		comspeed[i] = (massa * vectora[i] + massb * vectorb[i]) / fullmass
	return comspeed

def oldCalculateCOM(energyvector,fraction):   # calculate centre-of-mass kinetic energy distribution
	dimension = len(energyvector)
	comenergyvector = np.zeros((dimension),dtype=float)
	for i in range(dimension):
		comenergyvector[i] = energyvector[i] / fraction
	return comenergyvector

def CalculateCOMKineticEnergy(comspeedvector,massa,massb):
	dimension = len(comspeedvector)
	comenergyvector = np.zeros((dimension),dtype=float)
	fullmass = massa + massb
	for i in range(dimension):
		comenergyvector[i] = (comspeedvector[i]**2) * fullmass * (1/2000.0)
	return comenergyvector

def CalculateInternalEnergy(eavail,comenergyvector):    # Calculate the internal energy of a fragment
	dimension = len(comenergyvector)
	frag_internal = np.zeros((dimension),dtype=float)
	for i in range(dimension):
		frag_internal[i] = eavail - comenergyvector[i]
	return frag_internal

def ThresholdVector(threshold,vector):          # Returns the index of the vector that
	dimension = len(vector)                     # exceeds a given threshold
	FoundIt = False
	i = 0
	while FoundIt == False:
		FoundIt = vector[i] < threshold
		if i >= len(vector):
			print ' <ThresholdVector> No value found.'
			break
		else:
			i=i+1
	return i

def NormaliseVector(vector,type='max'):        # Routine for normalising a vector
	dimension = len(vector)
	normvector = np.zeros((dimension),dtype=float)
	if type == 'max':
		normfactor = np.max(vector)
	elif type == 'min':
		normfactor = np.min(vector)
	for i in range(dimension):
		normvector[i] = vector[i] / normfactor
	return normvector

def SplitVector(vector,index):                 # Routine for splitting a vector from an index
	dimension = len(vector)
	distance = dimension - index
	vectorone = np.zeros((index),dtype=float)   # vector for values up to the index
	vectortwo = np.zeros((distance),dtype=float)  # vector for the remaining values
	for i in range(0,index):
		vectorone[i] = vector[i]                    # populate the first half of the vector
	for i in range(0,distance):
		vectortwo[i] = vector[i + index]            # work where the split starts from
	return vectorone, vectortwo

def EnergyFraction(energyvector,massfraction):       # Routine for calculating how much energy each fragment takes
	dimension = len(energyvector)
	fraction_vector = np.zeros((dimension),dtype=float)
	for i in range(dimension):
		fraction_vector[i] = energyvector[i] * massfraction
	return fraction_vector

def CalculateGaussian(vector,centre,width=1):    # Generate a gaussian for the input bin
	dimension = len(vector)
	gaussian = np.zeros((dimension),dtype=float)
	twosigma = 2 * width**2
	parta = 1 / np.sqrt(twosigma * spc.pi)      # first half of the gaussian expression
	for i in range(len(vector)):
		gaussian[i] = parta * np.exp(-((vector[i] - centre)**2)/(twosigma))
	gaussian = NormaliseVector(gaussian)
	return gaussian, vector

def CalculateEavail(internalvector,barrier,excess):
	dimension = len(internalvector)
	eavailvector = np.zeros((dimension),dtype=float)
	for i in range(dimension):                   # take energy from surmounting barrier, add excess
		eavailvector[i] = (internalvector[i] - barrier) + excess
	return eavailvector

def ThresholdVectorNew(x,y,threshold):
    # Will threshold the x values i.e. looking for above a certain energy
    thresx = []
    thresy = []
    for xi in x:
        if xi >= threshold:
            ind = x.index(xi)
            thresx.append(x[ind])
            thresy.append(y[ind])
        else:
            pass
    return thresx, thresy

def CalculateVJ(v,j,B,D,we,wexe):                    # for a diatomic
    EJ = B * J * (J + 1)# - D**2 * (J + 1)**2         # rotational part
    EV = (v + 0.5) * we - (v + 0.5)**2 * wexe        # vibrational part
    return EJ + EV

def BoltzmannWeight(E,J,T):                          # Calculates the Boltzmann probability while
                                                     # including rotational degeneracy
    kb = 0.69503476       # Boltzmann in 1/cm
    return (2 * J + 1) * np.exp(-(E/(kb * T)))

def GenerateVJ():                                    # Generate a random v and J
    vmax = 2
    vmin = 0
    jmax = 50
    jmin = 0
    J = jmax - np.random.rand() * (jmax - jmin)
    v = vmax - np.random.rand() * (vmax - vmin)
    return v,J

def PlotDistribution(x,y):
	plot(x,y)
	show()

def Plot2D(xvector,yvector):
	dimension = np.shape(xvector)[0]
	for i in range(dimension):
		plot(xvector[i],yvector[i])
	show()

def congrid(a, newdims, method='linear', centre=False, minusone=False):
	# Routine I got from the SciPy cookbook for rebinning a given histogram
	# into another shape
	# There are three methods: neighbour, which will take the closest value 
	# from the original data; nearest and linear which will do a 1-D interpolation
	# using SciPy's interpolation routine; and spline which will use a routine from
	# the ndimage package of SciPy.
	# Centre is a boolean, if true makes the interpolation points at the centres of the bins
	# and if false they will be at the front edge of the bin
	# minusone is a fudge to make sure we don't extrapolate beyond our array
	if not a.dtype in [np.float64, np.float32]:
		a = np.cast[float](a)

	m1 = np.cast[int](minusone)
	ofs = np.cast[int](centre) * 0.5
	old = np.array(a.shape)
	ndims = len(a.shape)
	if len(newdims) != ndims:
		print '''[congrid] dimensions error.\n Must rebin to the same dimensions.'''
		return None
	newdims = np.asarray(newdims,dtype=float)
	dimlist = []

	if method == 'neighbour':
		for i in range(ndims):
			base = np.indices(newdims)[i]
			dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
			cd = np.array(dimlist).round().astype(int)
			newa = a[list(cd)]
		return newa

	elif method == ['nearest','linear']:
		# calculate new dimensions
		for i in range(ndims):
			base = np.arange((newdims[i]))
			dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
		# specify old dim
		olddims = [np.arange(i,dtype=np.float) for i in list(a.shape)]
		mint = scipy.interpolate.interp1d(olddims[-1],a,kind=method)
		newa = mint(dimlist[-1])

		trorder = [ndims - 1] + range(ndims - 1)
		for i in range(ndims - 2, -1, -1):
			newa = newa.transpose(trorder)
			mint = scipy.interpolate.interp1d(olddims[i],newa,kind=method)
			newa = mint(dimlist[i])

		if ndims > 1:
			newa = newa.transpose(trorder)
		return newa