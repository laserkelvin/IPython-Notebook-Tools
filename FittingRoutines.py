#!/bin/python

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, hplot
from bokeh.palettes import Spectral9, brewer
from scipy import constants
from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit, fmin        # For fitting the gaussians
import NotebookTools as NT

# unit conversion
kcm = constants.physical_constants["Boltzmann constant in inverse meters per kelvin"][0] / 100.0

######################### Fitting functions #########################

# x is self-explanatory, a is the amplitude, x0 is the mean and sigma is the width
def gauss_function(x, a, x0, sigma):
    return a * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2))

# x is self-explanatory, a is the amplitude, T is the temperature and m is the mass
# Used in the convolution version of the fitting.
def boltzmann_function(a, x, T):
    return a * np.exp(1) / (kcm * T) * x * np.exp(- x / (kcm * T))
#    return a * np.sqrt(1 / (2 * np.pi * kcm * T)**3) * 4 * np.pi * x * np.exp(-x / (kcm * T))

# A single boltzmann with variable maxima. Used in the Gaussian+Boltzmann version.
def independent_boltzmann(a, x, T):
    return a * np.sqrt(1 / (2 * np.pi * kcm * T)**3) * 4 * np.pi * x * np.exp(-(x) / (kcm * T))

# Pretty self-explanatory...
def straightline(x, m, c):
    return m * (x) + c

######################### Combination functions #########################

# Convolution of a Boltzmann and a Gaussian distribution. The mode="same" puts the length of
# the convolution result vector the same as the input. This function is called by the fitting
# routine.
def GaussMann(x, A, aG, aB, x0, sigma, T):
    Gaussian = gauss_function(x, aG, x0, sigma)
    Boltzmann = boltzmann_function(aB, x, T)
    Convolution = signal.fftconvolve(Gaussian, Boltzmann)
    return A * Convolution

def NewGaussMann(x, A, aG, aB, x0, sigma, T):
    """ This function will return the convolution of a Gaussian and Boltzmann
    in the original bin sizes without shifting.
    """
    BinSize = len(x) * 2 - 1                                         # Predict how large the convolution array wil be
    ConvolveX = np.linspace(x.iloc[0], x.iloc[-1], BinSize)          # Generate a new X vector for the convolution
    Gaussian = gauss_function(x, aG, x0, sigma)
    Boltzmann = boltzmann_function(aB, x, T)
    Convolution = signal.fftconvolve(Gaussian, Boltzmann)
    Reshape = interpolate.interp1d(ConvolveX, Convolution, kind="nearest")
    ReshapedConvolution = Reshape(x)
    return A * ReshapedConvolution

def NewDoubleGaussMann(x, A1, A2, x01, sigma1, x02, sigma2, T1, T2):
    GBOne = NewGaussMann(x, A1, 1., 1., x01, sigma1, T1)
    GBTwo = NewGaussMann(x, A2, 1., 1., x02, sigma2, T2)
    Combined = GBOne + GBTwo
    return Combined

def HotGaussian(x, aG1, aG2, x0, sigma):
    return gauss_function(x, aG1, 2600., 600) + gauss_function(x, aG2, x0, sigma)

# Convolution of a gaussian with the convolution of a gaussian and a boltzmann. This should ideally be
# assuming an impulsive 3F and a statistical/impulsive T1
def DoubleGaussMann(x, A, aG1, aG2, aB, x01, x02, sigma1, sigma2, T):
    return (gauss_function(x, aG1, x01, sigma1) + A * signal.fftconvolve(gauss_function(x, aG2, x02, sigma2),
                                                                        boltzmann_function(aB, x, T), mode="same"))

def TrueDoubleGaussMann(x, A, aG1, aG2, aB1, aB2, x01, x02, sigma1, sigma2, T1, T2):
    return A * (signal.fftconvolve(gauss_function(x, aG1, x01, sigma1),boltzmann_function(aB1, x, T1), mode="same") + 
                signal.fftconvolve(gauss_function(x, aG2, x02, sigma2),boltzmann_function(aB2, x, T2), mode="same"))
                

# The sum of gaussian and boltzmann
def SumGB(x, aG, aB, x0, sigma, T):
    return gauss_function(x, aG, x0, sigma) + independent_boltzmann(aB, x, T)

def DoubleGaussian(x, aG1, aG2, x01, x02, sigma1, sigma2):
    return gauss_function(x, aG1, x01, sigma1) + gauss_function(x, aG2, x02, sigma2)

######################### Fitting and printout routines #########################

def NewConvolveGaussianBoltzmann(DataFrame, Parameters, Bounds=(-np.inf, np.inf), Plotting="pyplot"):
    print "Initial parameters:\t" + str(Parameters)
    OptimisedParameters, CovarianceMatrix = curve_fit(NewGaussMann,
                                                      DataFrame["X Range"],
                                                      DataFrame["Experiment"],
                                                      Parameters,
                                                      bounds=Bounds)
    FitSummary = pd.DataFrame(data = OptimisedParameters,
                              index = ["Amplitude",
                                       "Gaussian Amplitude",
                                       "Boltzmann Amplitude",
                                       "Gaussian Centre",
                                       "Gaussian Width",
                                       "Boltzmann Temperature"],)
    print FitSummary
    ConvolutionResult = GaussMann(DataFrame["X Range"], *OptimisedParameters)
    FitResults = pd.DataFrame(data = zip(DataFrame["Experiment"],
                                         ConvolutionResult),
                              columns = ["Experiment",
                                         "Convolution"],
                              index = DataFrame["X Range"])
    if Plotting == "pyplot":
        plt.plot(FitResults.index, FitResults["Experiment"], "o")
        plt.plot(FitResults.index, FitResults["Convolution"], "-")
        plt.show()
    elif Plotting == "bokeh":
        p = figure(width=600, height=300, x_axis_label="CH3 translational energy",
                   x_range=[0,10000])
        p.background_fill_color = "beige"
        p.circle(FitResults.index, FitResults["Experiment"], legend = "Experiment",
                 color=brewer["Spectral"][5][0], radius=60, fill_alpha=0.6,)
        p.line(FitResults.index, FitResults["Convolution"], legend = "Convolution",
               color=brewer["Spectral"][5][4], line_width=2)
        show(p)
    return FitResults, FitSummary

def NewDoubleConvolveGaussianBoltzmann(DataFrame, Parameters, Bounds=(-np.inf, np.inf), Plotting="pyplot"):
    print "Initial parameters:\t" + str(Parameters)
    OptimisedParameters, CovarianceMatrix = curve_fit(NewDoubleGaussMann,
                                                      DataFrame["X Range"],
                                                      DataFrame["Experiment"],
                                                      Parameters,
                                                      bounds=Bounds)
    FitSummary = pd.DataFrame(data = OptimisedParameters,
                              index = ["Amplitude One",
                                       "Amplitude Two",
                                       "Gaussian One Centre",
                                       "Gaussian One Width",
                                       "Gaussian Two Centre",
                                       "Gaussian Two Width",
                                       "Temperature One",
                                       "Temperature Two"])
    print FitSummary
    CombinedResult = NewDoubleGaussMann(DataFrame["X Range"], *OptimisedParameters)
    ConvolutionOne = NewGaussMann(DataFrame["X Range"],
                                 *[FitSummary[0]["Amplitude One"],
                                  1.0,
                                  1.0, 
                                  FitSummary[0]["Gaussian One Centre"],
                                  FitSummary[0]["Gaussian One Width"],
                                  FitSummary[0]["Temperature One"]])
    ConvolutionTwo = NewGaussMann(DataFrame["X Range"],
                                 *[FitSummary[0]["Amplitude Two"],
                                  1.0,
                                  1.0, 
                                  FitSummary[0]["Gaussian Two Centre"],
                                  FitSummary[0]["Gaussian Two Width"],
                                  FitSummary[0]["Temperature Two"]])
    FitResults = pd.DataFrame(data = zip(DataFrame["Experiment"], CombinedResult, ConvolutionOne, ConvolutionTwo),
                              columns = ["Experiment", "Combined", "Convolution One", "Convolution Two"],
                              index = DataFrame["X Range"])
    if Plotting == "pyplot":
        plt.plot(FitResults.index, FitResults["Experiment"], "o", color="blue")
        plt.plot(FitResults.index, FitResults["Combined"], "-", color="red")
        plt.plot(FitResults.index, FitResults["Convolution One"], "-", color="magenta")
        plt.plot(FitResults.index, FitResults["Convolution Two"], "-", color="green")
        plt.show()
    elif Plotting == "bokeh":
        p = figure(width=600, height=300, x_axis_label="CH3 translational energy",
                   x_range=[0,10000])
        p.background_fill_color = "beige"
        p.circle(FitResults.index, FitResults["Experiment"], legend = "Experiment",
                 color=brewer["Spectral"][9][0], radius=60, fill_alpha=0.6,)
        p.line(FitResults.index, FitResults["Combined"], legend = "Combined",
               color=brewer["Spectral"][9][5], line_width=2)
        p.line(FitResults.index, FitResults["Convolution One"], legend = "Convolution One",
               color=brewer["Spectral"][9][7], line_width=2, line_alpha=0.6)
        p.line(FitResults.index, FitResults["Convolution Two"], legend = "Convolution Two",
               color=brewer["Spectral"][9][8], line_width=2, line_alpha=0.6)
        show(p)

# Function to do all of the fitting in a single fell swoop. Requires data as Panda input and an initial parameters
# vector. Returns the fitted curve and prints optimal parameters. Name is only for internal reference.
def FitGaussian(Name, Data, Parameters, Error=None):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(gauss_function, Data[0], Data[1], Parameters, sigma=Error)
    Result = [gauss_function(x, *popt) for x in Data[0]]
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Result),
                                columns=["X Range", "Experiment", "Gaussian"])
    print "------------------------------------------------------"
    print "                 Vanilla Gaussian Fit                 "
    print "------------------------------------------------------"
    print "Gaussian fitting output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Amplitude:\t" + str(popt[0])
    print "Centre:\t" + str(popt[1]) + "\t 1/cm"
    print "Width:\t" + str(popt[2]) + "\t 1/cm"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tGaussian", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0],Data[1],color=Spectral9[0],legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0],FittedCurves["Gaussian"],color=Spectral9[7],legend="Fit",line_width=2)
    show(p)
    return FittedCurves, popt, pcov

def FitBoundedGaussian(Name, Data, Parameters, Bounds=(-np.inf, np.inf)):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(gauss_function, Data[0], Data[1], Parameters, bounds=Bounds, method="trf")
    Result = [gauss_function(x, *popt) for x in Data[0]]
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Result),
                                columns=["X Range", "Experiment", "Gaussian"])
    print "------------------------------------------------------"
    print "                 Bounded Gaussian Fit                 "
    print "------------------------------------------------------"
    print "Gaussian fitting output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Amplitude:\t" + str(popt[0])
    print "Centre:\t" + str(popt[1]) + "\t 1/cm"
    print "Width:\t" + str(popt[2]) + "\t 1/cm"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tGaussian", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0],Data[1],color=Spectral9[0],legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0],FittedCurves["Gaussian"],color=Spectral9[7],legend="Fit",line_width=2)
    show(p)
    return FittedCurves, popt, pcov

def FitHotGaussian(Name, Data, Parameters, Bounds=(-np.inf, np.inf), Error=None):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(HotGaussian, Data[0], Data[1], Parameters, bounds=Bounds, method="trf", sigma=Error)
    Combined = [HotGaussian(x, *popt) for x in Data[0]]
    GaussianOne = [gauss_function(x, popt[0], 2600., 600.) for x in Data[0]]
    GaussianTwo = [gauss_function(x, popt[1], popt[2], popt[3]) for x in Data[0]]
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Combined, GaussianOne, GaussianTwo),
                                columns=["X Range", "Experiment", "Combined", "Gaussian One", "Gaussian Two"])
    print "------------------------------------------------------"
    print "                  Hot Gaussian Fit                    "
    print "------------------------------------------------------"
    print "Gaussian fitting output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Hot Gaussian Amplitude:\t" + str(popt[0])
    print "Cold Gaussian Amplitude:\t" + str(popt[1])
    print "Cold Centre:\t" + str(popt[2]) + "\t 1/cm"
    print "Cold Width:\t" + str(popt[3]) + "\t 1/cm"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tHot Gaussian", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,6000])
    p.background_fill_color="beige"
    p.circle(FittedCurves["X Range"], FittedCurves["Experiment"],
             fill_alpha=0.6, radius=60, legend="Data")
    p.line(FittedCurves["X Range"], FittedCurves["Combined"],
           line_width=2, color=brewer["Spectral"][9][0], legend="Combined")
    p.line(FittedCurves["X Range"], FittedCurves["Gaussian One"],
           line_width=2, color=brewer["Spectral"][9][6], legend="Hot Gaussian")
    p.line(FittedCurves["X Range"], FittedCurves["Gaussian Two"],
           line_width=2, color=brewer["Spectral"][9][2], legend="Cold Gaussian")
    show(p)
    return FittedCurves, popt, pcov

# Functional for fitting a convolution of a gaussian and boltzmann function.
# The plotting doesn't work, but it needs to be plotted separately!!
def FitBoundedConvolveGB(Name, Data, Parameters, Bounds=(-np.inf,np.inf), Error=None):
    print "Initial parameters:\t" + str(Parameters)
    if Bounds != (-np.inf, np.inf):                           # if we're specifying boundary conditions
        popt, pcov = curve_fit(GaussMann, Data[0], Data[1], 
                               Parameters, bounds=Bounds, method="trf", sigma=Error)
    else:                                                     # no conditions specified, default back
        popt, pcov = curve_fit(GaussMann, Data[0], Data[1], Parameters, sigma=Error)
    Result = GaussMann(Data[0], *popt)
    FittedCurve = pd.DataFrame(data=zip(Data[0], Data[1], Result),
                               columns=["X Range", "Experiment", "Convolution"])
    print "------------------------------------------------------"
    print "        Bounded Convolved Gaussian Boltzmann          "
    print "------------------------------------------------------"
    print "Convolved Gaussian + Boltzmann output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Total Amplitude:\t" + str(popt[0]) + "\t \t 0"
    print "Gaussian Amplitude:\t" + str(popt[1]) + "\t \t 1"
    print "Boltzmann Amplitude:\t" + str(popt[2]) + "\t \t 2"
    print "Gaussian Centre:\t" + str(popt[3]) + "\t 1/cm \t 3"
    print "Gaussian Width: \t" + str(popt[4]) + "\t 1/cm \t 4" 
    print "Temperature:  \t" + str(popt[5]) + "\t K \t 5"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tBounded Convolve GB", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0], Data[1], color=Spectral9[0], legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0], FittedCurve["Convolution"], color=Spectral9[8], legend="Fit",line_width=2)
    show(p)
    return FittedCurve, popt

def FitDoubleGaussian(Name, Data, Parameters, Error=None):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(DoubleGaussian, Data[0], Data[1], Parameters, sigma=Error)
    Combined = [DoubleGaussian(x, *popt) for x in Data[0]]
    GaussOne = [gauss_function(x, popt[0], popt[2], popt[4]) for x in Data[0]]
    GaussTwo = [gauss_function(x, popt[1], popt[3], popt[5]) for x in Data[0]]
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Combined, GaussOne, GaussTwo),
                                columns=["X Range", "Experiment", "Combined", "Gaussian One", "Gaussian Two"])
    p = figure(title=Name + "\tGaussian", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0],Data[1],color=Spectral9[0],legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0],FittedCurves["Combined"],color=Spectral9[6],legend="Combined",line_width=2)
    p.line(Data[0],FittedCurves["Gaussian One"],color=Spectral9[7],legend="Gauss One",line_width=2)
    p.line(Data[0],FittedCurves["Gaussian Two"],color=Spectral9[8],legend="Gauss Two",line_width=2)
    show(p)
    return FittedCurves, popt, pcov

# Functional for fitting the sum of a gaussian with the convolution of a gaussian and a boltzmann
# tailored for fitting the 3F distributions. Let's hope it works!
def FitBoundedDoubleConvolve(Name, Data, Parameters, Bounds=(-np.inf,np.inf), Error=None):
    print "Initial parameters:\t" + str(Parameters)
    if Bounds != (-np.inf, np.inf):                           # if we're specifying boundary conditions
        popt, pcov = curve_fit(DoubleGaussMann, Data[0], Data[1], 
                               Parameters, bounds=Bounds, method="trf", sigma=Error)
    else:                                                     # no conditions specified, default back
        popt, pcov = curve_fit(DoubleGaussMann, Data[0], Data[1], Parameters, sigma=Error)
    Result = DoubleGaussMann(Data[0], *popt)
    Gaussian = gauss_function(Data[0], popt[1], popt[4], popt[6])
    GB = GaussMann(Data[0], popt[0], popt[2], popt[3], popt[5], popt[7], popt[8])
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Result, Gaussian, GB),
                                columns=["X Range", "Experiment", "Combined", "Gaussian", "Convolution"])
    print "------------------------------------------------------"
    print "     Bounded Double Convolved Gaussian Boltzmann      "
    print "------------------------------------------------------"
    print "Double Convolved Gaussian + Boltzmann output for reference:\t" + str(Name)
    print "Boundary Parameters:\t" + str(Bounds)
    print "------------------------------------------------------"
    print "Total Amplitude:\t" + str(popt[0]) + "\t \t 0"
    print "3F Gaussian Amplitude:\t" + str(popt[1]) + "\t \t 1"
    print "T1 Gaussian Amplitude:\t" + str(popt[2]) + "\t \t 2"
    print "T1 Boltzmann Amplitude:\t" + str(popt[3]) + "\t \t 3"
    print "3F Gaussian Centre: \t" + str(popt[4]) + "\t 1/cm \t 4" 
    print "T1 Gaussian Centre:  \t" + str(popt[5]) + "\t 1/cm \t 5"
    print "3F Gaussian Sigma:   \t" + str(popt[6]) + "\t 1/cm \t 6"
    print "T1 Gaussian Sigma:   \t" + str(popt[7]) + "\t 1/cm \t 7"
    print "T1 Boltzmann Temperature: \t" + str(popt[8]) + "\t K \t 8"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tBounded Double Convolve GB", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.background_fill_color="beige"
    p.circle(Data[0], Data[1], color=brewer["Spectral"][4][0], legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0], FittedCurves["Combined"], color=brewer["Spectral"][4][3], legend="Fit",line_width=2)
    p.line(Data[0], FittedCurves["Gaussian"], color=brewer["Spectral"][4][2], legend="3F",line_width=2)
    p.line(Data[0], FittedCurves["Convolution"], color=brewer["Spectral"][4][1], legend="T1",line_width=2)
    show(p)
    GaussFrac = np.trapz(Gaussian, Data[0])
    ConvolveFrac = np.trapz(GB, Data[0])
    print "Relative Branching Ratios:"
    print "------------------------------------------------------"
    print "3F:\t " + str(GaussFrac / (ConvolveFrac + GaussFrac))
    print "------------------------------------------------------"
    return FittedCurves, popt, pcov, GaussFrac / (ConvolveFrac + GaussFrac)

# Functional for fitting the sum of two GB convolutions for fitting the 3F data.
# The previous routine works kinda, but it seems like we need a gaussian/boltzmann for 3F maybe...
def FitTrueDoubleConvolve(Name, Data, Parameters, Bounds=(-np.inf,np.inf), Error=None):
    print "Initial parameters:\t" + str(Parameters)
    if Bounds != (-np.inf, np.inf):                           # if we're specifying boundary conditions
        popt, pcov = curve_fit(TrueDoubleGaussMann, Data["X Range"], Data["Experiment"], 
                               Parameters, bounds=Bounds, method="trf", sigma=Error)
    else:                                                     # no conditions specified, default back
        popt, pcov = curve_fit(TrueDoubleGaussMann, Data["X Range"], Data["Experiment"], Parameters, sigma=Error)
    Result = TrueDoubleGaussMann(Data["X Range"], *popt)
    GB1 = GaussMann(Data["X Range"], popt[0], popt[1], popt[3], popt[5], popt[7], popt[9])
    GB2 = GaussMann(Data["X Range"], popt[0], popt[2], popt[4], popt[6], popt[8], popt[10])
    FittedCurves = pd.DataFrame(data=zip(Data["X Range"], Data["Experiment"], Result, GB1, GB2),
                                columns=["X Range", "Experiment", "Combined", "3F", "T1"])
    print "------------------------------------------------------"
    print "     True Double Convolved Gaussian Boltzmann      "
    print "------------------------------------------------------"
    print "True Double Convolved Gaussian Boltzmann output for reference:\t" + str(Name)
    print "Boundary Parameters:\t" + str(Bounds)
    print "------------------------------------------------------"
    print "Total Amplitude:\t" + str(popt[0]) + "\t \t 0"
    print "3F Gaussian Amplitude:\t" + str(popt[1]) + "\t \t 1"
    print "T1 Gaussian Amplitude:\t" + str(popt[2]) + "\t \t 2"
    print "------------------------------------------------------"
    print "3F Boltzmann Amplitude:\t" + str(popt[3]) + "\t \t 3"
    print "T1 Boltzmann Amplitude:\t" + str(popt[4]) + "\t \t 4"
    print "------------------------------------------------------"
    print "3F Gaussian Centre: \t" + str(popt[5]) + "\t 1/cm \t 5" 
    print "T1 Gaussian Centre:  \t" + str(popt[6]) + "\t 1/cm \t 6"
    print "------------------------------------------------------"
    print "3F Gaussian Sigma:   \t" + str(popt[7]) + "\t 1/cm \t 7"
    print "T1 Gaussian Sigma:   \t" + str(popt[8]) + "\t 1/cm \t 8"
    print "------------------------------------------------------"
    print "3F Boltzmann Temperature: \t" + str(popt[9]) + "\t K \t 9"
    print "T1 Boltzmann Temperature: \t" + str(popt[10]) + "\t K \t 10"
    print "------------------------------------------------------"
    p = figure(title=Name + "\tBounded Double Convolve GB", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.background_fill_color="beige"
    p.circle(Data["X Range"], Data["Experiment"], color=brewer["Spectral"][4][0], legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data["X Range"], FittedCurves["Combined"], color=brewer["Spectral"][4][3], legend="Fit",line_width=2)
    p.line(Data["X Range"], FittedCurves["3F"], color=brewer["Spectral"][4][2], legend="3F",line_width=2)
    p.line(Data["X Range"], FittedCurves["T1"], color=brewer["Spectral"][4][1], legend="T1",line_width=2)
    show(p)
    TripleFrac = np.trapz(GB1, Data["X Range"])
    TripletFrac = np.trapz(GB2, Data["X Range"])
    print "Relative Branching Ratios:"
    print "------------------------------------------------------"
    print "3F:\t " + str(TripleFrac / TripletFrac)
    print "------------------------------------------------------"
    return FittedCurves, popt, pcov, TripleFrac / TripletFrac

# Functional for fitting a sum of gaussian and boltzmann. This is used for the S0
# distributions as the two distributions are a sum, not a convolution.
def FitSumGB(Name, Data, Parameters, Error=None):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(SumGB, Data[0], Data[1], Parameters, sigma=Error)
    Result = SumGB(Data[0], *popt)
    Boltzmann = boltzmann_function(popt[1], Data[0], popt[4])
    Gaussian = gauss_function(Data[0], popt[0], popt[2], popt[3])
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Result, Boltzmann, Gaussian),
                               columns=["X Range", "Experiment", "Combined Fit", "Boltzmann", "Gaussian"])
    print "------------------------------------------------------"
    print "          Unconstrained Gaussian + Boltzmann          "
    print "------------------------------------------------------"
    print "Sum of Gaussian + Boltzmann output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Gaussian Amplitude:\t" + str(popt[0])
    print "Boltzmann Amplitude:\t" + str(popt[1])
    print "Gaussian Centre:\t" + str(popt[2]) + "\t 1/cm"
    print "Gaussian Width:\t" + str(popt[3]) + "\t 1/cm"
    print "Temperature:\t" + str(popt[4]) + "\t K"
    print "------------------------------------------------------"
    print "Copy me:\t" + str(popt)
    p = figure(title=Name + "\tSum GB", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0], Data[1], color=Spectral9[0], legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0], FittedCurves["Combined Fit"], color=Spectral9[8], 
           legend="Fit " + str(popt[4]) + " K",line_width=2)
    show(p)
    return FittedCurves, popt

# Routine for fitting a specific range in the distribution. An optional parameter defined
# will specify the maximum X value that will be used during the fitting process.
# If not specified, the MaxX variable is fudged to whatever is the last element (largest).
def FitSpecificSumGB(Name, Data, Parameters, MaxX=20000, Bounds=(-np.inf, np.inf), Error=None):
    FinalIndex = nb.find_nearest(Data[0],MaxX)                       # Find the value that matches closest
    SplitX = Data[0][:FinalIndex]                                    # to what was specified
    SplitY = Data[1][:FinalIndex]                                    # Only use the vectors below
    print "Initial parameters:\t" + str(Parameters)
    if Bounds != [-np.inf, np.inf]:
        popt, pcov = curve_fit(SumGB, SplitX, SplitY, 
                               Parameters, bounds=Bounds, method="trf", sigma=Error) 
    else:
        popt, pcov = curve_fit(SumGB, SplitX, SplitY, Parameters, method="lm", sigma=Error)
    Result = SumGB(Data[0], *popt)
    Boltzmann = boltzmann_function(popt[1], Data[0], popt[4])
    Gaussian = gauss_function(Data[0], popt[0], popt[2], popt[3])
    FittedCurves = pd.DataFrame(data=zip(Data[0], Data[1], Result, Boltzmann, Gaussian),
                               columns=["X Range", "Experiment", "Combined Fit", "Boltzmann", "Gaussian"])
    print "------------------------------------------------------"
    print "          Constrained Gaussian + Boltzmann Fit        "
    print "------------------------------------------------------"
    print "Sum of Gaussian + Boltzmann output for reference:\t" + str(Name)
    print "------------------------------------------------------"
    print "Gaussian Amplitude:\t" + str(popt[0])
    print "Boltzmann Amplitude:\t" + str(popt[1])
    print "Gaussian Centre:\t" + str(popt[2]) + "\t 1/cm"
    print "Gaussian Width:\t" + str(popt[3]) + "\t 1/cm"
    print "Temperature:\t" + str(popt[4]) + "\t K"
    print "------------------------------------------------------"
    print "Copy me:\t" + str(popt)
    p = figure(title=Name + "\tConstrained Sum GB", width=600, height=300,
               x_axis_label="CH3 Kinetic energy", x_range=[0,10000])
    p.circle(Data[0], Data[1], color=Spectral9[0], legend="Data", radius=60, fill_alpha=0.6)
    p.line(Data[0], FittedCurves["Combined Fit"], color=Spectral9[8], legend="Fit " + str(popt[4]) + " K",line_width=2)
    show(p)
    return FittedCurves, popt

# Routine for fitting a straight line lol
def FitLinear(Data, Parameters, Error=None):
    print "Initial parameters:\t" + str(Parameters)
    popt, pcov = curve_fit(straightline, Data[0], Data[1], Parameters, sigma=Error)
    FittedCurve = [straightline(x, *popt) for x in Data[0]]
    return FittedCurve, popt, pcov

######################### Integration Functions #########################

# Routine for integrating the gaussian and convolution components
# and calculating their respective ratios
def IntegrateDGB(Data, Parameters):
    # Get the parameters for the 3F gaussian
    GaussParameters = [ Parameters[1], Parameters[4], Parameters[6] ]
    # Get the parameters for the convolution
    GBParameters = [ Parameters[0], Parameters[2], Parameters[3],
                    Parameters[5], Parameters[7], Parameters[8]]
    # Generate the curves
    Gaussian = gauss_function(Data[0], *GaussParameters)
    GB = GaussMann(Data[0], *GBParameters)
    GaussianIntegral = np.trapz(Gaussian, Data[0])
    GBIntegral = np.trapz(GB, Data[0])
    return GaussianIntegral / (GBIntegral + GaussianIntegral), GBIntegral / (GBIntegral + GaussianIntegral)

# For calculating the fraction between the "hot" and "cold" gaussians down near the T1 barrier
def IntegrateHotGaussian(Data, Parameters):
    GaussianOneParam = Parameters[0]
    GaussianTwoParam = [ Parameters[1], Parameters[2], Parameters[3] ]
    GaussianOne = gauss_function(Data["X Range"], GaussianOneParam, 2600., 600.)
    GaussianTwo = gauss_function(Data["X Range"], *GaussianTwoParam)
    GaussianOneIntegral = np.trapz(GaussianOne, Data["X Range"])
    GaussianTwoIntegral = np.trapz(GaussianTwo, Data["X Range"])
    return GaussianOneIntegral / (GaussianOneIntegral + GaussianTwoIntegral), GaussianTwoIntegral / (GaussianOneIntegral + GaussianTwoIntegral)

# Routine for integrating the sum of a gaussian and a boltzmann, as used for the S0/T1 data
def IntegrateSGB(Data, Parameters):
    # Get the parameters for the Boltzmann
    BoltzParameters = [ Parameters[1], Parameters[4]]
    # Get the parameters for the gaussian
    GaussParameters = [ Parameters[0], Parameters[2], Parameters[3]]
    # Generate the curves
    Gaussian = gauss_function(Data[0], *GaussParameters)
    Boltzmann = boltzmann_function(BoltzParameters[0], Data[0], BoltzParameters[1])
    # Integrate the respective curves
    GaussianIntegral = np.trapz(Gaussian, Data[0])
    BoltzmannIntegral = np.trapz(Boltzmann, Data[0])
    return GaussianIntegral / (GaussianIntegral + BoltzmannIntegral), BoltzmannIntegral / (GaussianIntegral + BoltzmannIntegral)

######################### Error Functions #########################

# Wrote a function that will start the Bootstrap analysis process
# This way I don't have to keep copy-pasting how to run the analysis!
def RunBootStrap(Data, OptimisedParameters, Function, Trials = 1000):
    NParameters = len(OptimisedParameters)
    if Function == "HotGaussian" or "DoubleGB" or "SumGB":        # these functions also calculate ratios
        NParameters = NParameters + 2
    ParametersBin = np.zeros((Trials, NParameters), dtype=float)  # np.ndarray for holding the fitting results
    print " Using function " + Function
    for trial in range(Trials):
        ParametersBin[trial] = BootStrapError(Data, Function, OptimisedParameters)
        if trial % 100 == 0:                                     # Just a little progress bar
            print " Done " + str(trial) + " trials."
        else:
            pass
    return ParametersBin

# Routine using Monte Carlo sampling for Bootstrap error analysis
# First we take the optimised model parameters, and generate a new set of "synthetic" data
# We then re-fit the synthetic data to gauge how well our optimised parameters
# represent parameter space
def BootStrapError(Data, Function, InParameters, Bounds=(-np.inf, np.inf)):
    if Function == "Gaussian":
        try:
            ModelY = gauss_function(Data[0], *InParameters)
            RandomNoise = np.random.rand(len(Data[0])) * (max(ModelY) * 0.1)      # add 10% as noise
            SimulatedY = gauss_function(Data[0], *InParameters) + RandomNoise
            return curve_fit(gauss_function, Data[0], SimulatedY, InParameters, bounds=Bounds)[0]
        except RuntimeError:
            pass
    if Function == "NGB":
        try:
            ModelY = NewGaussMann(Data["X Range"], *InParameters)
            RandomNoise = np.random.rand(len(Data["X Range"])) * (max(ModelY) * 0.1)
            SimulatedY = ModelY + RandomNoise
            return curve_fit(NewGaussMann, Data["X Range"], SimulatedY, InParameters, bounds=Bounds)[0]
        except RuntimeError:
            pass
    if Function == "GB":
        try:
            ModelY = GaussMann(Data[0], *InParameters)
            RandomNoise = np.random.rand(len(Data[0])) * (max(ModelY) * 0.1)      # add 10% as noise
            SimulatedY = GaussMann(Data[0], *InParameters) + RandomNoise
            return curve_fit(GaussMann, Data[0], SimulatedY, InParameters, bounds=Bounds)[0]
        except RuntimeError:
            pass
    if Function == "HotGaussian":
        try:
            ModelY = HotGaussian(Data["X Range"], *InParameters)
            RandomNoise = np.random.rand(len(Data["X Range"])) * (max(ModelY) * 0.1)      # add 10% as noise
            SimulatedY = HotGaussian(Data["X Range"], *InParameters) + RandomNoise
            NewFitOpt = curve_fit(HotGaussian, Data["X Range"], SimulatedY, InParameters, bounds=Bounds)[0]
            # Generate individual distributions for integration
            HotDistribution = gauss_function(Data["X Range"], NewFitOpt[0], 2600., 600.)
            ColdDistribution = gauss_function(Data["X Range"], NewFitOpt[1], NewFitOpt[2], NewFitOpt[3])
            # Use trapz to integrate distributions for ratio
            HotIntegral = np.trapz(HotDistribution, Data["X Range"])
            ColdIntegral = np.trapz(ColdDistribution, Data["X Range"])
            Sum = HotIntegral + ColdIntegral
            NewFitOpt = np.append(NewFitOpt, HotIntegral / Sum)
            NewFitOpt = np.append(NewFitOpt, ColdIntegral / Sum)
            return NewFitOpt
        except RuntimeError:
            pass
    if Function == "DoubleGB":
        try:
            # This case is slightly modified to get the integrals as well
            ModelY = DoubleGaussMann(Data[0], *InParameters)
            RandomNoise = np.random.rand(len(Data[0])) * (max(ModelY) * 0.1)      # add 10% as noise
            SimulatedY = DoubleGaussMann(Data[0], *InParameters) + RandomNoise
            NewFitOpt = curve_fit(DoubleGaussMann, Data[0], SimulatedY, InParameters, bounds=Bounds)[0]
            # Calculate the integrals
            GaussianIntegral, GBIntegral = IntegrateDGB(Data, NewFitOpt)
            NewFitOpt = np.append(NewFitOpt, GaussianIntegral)
            NewFitOpt = np.append(NewFitOpt, GBIntegral)
            return NewFitOpt
        except RuntimeError:
            pass
    if Function == "SumGB":
        try:
            ModelY = SumGB(Data[0], *InParameters)
            RandomNoise = np.random.rand(len(Data[0])) * (max(ModelY) * 0.1)      # add 10% as noise
            SimulatedY = SumGB(Data[0], *InParameters) + RandomNoise
            NewFitOpt = curve_fit(SumGB, Data[0], SimulatedY, InParameters, bounds=Bounds)[0]
            GaussianRatio, BoltzmannRatio = IntegralSGB(Data, NewFitOpt)
            NewFitOpt = np.append(NewFitOpt, GaussianRatio)
            NewFitOpt = np.append(NewFitOpt, BoltzmannRatio)
            return NewFitOpt
        except RuntimeError:
            pass
    
# Routine to do the histogram analysis for an array of parameters from bootstrap results
# Will also fit the histogram with a gaussian, and get the mean/sigma from this.
def BootStrapAnalysis(ParametersArray):
    Histogram = np.histogram(ParametersArray, density=True, bins=50)
    # Here we generate a guess for the parameters by finding the average and standard deviation
    # without fitting; first number is amplitude
    Parameters = [ 1, np.average(ParametersArray), np.std(ParametersArray) ]
#    print "Initial guess for average:\t" + str(np.average(ParametersArray))
#    print "Initial guess for sigma:\t" + str(np.std(ParametersArray))
    # Here the X range is hacked; the bin vector returned is actually 1 longer than the Y...
    try:
        popt, pcov = curve_fit(gauss_function, Histogram[1][:-1], Histogram[0], Parameters)
        return popt
    except RuntimeError:                     # This is a hack; if the fit fails to converge
        return np.zeros((3),dtype=float)     # give up and just send back some zeros...

# Routine to output the relevant bootstrap statistics for a given function
# This is to make things a little neater and more systematic with data frames
def BootStrapOutput(Function, ParametersArray):
    if Function == "Gaussian":
        Centre = BootStrapAnalysis(ParametersArray[:,1])
        Width = BootStrapAnalysis(ParametersArray[:,2])
        return pd.DataFrame(data = zip(Centre, Width), 
                            columns = ["Centre","Width"])
    if Function == "GB":
        Centre = BootStrapAnalysis(ParametersArray[:,3])
        Width = BootStrapAnalysis(ParametersArray[:,4])
        Temperature = BootStrapAnalysis(ParametersArray[:,5])
        return pd.DataFrame(data = zip(Centre, Width, Temperature),
                            columns=["Centre","Width","Temperature"])
    if Function == "HotGaussian":
        HotFraction = BootStrapAnalysis(ParametersArray[:,4])
        ColdFraction = BootStrapAnalysis(ParametersArray[:,5])
        ColdCentre = BootStrapAnalysis(ParametersArray[:,2])
        ColdWidth = BootStrapAnalysis(ParametersArray[:,3])
        return pd.DataFrame(data = zip(HotFraction, ColdFraction, ColdCentre, ColdWidth),
                            columns=["Hot Fraction", "Cold Fraction", "Cold Centre", "Cold Width"])
    if Function == "DoubleGB":
        CentreA = BootStrapAnalysis(ParametersArray[:,4])
        CentreB = BootStrapAnalysis(ParametersArray[:,5])
        WidthA = BootStrapAnalysis(ParametersArray[:,6])
        WidthB = BootStrapAnalysis(ParametersArray[:,7])
        Temperature = BootStrapAnalysis(ParametersArray[:,8])
        GaussianIntegral = BootStrapAnalysis(ParametersArray[:,9])
        GBIntegral = BootStrapAnalysis(ParametersArray[:,10])
        return pd.DataFrame(data = zip(CentreA, CentreB, WidthA, WidthB, Temperature, GaussianIntegral, GBIntegral),
                            columns = ["3F Centre","T1 Centre","3F Width",
                                       "T1 Width","Temperature","3F Integral", "T1 Integral"])
    if Function == "SumGB":
        Centre = BootStrapAnalysis(ParametersArray[:,2])
        Width = BootStrapAnalysis(ParametersArray[:,3])
        Temperature = BootStrapAnalysis(ParametersArray[:,4])
        GaussianRatio = BootStrapAnalysis(ParametersArray[:,5])
        BoltzmannRatio = BootStrapAnalysis(ParametersArray[:,6])
        return pd.DataFrame(data = zip(Centre, Width, Temperature, GaussianRatio, BoltzmannRatio),
                            columns = ["Gaussian Centre", "Gaussian Width", "Temperature",
                                       "Gaussian Ratio", "Boltzmann Ratio"])
    else:
        print "Function unknown. Exiting."   

def AssumedError(Data, Error):
    # Returns a 1D array of error values in Y
    return np.repeat(Error, len(Data[1]))

# Returns the RMS error for a set of data and its corresponding fit
def CalculateRMS(Data, Fit):
    RMS = np.sqrt(sum((Data[1] - Fit)**2) / len(Data[0]) )
    print "RMS error:\t" + str(RMS)
    return RMS

# Takes input as the target data and the covariance matrix of the fit
# supposed to return a dataframe with everything packaged up
# Parameters is the optimised fitting parameters, and Function is a string
# that tells the routine what function was used to return the error as
def CalculateYError(XData, Parameters, CovarianceMatrix, Function):
    if Function != "DoubleGaussMann":
        StdDev = np.sqrt(np.diag(CovarianceMatrix))
        PlusError = [ (Parameters[i] + StdDev[i]) for i in range(len(StdDev)) ]
        MinusError = [ (Parameters[i] - StdDev[i]) for i in range(len(StdDev)) ]
        if Function == "Linear":
            YplusError = straightline(XData, *PlusError)
            YminusError = straightline(XData, *MinusError)
        elif Function == "GaussMann":
            YplusError = GaussMann(XData, *PlusError)
            YminusError = GaussMann(XData, *MinusError)
        elif Function == "Gaussian":
            YplusError = gauss_function(XData, *PlusError)
            YminusError = gauss_function(XData, *MinusError)
        if Function != "Linear":
            YplusError = YplusError / max(YplusError)
            YminusError = YminusError / max(YminusError)
        return pd.DataFrame(data=zip(XData, YplusError, YminusError))
    else:
        # The double gaussian is a special case since we have so many values to unpack...
        StdDev = np.sqrt(np.diag(CovarianceMatrix))
        PlusError = [ (Parameters[i] + StdDev[i]) for i in range(len(StdDev)) ]
        MinusError = [ (Parameters[i] - StdDev[i]) for i in range(len(StdDev)) ]
        # generate the gaussian
        GaussplusError = gauss_function(XData, PlusError[1], PlusError[4], PlusError[6])
        GaussminusError = gauss_function(XData, MinusError[1], MinusError[4], MinusError[6])
        GBplusError = GaussMann(XData, 
                                PlusError[0], PlusError[2], PlusError[3], PlusError[5],
                               PlusError[7], PlusError[8])
        GBminusError = GaussMann(XData, 
                                MinusError[0], MinusError[2], MinusError[3], MinusError[5],
                               MinusError[7], MinusError[8])
        # return the positive values
        GBplusError = abs(GBplusError)
        GBminusError = abs(GBminusError)
        # Integrate the gaussian and gaussian/boltzmann to get a fraciton out
        GaussplusErrorArea = np.trapz(GaussplusError, XData)
        GaussminusErrorArea = np.trapz(GaussminusError, XData)
        GBplusErrorArea = np.trapz(GBplusError, XData)
        GBminusErrorArea = np.trapz(GBminusError, XData)
        print "PlusFraction:\t" + str( GaussplusErrorArea / (GaussplusErrorArea + GBplusErrorArea))
        print "MinusFraction:\t" + str( GaussminusErrorArea / (GaussminusErrorArea + GBminusErrorArea))
        return pd.DataFrame(data = zip(XData, GaussplusError, GaussminusError, 
                                       GBplusError, GBminusError))
        
######################### Comparison Functions #########################

# Function for compiling the fit results to a list, which we will later turn into
# a pandas dataframe
def PackageParameters(Dataset, Wavelength, Parameters):
    Package = [Wavelength, 1e7 / Wavelength]
    for item in Parameters:
        Package.append(item)
    Dataset.append(Package)
    return Dataset

######################### Export Functions #########################

# Function for packing up whatever and saving it
def ExportFits(Data, Filename, *args):
    Package = zip(Data[0], Data[1])
    for i in enumerate(args):
        Package = zip(Package, args[i])
    Export = pd.DataFrame(data = Package)
    Export.to_csv(Filename)
    
