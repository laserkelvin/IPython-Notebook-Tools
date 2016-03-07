#!/bin/python

# notebook.py
# Contains supplementary routines for use in the iPython notebooks

#import pybel as pb
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
#from chemviewer import enable_notebook, MolecularViewer
#import openbabel as ob
#from openbabel import OBElementTable as obt

################# ChemViewer ###################

#def GetCoords(file,format):
# routine for parsing a chemical file format into
# python constructs with pybel and openbabel

#    molecule = pb.readfile(format,file).next()  # load into pybel
# get the coordinates, loop over individual atoms in molecule
#    molcoords = [atom.coords for atom in molecule]
# get the atomic numbers and then translate to atomic symbol
#    molsym = [obt().GetSymbol(atom.atomicnum) for atom in molecule]
#    return molcoords, molsym

#def DrawMolecule(coords,symbols,bonds=None):
# render a molecule in a notebook
#    mv = MolecularViewer(coords,topology={'atom_types': symbols,'bonds': bonds})
#    mv.ball_and_sticks()
#    mv

################## General notebook functions ####################

# Because I'm a lazy and forgetful SOB write a function to read with pandas
def PandaRead(file,delimiter="\t"):
    df = pd.read_csv(file,delimiter=delimiter,header=None)  # by default delimiter is tab
    df = df.dropna(axis=0)           # removes all NaN values
    return df

################### Speed Distribution Analysis ###################

# function to convert speed into kinetic energy, using data frames
def Speed2KER(Data, Mass):
    KER = np.zeros(len(Data[0]), dtype=float)
    for index in range(len(Data[0])):
        KER[index] = ((Data[0][index]**2) * Mass / 2000) * 83.59
    return KER

# function to convert P(s) into P(E), using Pandas data frames
def PS2PE(Data, Mass):
    PE = np.zeros(len(Data[1]), dtype=float)
    for index in range(len(Data[0])):
        PE[index] = Data[1][index] / (Mass * Data[0][index])
    return PE

# Function to convert a speed distribution loaded with Pandas dataframe into a
# kinetic energy distribution
def ConvertSpeedToKER(Data, Mass):
    KER = Speed2KER(Data, Mass)
    PE = PS2PE(Data, Mass)
    return pd.DataFrame(data = zip(KER, PE))

################### General analysis functions ##################

def SplitArray(x,index):          # For a given array, split into two based on index
    A = x[index:]
    B = x[:index]
    return A, B

def find_nearest(array,value):    # Returns the index for the value closest to the specified
    idx = (np.abs(array-value)).argmin()
    return idx

# Uses the Savitzky-Golay filter to smooth an input array Y
def SGFilter(Y, WindowSize, Order=2, Deriv=0, Rate=1):
    return savgol_filter(Y, WindowSize, Order, Deriv)
