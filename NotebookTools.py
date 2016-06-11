#!/bin/python

# notebook.py
# Contains supplementary routines for use in the iPython notebooks

#import pybel as pb
import pandas as pd
import numpy as np
import csv
import cPickle as pickle
from scipy import constants
from scipy.signal import savgol_filter
from contextlib import contextmanager
import sys
import os
import glob
import h5py

################## General notebook functions ####################

@contextmanager
def suppress_stdout():
    """ Will supress the standard output
        Usage:
        with suppress_stdout():
            function_call()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def ConvertUnit(Initial, Target):
    """ Convert units by specify what the initial unit is
    and what you want the unit to be.

    Returns the conversion factor.
    """
    UnitConversions = pd.DataFrame(index=["Hartree",
                                          "eV",
                                          "1/cm", 
                                          "kcal/mol",
                                          "kJ/mol",
                                          "Kelvin",
                                          "Joule",
                                          "Hz"])
    UnitConversions["Hartree"] = zip([1., 
                                 0.0367502, 
                                 4.55633e-6, 
                                 0.00159362, 
                                 0.00038088,
                                 0.00000316678,
                                 2.294e17,
                                 1.51983e-16])
    UnitConversions["eV"] = zip([27.2107,
                            1.,
                            1.23981e-4,
                            0.0433634,
                            0.01036410,
                            0.0000861705,
                            6.24181e18,
                            4.13558e-15])
    UnitConversions["1/cm"] = zip([219474.63,
                                  8065.73,
                                  1.,
                                  349.757,
                                  83.593,
                                  0.695028,
                                  5.03445e22,
                                  3.33565e-11])
    UnitConversions["kcal/mol"] = zip([627.503,
                                      23.0609,
                                      0.00285911,
                                      1.,
                                      0.239001,
                                      0.00198717,
                                      1.44e20,
                                      9.53702e-14])
    UnitConversions["kJ/mol"] = zip([2625.5,
                                    96.4869,
                                    0.0119627,
                                    4.18400,
                                    1.0,
                                    0.00831435,
                                    6.02e20,
                                    0.])
    UnitConversions["Kelvin"] = zip([315777,
                                    11604.9,
                                    1.42879,
                                    503.228,
                                    120.274,
                                    1.,
                                    7.24354e22,
                                    4.79930e-11])
    UnitConversions["Joule"] = zip([43.60e-19,
                                   1.60210e-19,
                                   1.98630e-23,
                                   6.95e-21,
                                   1.66e-21,
                                   1.38054e-23,
                                   1.,
                                   6.62561e-34])
    UnitConversions["Hz"] = zip([6.57966e15,
                                2.41804e14,
                                2.99793e10,
                                1.04854e13,
                                2.50607e12,
                                2.08364e10,
                                1.50930e33,
                                1.])
    return UnitConversions[Initial][Target][0]

############################ File I/O ############################

# Because I'm a lazy and forgetful SOB write a function to read with pandas
def PandaRead(file):
    Delimiter = DetectDelimiter(file)
    df = pd.read_csv(file,delimiter=Delimiter,header=None)
    df = df.dropna(axis=0)           # removes all NaN values
    return df

def DetectDelimiter(File):
    """ Routine to see what delimiter the file has
    """
    sniffer = csv.Sniffer()
    f = open(File, "r")                   # open file and read the first line
    fc = f.readline()
    f.close()
    line = sniffer.sniff(fc)
    return line.delimiter

def DetectHeader(File):
    """ Routine to detect whether or not there are headers
        in the csv file.
    """
    with open(File, "rU") as f:        # "rU" opens in universal newline mode!
        return csv.Sniffer().has_header(f.read(1024))

def SaveObject(Object, Database):
    """ Function for saving class structure data
        to a database using pickle

        The way to use this would be to have a dictionary
        with all of the instances in my notebook, with the
        dictionary keys as the references
    """
    with open(Database, "wb") as db:
        pickle.dump(Object, db, pickle.HIGHEST_PROTOCOL)
    db.close()

def LoadObject(Database):
    """ Function to read in a pickle database
    """
    with open(Database, "r") as db:
        temp = pickle.load(db)
    db.close()
    return temp

###################         HDF5 Tools          ###################

""" Technically should be under I/O, but I think this is a special
    case since I hope to use it more often...

    Still need to write a write data function to add stuff into
    a database.
"""

def StripSuffixes(Filename):
    """ Removes all of the suffixes for a file, which means
        the file extension and also anything that I might've
        added as comment like _CBS or _REMPI.
    """
    return Filename.split(".")[0].split("_")[0]

def StripExtension(Filename):
    """ Strips the extension of the file only. """
    return Filename.split(".")[0]

def LoadDatabase(Database):
    """ Attempts to load a database with exception catching.
        If the database doesn't exist, raise an error.
        
        Returns the loaded database
    """
    try:
        Temp = h5py.File(Database, "r+")
    except IOError:
        print "Database does not exist."
        exit()
    return Temp

def LoadReference(Database, Reference):
    """ Used to retrieve a reference from a database.
        Returns all of the data in a dictionary associated 
        with a logbook reference.
    """
    Dictionary = dict()
    for Key in Database[Reference].keys():
        Dictionary[Key] = Database[Reference][Key][...]
    return Dictionary

def AddDatabaseEntry(Database, File):
    """ Used to add a file to a database. """
    Reference = StripSuffixes(File)
    if Reference not in Database.keys():         # See if there's already a group
        Database.create_group("/" + Reference)   # in the database, if not create it
    try:
        Database[Reference].create_dataset(StripExtension(File),    # Load the data
                                           data=np.fromfile(File),  # into database
                                           compression="gzip",      # and compress it
                                           compression_opts=9
                                           )
    except (RuntimeError, ValueError):                          # Catches numpy load
        print StripExtension(File) + " could be loaded."        # and compression errors
        pass

def PackDirectory(Database, FileTypes=["*.dat", "*.bin"]):
    """ Pack a directory of files with a given extension into
        an HDF5 database.
        I wrote this with logbook references in mind, so the
        general group hierarchy is:
        
        /Reference/Datafile/Data
        
        where Data is the actual dataset, and Datafile is the
        actual filename, and Reference is the logbook reference
        stripped of all suffixes.
    """
    with h5py.File(Database,"a") as HF:
        for Files in FileTypes:              # Loop over the specified filetypes
            A = glob.glob(Files)             # and generate a list of files
            for item in A:
                AddDatabaseEntry(HF, item)   # Call routine to add a file
        HF.close() 


################### Speed Distribution Analysis ###################

def amu2kg(Mass):
    """Converts mass in atomic units to kg
    """
    return (Mass / constants.Avogadro) / 1000

""" Mass in the following functions is specified in kg/mol """
# function to convert speed into kinetic energy, using data frames
def Speed2KER(Data, Mass):
    """ Convert metres per second to kinetic energy
        in wavenumbers.
    """
    KER = np.zeros(len(Data[0]), dtype=float)
    for index, energy in enumerate(Data[0]):
        KER[index] = ((energy**2) * Mass / 2000) * 83.59
    return KER

def KER2Speed(Data, Mass):
    """ Convert kinetic energy in 1/cm to metres per second.
    """
    Speed = np.zeros(len(Data[0]), dtype=float)
    for index, energy in enumerate(Data[0]):
        Speed[index] = np.sqrt(((energy / 83.59) * 2000.) / Mass)
    return Speed

# function to convert P(s) into P(E), using Pandas data frames
def PS2PE(Data, Mass):
    PE = np.zeros(len(Data[1]), dtype=float)
    for index, probability in enumerate(Data[1]):
        PE[index] = probability / (Mass * Data[0][index])
    return PE

def PE2PS(Data, Mass):
    """ Convert translational energy probability
        to speed probability.
    """
    PS = np.zeros(len(Data[1]), dtype=float)
    for index, probability in enumerate(Data[1]):
        PS[index] = probability * (Mass * Data[0][index])
    return PS

# Function to convert a speed distribution loaded with Pandas dataframe into a
# kinetic energy distribution
def ConvertSpeedToKER(Data, Mass):
    KER = Speed2KER(Data, Mass)
    PE = PS2PE(Data, Mass)
    return pd.DataFrame(data = PE,
                        index = KER,
                        columns=["PE"])

# Function to convert a translational energy distribution 
# loaded with Pandas dataframe into a speed distribution
def ConvertKERToSpeed(Data, Mass):
    Speed = KER2Speed(Data, Mass)
    PS = PE2PS(Data, Mass)
    return pd.DataFrame(data = PS,
                        index = Speed,
                        columns=["PS"])

######################## Data List Functions ####################

def InitialiseDataList():
    return pd.DataFrame(index=["Data type",
                                 "Pump wavelength",
                                 "Probe wavelength",
                                 "Background reference",
                                 "Tags",
                                 "Comments"])

def AddDataEntry(DataList):
    Reference = raw_input("What is the data reference?")
    DataType = raw_input("What kind of data is \t" + Reference + "?")
    PumpWavelength = raw_input("What was the pump wavelength?")
    ProbeWavelength = raw_input("What was the probe wavelength?")
    BackgroundReference = raw_input("What is the logbook reference for background data?")
    Tags = raw_input("Tags to group the data?")
    Comments = raw_input("Any extra comments?")
    DataList[Reference] = [DataType,
                           PumpWavelength, 
                           ProbeWavelength, 
                           BackgroundReference,
                           Tags,
                           Comments]

################### General analysis functions ##################

def SplitArray(x,index):          # For a given array, split into two based on index
    A = x[index:]
    B = x[:index]
    return A, B

def CheckOffDiagonal(Array):
    """ Extract the off-diagonal elements of an array
        and return a sorted list quickly.
    """
    iterator = np.nditer(Array, flags=['multi_index'])
    Elements = []
    while not iterator.finished:   # start the loop
        if np.diff(iterator.multi_index) != 0:   # if the indices aren't equal
            Elements.append(iterator[0])
        else:
            pass
        iterator.iternext()        # next iteration of loop
    Elements.sort()                # sort by size, ascending order
    return Elements

def find_nearest(array,value):    # Returns the index for the value closest to the specified
    idx = (np.abs(array-value)).argmin()
    return idx

# Uses the Savitzky-Golay filter to smooth an input array Y
def SGFilter(Y, WindowSize, Order=2, Deriv=0, Rate=1):
    if WindowSize % 2 == 1:
        return savgol_filter(Y, int(WindowSize), Order, Deriv)
    else:
        print " WindowSize is " + str(WindowSize) + " which isn't odd!"
        print " Please specify an odd number!"
