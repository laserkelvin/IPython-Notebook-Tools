# InteractiveWidgets.py

from IPython.html.widgets import *

"""

Routines for interactive widgets in a
Jupyter notebook.

"""

class SliderWidget:
	def __init__(self, Name):
		self.Name = Name

	def SetFunction(self, Function):
		self.Function = 