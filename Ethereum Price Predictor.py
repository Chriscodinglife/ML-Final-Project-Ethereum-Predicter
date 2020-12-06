#### Christian Orellana
#### December 2020
#### Machine Learning
#### Pratt Institute
#### Tkinter App For Ethereum Price Predictor

"""
The following requirements are needed for this app:

1. Python 3.8 https://www.python.org/
2. Chrome Driver https://pypi.org/project/selenium/
3. Selenium https://pypi.org/project/selenium/

"""
##### IMPORT

from datetime import datetime, time
import time as tiempo
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import sys
from tkinter import font
from tkinter import messagebox
import tkinter
import subprocess
import plistlib
import numpy as np
from numpy import nan
from numpy import isnan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from matplotlib import pyplot
from matplotlib.pyplot import figure

logo = '''
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hmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYhmEYxgMi0f8DlInxCNGqoggAAAAASUVORK5CYIJENCtLazkUADCJqNHw4cPHnT9//pEijyfU0HV/i8VSqGlacsOGDdetW7duJSKe972/qud64cKFbfbt2zf5jjvu+OusWbPOAoAkInXSpEn9UlJSHrBardrdd989w+FwZFWUft951jQNnnjiiX4pKSnjizyeCN3jaYiI0mq1nvLz8/ukd+/eyydOnPhf+PW6rOx8IADQihUrmuzcuXNs69at18ybN++07/cVbTN37tyWP/300/DQ0ND3nn/++eyKtlFVFUaOHPmclDJl9erVGwBAvvrqq3d++eWXk3Vd7+HxeJojoqFp2o8tWrR4/7333luFiLLUOUEAoAULFty7a9euiR6Pp6fH42mqKIrHarUeaNGixXtr165d43a7K71ma4raEN19kZiIKHDgwIF/27Fjx1cXLlwYCELsbN269dwbb7wxQQixPS8vL+arr776sn///h8cPHiwCSJSVe60bre7eWFhYfuioqL2bre7fX5+/h8BQOi63rSoqOh2j8fzp8LCwj8VFRXdruu6f2X789490WKxyBEjRkxat25dUn5+/gDTNL9p1KjRa35+fquKior8U1NTV8TExOzYtWtXq6qm1cdmsymIaM6YMaNP9x49fkxPT3/V4/HIoMDAv7Vp0+bFgICAv3k8HpmRkbGke/fuP86cObOvEMKE31Y9ytq3AADIyMi4JzMzc+qxY8f+CABy9uzZkX169dr/448/flRQUPCYruuhAQEBWkX78gUNp9N5Q69evT45cODApsLCwo6axfKvNm3axLds2XK+lPKH8+fPx61bt+6HESNGvFSq9FaldB46dCg0Kytr0fHjx8NK/76ibVKOHw/LyspalJKS8qcKtkEAAF3XlfT09OdPnDgxVNM0OXTo0Bc//vjj/xYWFvY2TXNXkyZNXtM0bbVH1wNSU1Pfe/DBB7/YvHlzC0Qkm82maJpGI0aMeGHLli17c3NzbUT0ddOmTZdpmra2qKio6c8///xB9+7dP9uxY0ezUoGGXSGE4oa0gH79+v07IiKChg8f7iCihqWLyogIRNRg5MiRMyIiIigmJuZrImoEv7ZTVCgoKAiICL3/BkVERJyLi4v7a3BwMBCR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'''


class App:

	def __init__(self, master):

		### Config of Tkinter
		self.master = master
		self.master.resizable(False, False)
		self.master.title("Ethereum Price Predictor")

		self.master.protocol("WM_DELETE_WINDOW", self.close)
		self.master.call('wm', 'attributes', '.', '-topmost', False)
		self.master.geometry("1000x900")

		### Background color
		bgcolor = '#F0F0F0'
		self.master.tk_setPalette(background=bgcolor, highlightbackground=bgcolor)

		### Font
		letter = font.Font(family='system', size=14)
		self.master.option_add("*Font", letter)

		### Menu Bar
		menu_bar = tkinter.Menu(self.master)
		self.master.config(menu=menu_bar)

		### Welcoming Text
		self.frame1 = tkinter.Frame(self.master)
		title_label = tkinter.Label(self.frame1, text="Welcome to the Ethereum Price Predictor")
		title_label.grid(row=0, column= 0)
		self.frame1.pack(padx=40, pady=(10, 5))

		### Logo Image
		self.icon_data = tkinter.PhotoImage(data=logo)
		self.frame2 = tkinter.Frame(self.master)
		self.photo_canvas = tkinter.Canvas(self.frame2, width=250, height=141)
		self.photo_canvas.pack()
		self.icon = self.photo_canvas.create_image(40, 0, anchor="nw", image=self.icon_data)
		self.frame2.pack(padx=40, pady=(30, 5))

		### INPUT VARIABLES
		self.input_date = tkinter.StringVar()
		self.input_nystock_open = tkinter.DoubleVar()
		self.input_nystock_high = tkinter.DoubleVar()
		self.input_nystock_low = tkinter.DoubleVar()
		self.input_nystock_close = tkinter.DoubleVar()
		self.input_nystock_adj_close = tkinter.DoubleVar()
		self.input_dowjones_open = tkinter.DoubleVar()
		self.input_dowjones_high = tkinter.DoubleVar()
		self.input_dowjones_low = tkinter.DoubleVar()
		self.input_dowjones_close = tkinter.DoubleVar()
		self.input_dowjones_adj_close = tkinter.DoubleVar()
		self.input_shanghai_open = tkinter.DoubleVar()
		self.input_shanghai_high = tkinter.DoubleVar()
		self.input_shanghai_low = tkinter.DoubleVar()
		self.input_shanghai_close = tkinter.DoubleVar()
		self.input_shanghai_adj_close = tkinter.DoubleVar()
		self.input_ethereum_open = tkinter.DoubleVar()
		self.input_ethereum_high = tkinter.DoubleVar()
		self.input_ethereum_low = tkinter.DoubleVar()

		self.output_result = tkinter.DoubleVar()
		self.output_error = tkinter.DoubleVar()

		
		### INPUT FRAME
		self.frame3 = tkinter.Frame(self.master)
		
		intro_text = tkinter.Label(self.frame3, text="Provide NYSE, DJI, SHANGHAI, and Ethereum data with date:")
		

		#Date
		date_label = tkinter.Label(self.frame3, text="Date:")
		self.entry_date = tkinter.Entry(self.frame3, background='white', textvariable=self.input_date, width=15)
		# NYstock Open
		nystock_open_label = tkinter.Label(self.frame3, text="NYSTOCK Open:")
		self.entry_nystock_open = tkinter.Entry(self.frame3, background='white', textvariable=self.input_nystock_open, width=15)
		# Nystock High
		nystock_high_label = tkinter.Label(self.frame3, text="NYSTOCK High:")
		self.entry_nystock_high = tkinter.Entry(self.frame3, background='white', textvariable=self.input_nystock_high, width=15)
		#low
		nystock_low_label = tkinter.Label(self.frame3, text="NYSTOCK Low:")
		self.entry_nystock_low = tkinter.Entry(self.frame3, background='white', textvariable=self.input_nystock_low, width=15)
		#close
		nystock_close_label = tkinter.Label(self.frame3, text="NYSTOCK Close:")
		self.entry_nystock_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_nystock_close, width=15)
		#adj close
		nystock_adj_close_label = tkinter.Label(self.frame3, text="NYSTOCK ADJ Close:")
		self.entry_nystock_adj_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_nystock_adj_close, width=15)


		# DowJones Open
		dowjones_open_label = tkinter.Label(self.frame3, text="DowJones Open:")
		self.entry_dowjones_open = tkinter.Entry(self.frame3, background='white', textvariable=self.input_dowjones_open, width=15)
		# DowJones High
		dowjones_high_label = tkinter.Label(self.frame3, text="DowJones High:")
		self.entry_dowjones_high = tkinter.Entry(self.frame3, background='white', textvariable=self.input_dowjones_high, width=15)
		#low
		dowjones_low_label = tkinter.Label(self.frame3, text="DowJones Low:")
		self.entry_dowjones_low = tkinter.Entry(self.frame3, background='white', textvariable=self.input_dowjones_low, width=15)
		#close
		dowjones_close_label = tkinter.Label(self.frame3, text="DowJones Close:")
		self.entry_dowjones_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_dowjones_close, width=15)
		#adj close
		dowjones_adj_close_label = tkinter.Label(self.frame3, text="DowJones ADJ Close:")
		self.entry_dowjones_adj_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_dowjones_adj_close, width=15)


		# Shanghai Open
		shanghai_open_label = tkinter.Label(self.frame3, text="Shanghai Open:")
		self.entry_shanghai_open = tkinter.Entry(self.frame3, background='white', textvariable=self.input_shanghai_open, width=15)
		# Shanghai High
		shanghai_high_label = tkinter.Label(self.frame3, text="shanghai High:")
		self.entry_shanghai_high = tkinter.Entry(self.frame3, background='white', textvariable=self.input_shanghai_high, width=15)
		#low
		shanghai_low_label = tkinter.Label(self.frame3, text="shanghai Low:")
		self.entry_shanghai_low = tkinter.Entry(self.frame3, background='white', textvariable=self.input_shanghai_low, width=15)
		#close
		shanghai_close_label = tkinter.Label(self.frame3, text="shanghai Close:")
		self.entry_shanghai_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_shanghai_close, width=15)
		#adj close
		shanghai_adj_close_label = tkinter.Label(self.frame3, text="shanghai ADJ Close:")
		self.entry_shanghai_adj_close = tkinter.Entry(self.frame3, background='white', textvariable=self.input_shanghai_adj_close, width=15)

		# ethereum Open
		ethereum_open_label = tkinter.Label(self.frame3, text="ethereum Open:")
		self.entry_ethereum_open = tkinter.Entry(self.frame3, background='white', textvariable=self.input_ethereum_open, width=15)
		# ethereum High
		ethereum_high_label = tkinter.Label(self.frame3, text="ethereum High:")
		self.entry_ethereum_high = tkinter.Entry(self.frame3, background='white', textvariable=self.input_ethereum_high, width=15)
		#low
		ethereum_low_label = tkinter.Label(self.frame3, text="ethereum Low:")
		self.entry_ethereum_low = tkinter.Entry(self.frame3, background='white', textvariable=self.input_ethereum_low, width=15)

		
		# Pack the labels and entry boxes

		intro_text.grid(row=1, column=1, padx=5, pady=(2, 8))
		date_label.grid(row=2, column=1, padx=5, pady=(2, 5))
		self.entry_date.grid(row=3, column=1, padx=5, pady=(2, 5))
		
		nystock_open_label.grid(row=4, column=0, padx=5, pady=(2, 8))
		self.entry_nystock_open.grid(row=5, column=0, padx=5, pady=(2, 8))
		nystock_high_label.grid(row=6, column=0, padx=5, pady=(2, 8))
		self.entry_nystock_high.grid(row=7, column=0, padx=5, pady=(2, 8))
		nystock_low_label.grid(row=8, column=0, padx=5, pady=(2, 8))
		self.entry_nystock_low.grid(row=9, column=0, padx=5, pady=(2, 8))
		nystock_close_label.grid(row=10, column=0, padx=5, pady=(2, 8))
		self.entry_nystock_close.grid(row=11, column=0, padx=5, pady=(2, 8))
		nystock_adj_close_label.grid(row=12, column=0, padx=5, pady=(2, 8))
		self.entry_nystock_adj_close.grid(row=13, column=0, padx=0, pady=(2, 8))

		dowjones_open_label.grid(row=4, column=1, padx=5, pady=(2, 8))
		self.entry_dowjones_open.grid(row=5, column=1, padx=5, pady=(2, 8))
		dowjones_high_label.grid(row=6, column=1, padx=5, pady=(2, 8))
		self.entry_dowjones_high.grid(row=7, column=1, padx=5, pady=(2, 8))
		dowjones_low_label.grid(row=8, column=1, padx=5, pady=(2, 8))
		self.entry_dowjones_low.grid(row=9, column=1, padx=5, pady=(2, 8))
		dowjones_close_label.grid(row=10, column=1, padx=5, pady=(2, 8))
		self.entry_dowjones_close.grid(row=11, column=1, padx=5, pady=(2, 8))
		dowjones_adj_close_label.grid(row=12, column=1, padx=5, pady=(2, 8))
		self.entry_dowjones_adj_close.grid(row=13, column=1, padx=5, pady=(2, 8))

		shanghai_open_label.grid(row=4, column=2, padx=5, pady=(2, 8))
		self.entry_shanghai_open.grid(row=5, column=2, padx=5, pady=(2, 8))
		shanghai_high_label.grid(row=6, column=2, padx=5, pady=(2, 8))
		self.entry_shanghai_high.grid(row=7, column=2, padx=5, pady=(2, 8))
		shanghai_low_label.grid(row=8, column=2, padx=5, pady=(2, 8))
		self.entry_shanghai_low.grid(row=9, column=2, padx=5, pady=(2, 8))
		shanghai_close_label.grid(row=10, column=2, padx=5, pady=(2, 8))
		self.entry_shanghai_close.grid(row=11, column=2, padx=5, pady=(2, 8))
		shanghai_adj_close_label.grid(row=12, column=2, padx=5, pady=(2, 8))
		self.entry_shanghai_adj_close.grid(row=13, column=2, padx=5, pady=(2, 8))

		ethereum_open_label.grid(row=4, column=3, padx=5, pady=(2, 8))
		self.entry_ethereum_open.grid(row=5, column=3, padx=5, pady=(2, 8))
		ethereum_high_label.grid(row=6, column=3, padx=5, pady=(2, 8))
		self.entry_ethereum_high.grid(row=7, column=3, padx=5, pady=(2, 8))
		ethereum_low_label.grid(row=8, column=3, padx=5, pady=(2, 8))
		self.entry_ethereum_low.grid(row=9, column=3, padx=5, pady=(2, 8))

		self.frame3.pack(padx=40, pady=10)

		### BUTTONS

		self.frame4 = tkinter.Frame(self.master)

		predict = tkinter.Button(self.frame4, text='Predict', height=1, command=self.predict)
		predict.grid(row=0, column=1, padx=10, pady=(2, 15))

		output_label = tkinter.Label(self.frame4, text="Predicted Closed and Adj Close:")
		output_label.grid(row=1, column=1, padx=5, pady=(2, 8))

		output_result_label = tkinter.Label(self.frame4, text="(results)")
		output_result_label.grid(row=1, column=2, padx=5, pady=(2, 8))

		output_error_label = tkinter.Label(self.frame4, text="Error Amount:")
		output_error_label.grid(row=2, column=1, padx=5, pady=(2, 8))

		output_error_result_label = tkinter.Label(self.frame4, text="(error amount)")
		output_error_result_label.grid(row=2, column=2, padx=5, pady=(2, 8))

		self.frame4.pack(padx=40, pady=(5, 30))

		self.frame5 = tkinter.Frame(self.master)

		self.frame5.pack(padx=40, pady=(5,30))


		self.frame6 = tkinter.Frame(self.master)
		close = tkinter.Button(self.frame6, text='Close', height=1, width=8, command=self.close)
		close.pack(side='right')
		self.frame6.pack(padx=40, pady=(5, 30))


		### Assignment of Variables
		self.user_date=self.input_date
		self.user_nystock_open=self.input_nystock_open
		self.user_nystock_high=self.input_nystock_high
		self.user_nystock_low=self.input_nystock_low
		self.user_nystock_close=self.input_nystock_close
		self.user_nystock_adj_close=self.input_nystock_adj_close

		self.user_dowjones_open=self.input_dowjones_open
		self.user_dowjones_high=self.input_dowjones_high
		self.user_dowjones_low=self.input_dowjones_low
		self.user_dowjones_close=self.input_dowjones_close
		self.user_dowjones_adj_close=self.input_dowjones_adj_close

		self.user_shanghai_open=self.input_shanghai_open
		self.user_shanghai_high=self.input_shanghai_high
		self.user_shanghai_low=self.input_shanghai_low
		self.user_shanghai_close=self.input_shanghai_close
		self.user_shanghai_adj_close=self.input_shanghai_adj_close

		self.user_ethereum_open=self.input_ethereum_open
		self.user_ethereum_high=self.input_ethereum_high
		self.user_ethereum_low=self.input_ethereum_low


		

		self.user_date=self.input_date
		self.user_nystock_open=self.input_nystock_open
		self.user_nystock_high=self.input_nystock_high
		self.user_nystock_low=self.input_nystock_low
		self.user_nystock_close=self.input_nystock_close
		self.user_nystock_adj_close=self.input_nystock_adj_close

		self.user_dowjones_open=self.input_dowjones_open
		self.user_dowjones_high=self.input_dowjones_high
		self.user_dowjones_low=self.input_dowjones_low
		self.user_dowjones_close=self.input_dowjones_close
		self.user_dowjones_adj_close=self.input_dowjones_adj_close

		self.user_shanghai_open=self.input_shanghai_open
		self.user_shanghai_high=self.input_shanghai_high
		self.user_shanghai_low=self.input_shanghai_low
		self.user_shanghai_close=self.input_shanghai_close
		self.user_shanghai_adj_close=self.input_shanghai_adj_close

		self.user_ethereum_open=self.input_ethereum_open
		self.user_ethereum_high=self.input_ethereum_high
		self.user_ethereum_low=self.input_ethereum_low

		




	#### FUNCTIONS

	
	def predict(self):

		# Create a function to take an input from a user and prepare it for prediction
		def predict_button(value):
			"This function will be used to prepare the input to a format that our polynomial model can understand"

			###### Machine Learning Section

			# Sample DataSets for Practice
			# New york Stock Exchange
			# nystock = pd.read_csv("data/nov 3 - dec 3/nystockex.csv")
			# # Dow Jones Industrial Average 
			# dowjones = pd.read_csv("data/nov 3 - dec 3/dowjones.csv")
			# # Shanghai Composite Index
			# shanghai = pd.read_csv("data/nov 3 - dec 3/shanghai.csv")
			# # Ethereum Price
			# ethereum = pd.read_csv("data/nov 3 - dec 3/ethereum.csv")

			# December 2019 to December 2020 Data Sets
			# New york Stock Exchange
			nystock = pd.read_csv("data/dec3-19 to dec4-20/nystock.csv")
			# Dow Jones Industrial Average 
			dowjones = pd.read_csv("data/dec3-19 to dec4-20/dowjones.csv")
			# Shanghai Composite Index
			shanghai = pd.read_csv("data/dec3-19 to dec4-20/shanghai.csv")
			# Ethereum Price
			ethereum = pd.read_csv("data/dec3-19 to dec4-20/ethereum.csv")

			# Create a function to remove the volume column and then convert the date time to a int timestamp

			def convert_to_timestamp_rm_volume(data):
				"This will convert the first column that is a data object to a date type and then convert it to a int object"
				data['Date'] = pd.to_datetime(data['Date']).dt.strftime("%Y%m%d")
				data['Date'] = pd.to_datetime(data['Date'], format = "%Y%m%d")
				data['Date'] = data['Date'].view('int64')
				data.drop(columns='Volume', axis = 1, inplace = True)
				return data

			# Create a function that will test to see if the data is a pd.dataframe

			def testframe(data):
				"This will check if the input is a Pandas Dataframe object"
				if isinstance(data, pd.DataFrame):
					print("The input is a dataframe")
				else:
					print("This is not a dataframe")


			# Run a for loop to run the conversion timestamp function

			data_list = [nystock, dowjones, shanghai, ethereum]
			for i in data_list:
				convert_to_timestamp_rm_volume(i)

			# Create a merger function for the datasets 
			def merger(data1, data2, data3, data4, column, method):
				"This will merge all the data sets into one dataset"
				data = pd.merge(data1, data2, on = column, how = method)
				data = pd.merge(data, data3, on = column, how = method)
				data = pd.merge(data, data4, on = column, how = method)
				return data

			guessdata = merger(nystock, dowjones, shanghai, ethereum, "Date", "outer")

			# Prepare the data for piping by removing the dates column
			guessdata_dates = guessdata.iloc[:,0:1]
			official_column_names = guessdata.columns.tolist()
			column_names_for_input = official_column_names[:-2]
			guessdata_nodates = guessdata.drop(columns='Date', axis = 1)

			# Run the imputer and convert the data back to a dataframe with scale
			imputer = SimpleImputer(missing_values=nan, strategy='mean')
			guessdata_nodates = imputer.fit_transform(guessdata_nodates)
			guessdata_nodates = pd.DataFrame(guessdata_nodates)
			scaler = StandardScaler()
			scaler.fit(guessdata_nodates.iloc[:,0:18])
			# guessdata_nodates_scaled = preprocessing.scale(guessdata_nodates)
			guessdata_nodates_scaled = scaler.transform(guessdata_nodates.iloc[:,0:18])
			guessdata_nodates_scaled = pd.DataFrame(guessdata_nodates_scaled)

			# Put back the dates column into the final cleaned dataset and rename the columns
			guessdata_withdates_scaled = guessdata_nodates_scaled
			guessdata_withdates_scaled['Date'] = guessdata_dates
			dates = guessdata_withdates_scaled['Date']
			guessdata_withdates_scaled.drop(columns='Date', axis = 1, inplace = True)
			guessdata_withdates_scaled.insert(0, 'Date', dates)
			guessdata_withdates_scaled.columns = column_names_for_input


			# Define the data and target values
			x = guessdata_withdates_scaled.iloc[:,:19]
			y = guessdata.iloc[:,19:21]
			y = preprocessing.scale(y)
			y = pd.DataFrame(y)
			y.columns = ['Close_y', 'Adj_Close_y']

			pft = PolynomialFeatures(degree = 2)
			X_poly = pft.fit_transform(x)
			X_poly = pd.DataFrame(X_poly)
			X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.3, random_state = 42)
			poly_model = linear_model.Ridge(alpha = 400)
			redge2 = Ridge(alpha = 400)

			poly_model.fit(X_train, y_train)
			# Predict and determine the mean squared error of the polynomial regression model
			y_pred = poly_model.predict(X_test)
			error = mean_squared_error(y_test, y_pred)
			
			
			
			# Check to make sure that the input was 19 values in total
			# Check to make sure that the input was 19 values in total
			assert value != 19
			
			# Convert the input into a dataframe
			value = pd.DataFrame(value)
			
			# Define the names of the columns
			value.columns = column_names_for_input
			
			# Convert the date column into a timestamp
			value['Date'] = pd.to_datetime(value['Date']).dt.strftime("%Y%m%d")
			value['Date'] = pd.to_datetime(value['Date'], format = "%Y%m%d")
			value['Date'] = value['Date'].view('int64')
			
			# Separate the date column and prepare the data for scaling
			value_date = value.iloc[:,0:1]
			value_nodate = value.drop(columns='Date', axis = 1)
			
			# Run scaling on the input and convert back to dataframe
			value_nodate_scaled = scaler.transform(value_nodate)
			value_nodate_scaled = pd.DataFrame(value_nodate_scaled)
			
			# Put back the date column and rename the columns 
			value_withdate_scaled = value_nodate_scaled
			value_withdate_scaled['Date'] = value_date
			date = value_withdate_scaled['Date']
			value_withdate_scaled.drop(columns='Date', axis = 1, inplace = True)
			value_withdate_scaled.insert(0, 'Date', date)
			value_withdate_scaled.columns = column_names_for_input

			
			output = poly_model.predict(pft.fit_transform(value))

			user_error = mean_squared_error(y_test, y_pred)

			self.output_error = user_error

			self.output_result = output
			
			print(self.output_result)
			print(user_error)

			self.output_error = user_error
			output_result_label = tkinter.Label(self.frame4, text=self.output_result)
			output_result_label.grid(row=1, column=2, padx=5, pady=(2, 8))

			output_error_result_label = tkinter.Label(self.frame4, text=self.output_error)
			output_error_result_label.grid(row=2, column=2, padx=5, pady=(2, 8))
			



		date = self.user_date.get()
		nystock_open = self.user_nystock_open.get()
		nystock_high = self.user_nystock_high.get()
		nystock_low = self.user_nystock_low.get()
		nystock_close = self.user_nystock_close.get()
		nystock_adj_close = self.user_nystock_adj_close.get()
		dowjones_open = self.user_dowjones_open.get()
		dowjones_high = self.user_dowjones_high.get()
		dowjones_low = self.user_dowjones_low.get()
		dowjones_close = self.user_dowjones_close.get()
		dowjones_adj_close = self.user_dowjones_adj_close.get()
		shanghai_open = self.user_shanghai_open.get()
		shanghai_high = self.user_shanghai_high.get()
		shanghai_low = self.user_shanghai_low.get()
		shanghai_close = self.user_shanghai_close.get()
		shanghai_adj_close = self.user_shanghai_adj_close.get()
		ethereum_open = self.user_ethereum_open.get()
		ethereum_high = self.user_ethereum_high.get()
		ethereum_low = self.user_ethereum_low.get()


		user_input = [[date,
				nystock_open,
				nystock_high,
				nystock_low,
				nystock_close,
				nystock_adj_close,
				dowjones_open,
				dowjones_high,
				dowjones_low,
				dowjones_close,
				dowjones_adj_close,
				shanghai_open,
				shanghai_high,
				shanghai_low,
				shanghai_close,
				shanghai_adj_close,
				ethereum_open,
				ethereum_high,
				ethereum_low]]
		
		print(user_input)
		predict_button(user_input)
		

	def close(self):

		self.master.destroy()



def main():

	root = tkinter.Tk()
	app = App(root)
	app.master.mainloop()

if __name__ == '__main__':
	main()

