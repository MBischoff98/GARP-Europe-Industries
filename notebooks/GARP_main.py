import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Meta variables
trx_cost = 0.001
trx_costs_high = 0.0025
GrowthTrend = 36
lag = 4
StartingPoint = 90

# Define color palette
colors = {
    'GARP': 'navy',  # Dark Blue
    'Growth': 'darkred',  # Wine Red
    'QV': '#54ff9f',  # Neon Green
    'Benchmark': 'darkgrey'  # Dark Grey
}

fileName = 'data/GARP Data Industries Europe.xlsx' # Change as needed
if 'Sectors' in fileName:
    dataType = 'Sector'
elif 'Industries' in fileName:
    dataType = 'Industry'
else:
    raise ValueError('Invalid file name. The file should be either Sectors or Industries data.')

xls = pd.ExcelFile(fileName)
sheets = xls.sheet_names
dataStruct = {}

for sheet in sheets:
    dataStruct[sheet] = pd.read_excel(fileName, sheet_name=sheet, na_values=['#N/A N/A'])