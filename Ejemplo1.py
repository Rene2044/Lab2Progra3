import pandas as pd
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file
csv_path = os.path.join(script_dir, '..', 'youtube-top-100-songs-2025.csv')

datos = pd.read_csv(csv_path)
print(datos.head())
print(datos.shape)