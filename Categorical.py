import pandas as pd
import numpy as np

vg_df = pd.read_csv('datasets/vgsales.csv', encoding='utf-8')
vg_df[['Name', 'Platform', 'Year', 'Genre', 'Publisher']].iloc[1:7]

# Preparing dataframe for label encoding
genres = np.unique(vg_df['Genre'])
genres

from sklearn.preprocessing import LabelEncoder

gle = LabelEncoder()
genre_labels = gle.fit_transform(vg_df['Genre'])
genre_mappings = {index: label for index, label in enumerate(gle.classes_)} # Dsiplaying all the categories and their labels 
genre_mappings

vg_df['GenreLabel'] = genre_labels
vg_df[['Name', 'Platform', 'Year', 'Genre', 'GenreLabel']].iloc[1:7]  # Adding the Label Encoded column to the DF 
