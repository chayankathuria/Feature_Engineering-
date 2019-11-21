import pandas as pd
import numpy as np

# Transforming Nominal Data
'''
Nominal data consists of names as labels and we need to transform these labels into numbers so that the algortithm can understand.
One such technique is Label Encoding which converts labels into integers starting from 1.
'''

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

# Transforming Ordinal Data
'''
Ordinal Data is another type of categorical data where the labels cannot be quantized but an order can be set. 
For ex. Unhappy < OK < Happy < Verry Happy , but we cannot quantize how much.
'''

poke_df = pd.read_csv('datasets/Pokemon.csv', encoding='utf-8')
poke_df = poke_df.sample(random_state=1, frac=1).reset_index(drop=True)

np.unique(poke_df['Generation'])
# array(['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5', 'Gen 6'], dtype=object)

gen_ord_map = {'Gen 1': 1, 'Gen 2': 2, 'Gen 3': 3, 
               'Gen 4': 4, 'Gen 5': 5, 'Gen 6': 6}

poke_df['GenerationLabel'] = poke_df['Generation'].map(gen_ord_map)
poke_df[['Name', 'Generation', 'GenerationLabel']].iloc[4:10]

'''
Using map function is just another way to label encode categorical features, ordinal or nominal. But it is recommended only when the number
of features is less because it requires hard-coding.

    Name	          Generation	GenerationLabel
4	Octillery	          Gen 2	          2
5	Helioptile	        Gen 6	          6
6	Dialga	            Gen 4	          4
7	DeoxysDefense Forme	Gen 3	          3
8	Rapidash	          Gen 1	          1
9	Swanna	            Gen 5	          5
'''

