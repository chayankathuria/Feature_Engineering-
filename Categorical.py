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

# One-Hot Encoding 

'''
Label Encoding Encodes labels with value between 0 and n_classes-1.

OHE, on the other hand, Encodes categorical integer features using a one-hot aka one-of-K scheme.

The input to the transformer should be a matrix of integers, denoting
the values taken on by categorical (discrete) features. The output will be
a sparse matrix where each column corresponds to one possible value of one
feature. It is assumed that input features take on values in the range
[0, n_values).

'''
# Let's take a look at our df
poke_df[['Name', 'Generation', 'Legendary']].iloc[4:10]

'''
     Name	         Generation	 Legendary
4	Octillery	          Gen 2	    False
5	Helioptile	        Gen 6	    False
6	Dialga	            Gen 4	    True
7	DeoxysDefense Forme	Gen 3	    True
8	Rapidash	          Gen 1	    False
9	Swanna	            Gen 5	    False
'''

# Running label encoder first
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# transform and map pokemon generations
gen_le = LabelEncoder()
gen_labels = gen_le.fit_transform(poke_df['Generation'])
poke_df['Gen_Label'] = gen_labels

# transform and map pokemon legendary status
leg_le = LabelEncoder()
leg_labels = leg_le.fit_transform(poke_df['Legendary'])
poke_df['Lgnd_Label'] = leg_labels

poke_df_sub = poke_df[['Name', 'Generation', 'Gen_Label', 'Legendary', 'Lgnd_Label']]
poke_df_sub.iloc[4:10]

'''
Name	Generation	Gen_Label	  Legendary	Lgnd_Label
4	Octillery	            Gen 2	    1	      False	0
5	Helioptile	          Gen 6	    5	      False	0
6	Dialga	              Gen 4	    3	      True	1
7	DeoxysDefense Forme	  Gen 3	    2	      True	1
8	Rapidash	            Gen 1	    0	      False	0
9	Swanna	              Gen 5	    4	      False	0
'''

# encode generation labels using one-hot encoding scheme
gen_ohe = OneHotEncoder()
gen_feature_arr = gen_ohe.fit_transform(poke_df[['Gen_Label']]).toarray()
gen_feature_labels = list(gen_le.classes_)
gen_features = pd.DataFrame(gen_feature_arr, columns=gen_feature_labels)

# encode legendary status labels using one-hot encoding scheme
leg_ohe = OneHotEncoder()
leg_feature_arr = leg_ohe.fit_transform(poke_df[['Lgnd_Label']]).toarray()
leg_feature_labels = ['Legendary_'+str(cls_label) for cls_label in leg_le.classes_]
leg_features = pd.DataFrame(leg_feature_arr, columns=leg_feature_labels)

# Next we have to concatenate the new dfs into the original df

poke_df_ohe = pd.concat([poke_df_sub, gen_features, leg_features], axis=1)
columns = sum([['Name', 'Generation', 'Gen_Label'],gen_feature_labels,
              ['Legendary', 'Lgnd_Label'],leg_feature_labels], [])
poke_df_ohe[columns].iloc[4:10]

# Dummy Coding Scheme

'''
Another way of one hot encoding the categorical data is by dummy encoding.
'''

gen_dummy_features = pd.get_dummies(poke_df['Generation'], drop_first=True)
pd.concat([poke_df[['Name', 'Generation']], gen_dummy_features], axis=1).iloc[4:10]



# Feature Hashing scheme

unique_genres = np.unique(vg_df[['Genre']])
print("Total game genres:", len(unique_genres))
print(unique_genres)

from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=6, input_type='string')
hashed_features = fh.fit_transform(vg_df['Genre'])
hashed_features = hashed_features.toarray()
pd.concat([vg_df[['Name', 'Genre']], pd.DataFrame(hashed_features)], axis=1).iloc[1:7]

'''

Name	Genre	0	1	2	3	4	5
1	Super Mario Bros.	Platform	0.0	2.0	2.0	-1.0	1.0	0.0
2	Mario Kart Wii	Racing	-1.0	0.0	0.0	0.0	0.0	-1.0
3	Wii Sports Resort	Sports	-2.0	2.0	0.0	-2.0	0.0	0.0
4	Pokemon Red/Pokemon Blue	Role-Playing	-1.0	1.0	2.0	0.0	1.0	-1.0
5	Tetris	Puzzle	0.0	1.0	1.0	-2.0	1.0	-1.0
6	New Super Mario Bros.	Platform	0.0	2.0	2.0	-1.0	1.0	0.0
'''



