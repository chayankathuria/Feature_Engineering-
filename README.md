# Feature_Engineering-
Experimenting on features with various types of data 


# Numerical Data
-----------------------------------------------------
## Binarization

Binarization is nothing but converting numerical data into binary values of 0s and 1s based on
a threshold value. If the value is more than the threshold, it is assigned 1, else 0.

This can be done by simply putting a condition on the column and assigining 1 to values more than a threshold.
Binarizer in the preprocessing module does this task by using the threshold parameter

-----------------------------------------------------
## Rounding

Rounding is simply rounding off the values to the nearest 10s or 100s for ease of calculation. This might depend on your problem.
It might not be advisable to round off your data each time. Certainly not!

-----------------------------------------------------

## Interactions

The polynomial regression adds interaction terms like x.y or 2.x.y etc. This interaction term can be a better 
feature as it might have better correaltion with the target

-----------------------------------------------------
## Transformations 

Many times the data we envounter is skewed. Hence we need to apply some transformations in order to make it normally distributed.
Two such transformations are Log transformations and Box-Cox transformation.

Log transformation apply the log function to the data.

-----------------------------------------------------
## Box-cox Transformations

A Box Cox transformation is a way to transform non-normal dependent variables into a normal shape. 
Normality is an important assumption for many statistical techniques; 
if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.

At the core of the Box Cox transformation is an exponent, lambda (λ), which varies from -5 to 5. All values of λ are considered and the optimal value for your data is selected; The “optimal value” is the one which results in the best approximation of a normal distribution curve.

------------
------------

# Categorical Data
## Nominal Data

Nominal data consists of names as labels and we need to transform these labels into numbers so that the algortithm can understand.
One such technique is Label Encoding which converts labels into integers starting from 1.
-------
## Ordinal Data

Ordinal Data is the typ of categorical data where the categories cannot be quantified but an order can be defined stating which is the least and which one is the highest. For ex. Unhappy < OK < Happy < Verry Happy

Using map function is just another way to label encode categorical features, ordinal or nominal. But it is recommended only when the number
of features is less because it requires hard-coding

## One Hot Encoding and Label Encoding

Both these techniques are used to convert categorical data into numerical.
One hot encoding uses a One vs. All technique. And adds n columns for n classes. 

Label encoding denotes an integer to every class starting from 1.
