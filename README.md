# Feature_Engineering-
Experimenting on features with various types of data 

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

