# number_plate-character-identifing
Text detection in Indian number plate
# Deep learning models for classify the alphabets and number
Create a seperate datasets for number and alphabets letters under following format
  Alphabets
       -Train
          -A
          -B
          -.
          -.
          -Z
       -Test
          -A
          -B
          -.
          -.
          -Z
   for the alphabet dataset check my rep
   
   Number
        -Train
             -1
             -2
             -.
             -.
             -0
        -Test
             -1
             -2
             -.
             -.
             -0
Train the dataset using the cnn.

# extract number_plate using vision api and split the characters in the plate
split the each character in the number_plate and classify the img. Then insert the predicted values in a separte list for number and text. finally concatenate the two list and we get the characters as text.

# why i am choosing this apporach 
Because sometimes the model can able to predict 5 as S and 8 as 3 and B etc. on my approach we can train and predict the letters as well as  numbers then concatenate into single formate. 
