# number_plate-character-identifying
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
Because sometimes the model can able to predict 5 as S and 8 as 3 and B etc. on my approach we can train and predict the letters as well as  numbers seprately 
then concatenate into single format. 


**Output**
1/1 [==============================] - 0s 195ms/step
1/1 [==============================] - 0s 93ms/step
1/1 [==============================] - 0s 41ms/step
1/1 [==============================] - 0s 40ms/step
1/1 [==============================] - 0s 198ms/step
1/1 [==============================] - 0s 58ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 29ms/step
1/1 [==============================] - 0s 32ms/step
1/1 [==============================] - 0s 31ms/step
MB01AE8017
