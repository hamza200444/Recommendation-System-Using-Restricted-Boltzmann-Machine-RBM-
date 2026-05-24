import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the movies dataset and also pass header=None since files don't contain any headers
movies_df = pd.read_csv('ml-1m\movies.dat', sep='::', header=None, engine='python',   encoding='latin1')
print(movies_df.head())

# Load the ratings dataset
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python',     encoding='latin1')
print(ratings_df.head())

# Lets rename our columns in these data frames so we can convey their data better
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Verify the changes done to the dataframes
print(movies_df.head())
print(ratings_df.head())

# Data Correction and Formatting
print('The Number of Movies in Dataset', len(movies_df))

"""
- Our Movie ID's vary from 1 to 3952 while we have 3883 movies. 
- Due to this, we won't be able to index movies through their ID since we would get memory indexing errors. 
- To amend we can create a column that shows the spot in our list that particular movie is in:
"""

movies_df['List Index'] = movies_df.index
print(movies_df.head())

# Merge movies_df with ratings_df by MovieID
merged_df = movies_df.merge(ratings_df, on='MovieID')

# Drop unnecessary columns
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

# Display the result
print(merged_df.head())

# Lets Group up the Users by their user ID's
user_Group = merged_df.groupby('UserID')
print(user_Group.head())

"""
Formatting the data into input for the RBM. 
Store the normalized users ratings into a list of lists called trX.
"""

# Amount of users used for training
amountOfUsedUsers = 1000

# Creating the training list
trX = []

# For each user in the group
for userID, curUser in user_Group:

    # Create a temp that stores every movie's rating
    temp = [0]*len(movies_df)

    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():

        # Divide the rating by 5 and store it
        temp[movie['List Index']] = movie['Rating']/5.0

    # Add the list of ratings into the training list
    trX.append(temp)

    # Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
print(trX)

# Setting the models Parameters
hiddenUnits = 50
visibleUnits = len(movies_df)
vb = tf.Variable(tf.zeros([visibleUnits], tf.float32))
hb = tf.Variable(tf.zeros([hiddenUnits], tf.float32))
W = tf.Variable(
    tf.random.normal(
        [visibleUnits, hiddenUnits],
        stddev=0.01
    )
)

# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random.uniform(tf.shape(_h0))))  # Gibb's Sampling

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random.uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

""" Set RBM Training Parameters """

# Learning rate
alpha = 1.0

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

# Create methods to update the weights and biases
learning_rate = 0.001

update_w = tf.assign(W, W + learning_rate * CD)
update_vb = tf.assign(vb, vb + learning_rate * tf.reduce_mean(v0 - _v1, 0))
update_hb = tf.assign(hb, hb + learning_rate * tf.reduce_mean(h0 - h1, 0))

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)

""" Initialize our Variables with Zeroes using Numpy Library """

# Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch
epochs = 50
batchsize = 100
errors = []
for i in range(epochs):

    for start in range(0, len(trX), batchsize):

        end = start + batchsize


        batch = trX[start:end]

        # Update parameters
        sess.run(update_w, feed_dict={v0: batch})
        sess.run(update_vb, feed_dict={v0: batch})
        sess.run(update_hb, feed_dict={v0: batch})

    # Get latest values
    cur_w = sess.run(W)
    cur_vb = sess.run(vb)
    cur_hb = sess.run(hb)

    # Calculate error
    error = sess.run(err_sum, feed_dict={v0: trX})

    errors.append(error)

    print("Epoch:", i, "Error:", error)
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

"""
Recommendation System :-

- We can now predict movies that an arbitrarily selected user might like. 
- This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the 
  input. 
- The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched 
  based on the preferences of the users that the RBM was trained on.
"""

# Select the input User
inputUser = [trX[50]]

# Feeding in the User and Reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser})
rec = sess.run(vv1, feed_dict={v0: inputUser})

# List the 20 most recommended movies for our mock user by sorting it by their scores given by our model.
scored_movies_df_50 = movies_df
scored_movies_df_50["Recommendation Score"] = rec[0]
print(scored_movies_df_50.sort_values(["Recommendation Score"], ascending=False).head(20))

""" Recommend User what movies he has not watched yet """

# Find the mock user's UserID from the data
print(merged_df.iloc[50])  # Result you get is UserID 150

# Find all movies the mock user has watched before
movies_df_50 = merged_df[merged_df['UserID'] == 150]
print(movies_df_50.head())

""" Merge all movies that our mock users has watched with predicted scores based on his historical data: """

# Merging movies_df with ratings_df by MovieID
merged_df_50 = scored_movies_df_50.merge(movies_df_50, on='MovieID', how='outer')

# Dropping unnecessary columns
merged_df_50 = merged_df_50.drop('List Index_y', axis=1).drop('UserID', axis=1)

# Sort and take a look at first 20 rows
print(merged_df_50.sort_values(['Recommendation Score'], ascending=False).head(20))

""" There are some movies the user has not watched and has high score based on our model. So, we can recommend them. """
# Save trained RBM model
final_w = sess.run(W)
final_vb = sess.run(vb)
final_hb = sess.run(hb)
movies_df[['MovieID', 'List Index', 'Title', 'Genres']].to_csv(
    "movie_mapping.csv",
    index=False
)
np.save("rbm_weights.npy", final_w)
np.save("rbm_visible_bias.npy", final_vb)
np.save("rbm_hidden_bias.npy", final_hb)
np.save("training_matrix.npy", np.array(trX))
print("RBM model saved successfully!")

print("RBM model saved successfully!")