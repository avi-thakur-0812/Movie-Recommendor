import numpy as nm 
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path  
import glob
import re
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

# Not setting index_col=False would set the first column as an index, but we actually need our first clumn as 'Title'
netflix_data = pd.read_csv('test_data.csv',encoding='latin-1', index_col = False)

def clean_text(text):
    if type(text)==str:
        # Convert to lowercase
        text = text.lower()
    
        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text
text_columns = netflix_data.columns[netflix_data.dtypes == 'object']
for col in text_columns:
    netflix_data[col] = netflix_data[col].apply(clean_text)
netflix_data.to_csv("cleaned_data.csv", index=False)


# Making a pandas dataframe
cleaned_data = pd.read_csv('cleaned_data.csv', )
df = pd.read_table("cleaned_data.csv", delimiter =", ")


# Our columns are singular, ie they aren't separated into 'Title' 'Genre' 'Tags' etc, it is all a singular string
df = df['Title,Genre,Tags,Languages,Director,Writer,Actors,View Rating,IMDb Score,Awards Received,Awards Nominated For,Boxoffice,Netflix Link,Summary,IMDb Votes,Image'].str.split(',', expand=True)

# Assign column names to the new dataframe
df.columns = ['Title', 'Genre', 'Tags', 'Languages', 'Director', 'Writer', 'Actors', 'View Rating', 'IMDb Score', 'Awards Received', 'Awards Nominated For', 'Boxoffice', 'Netflix Link', 'Summary', 'IMDb Votes', 'Image']



# Use TF-IDF vectorization on the first 6 columns to convert text data to numbers
text_data1 = df['Genre'].tolist()
vectorizer1 = TfidfVectorizer()
tfidf_matrix1 = vectorizer1.fit_transform(text_data1)
dense_tfidf_matrix1 = tfidf_matrix1.todense()

text_data2 = df['Tags'].tolist()
vectorizer2 = TfidfVectorizer()
tfidf_matrix2 = vectorizer2.fit_transform(text_data2)
dense_tfidf_matrix2 = tfidf_matrix2.todense()

text_data3 = df['Languages'].tolist()
vectorizer3 = TfidfVectorizer()
tfidf_matrix3 = vectorizer3.fit_transform(text_data3)
dense_tfidf_matrix3 = tfidf_matrix3.todense()

text_data4 = df['Director'].tolist()
vectorizer4 = TfidfVectorizer()
tfidf_matrix4 = vectorizer4.fit_transform(text_data4)
dense_tfidf_matrix4 = tfidf_matrix4.todense()

text_data5 = df['Writer'].tolist()
vectorizer5 = TfidfVectorizer()
tfidf_matrix5 = vectorizer5.fit_transform(text_data5)
dense_tfidf_matrix5 = tfidf_matrix5.todense()

text_data6 = df['Actors'].tolist()
vectorizer6 = TfidfVectorizer()
tfidf_matrix6 = vectorizer6.fit_transform(text_data6)
dense_tfidf_matrix6 = tfidf_matrix6.todense()



# combine all 6 matrices to a singular matrix 
combined_matrix = hstack((tfidf_matrix1, tfidf_matrix2, tfidf_matrix3, tfidf_matrix4, tfidf_matrix5, tfidf_matrix6))

# applying KNN
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(combined_matrix)

'''
combined_matrix = combined_matrix.todense()
combined_matrix = nm.asarray(combined_matrix)
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(combined_matrix)
'''


def get_recommendations(movie_id, X, model):
    # Get the row corresponding to the movie of interest
    movie = X[movie_id, :]
    
    # Get the indices and distances of the nearest neighbors
    distances, indices = model.kneighbors(movie.reshape(1, -1))
    
    # Return the movie titles corresponding to the nearest neighbors
    return df['Title'].iloc[indices[0]]

movie_id = 1810
print("Recommendations for movie:", df['Title'].iloc[movie_id])
print(get_recommendations(movie_id, combined_matrix, neigh))
