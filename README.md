# Movie Recommendation System Using RBM

## Project Information

**Course:** Artificial Neural Network & Deep Learning  
---

# Introduction

This project is a Movie Recommendation System developed using Python, TensorFlow, and Machine Learning techniques. The main purpose of the system is to recommend movies to users based on their interests and watching behavior.

Modern platforms like Netflix, YouTube, and Amazon use recommendation systems to suggest content to users. Similarly, this system recommends movies that users may like by learning their preferences.

The project uses the MovieLens Dataset and TMDB API. A machine learning model called **Restricted Boltzmann Machine (RBM)** is used to learn user behavior and generate recommendations.

---

# Objectives

- Build a movie recommendation system  
- Recommend movies based on user interests  
- Use machine learning for prediction  
- Learn user watching patterns  
- Show personalized movie recommendations  
- Improve user experience  

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow | Machine Learning Model |
| Pandas | Data Handling |
| NumPy | Mathematical Operations |
| Matplotlib | Graph Plotting |
| MovieLens Dataset | User Ratings Data |
| TMDB API | Movie Posters and Details |

---

# Dataset Information

## 1. MovieLens Dataset

The MovieLens dataset contains:

- User IDs  
- Movie IDs  
- Movie Ratings  
- Movie Titles  
- Genres  

Ratings are provided by users from **1 to 5**.

## 2. TMDB API

The TMDB API is used to fetch:

- Movie Posters  
- Movie Overview  
- Genres  
- Popular Movies  

Around **500 popular movies** were fetched using the API.

---

# User Activity Used

The recommendation system also uses user activity data:

- Clicked Movies  
- Liked Movies  
- Watch Later Movies  

These activities help the model understand user interests more accurately.

### Example

- If a user frequently clicks action movies, the system learns that the user likes action content.
- Movies added to the watch later list are treated as user interest.
- Liked movies improve recommendation quality.

---

# Restricted Boltzmann Machine (RBM)

RBM is a machine learning algorithm commonly used for recommendation systems.

It contains two layers:

- Visible Layer  
- Hidden Layer  

The visible layer stores movie ratings, while the hidden layer learns user interests and patterns.

## Model Configuration

- Hidden Units: 50  
- Learning Rate: 0.001  
- Epochs: 50  

---

# Model Training

The model is trained using user movie ratings.

During training:

- The RBM learns user preferences  
- Error decreases after each epoch  
- Recommendations become more accurate  

---

# Recommendation Process

After training, the system predicts movies for users using the following steps:

1. Select a user  
2. Pass user data into RBM  
3. Predict unseen movies  
4. Sort movies by score  
5. Recommend top movies  

Movies with the highest recommendation scores are suggested to the user.
<img width="657" height="314" alt="Picture2" src="https://github.com/user-attachments/assets/88d83f86-8af3-4de5-bdf3-31f838ffc704" />
<img width="627" height="259" alt="Picture3" src="https://github.com/user-attachments/assets/15bed678-8e3e-45a8-b827-1f7f1ef63bce" />

---

# User Interest Monitoring

The system dynamically updates user interests based on activity.

| Category | Starting Weight | Decay | Minimum Weight | Importance |
|---|---|---|---|---|
| Recent | 1.5 | -0.1 per movie | 0.3 | High |
| Liked | 2.0 | -0.15 per movie | 0.4 | Strongest |
| Watch Later | 1.2 | -0.1 per movie | 0.2 | Medium |
| Seen Movies | +0.5 fixed | None | 0.5 | Small Boost |

### Cases

#### Case 1
User initially likes:
- Animation
- Family Movies

#### Case 2
User interest changes to:
- Horror
- Comedy Movies

The system adapts recommendations based on changing user behavior.

---

# Advantages of the Project

- Personalized movie recommendations  
- Automatically learns user interests  
- Improves user experience  
- Uses machine learning techniques  
- Supports liked and watch later movies  

---

# Conclusion

This project successfully developed a Movie Recommendation System using RBM and TensorFlow.

The recommendation model uses:

- User Ratings  
- Clicked Movies  
- Liked Movies  
- Watch Later Movies  

The system learns user behavior and recommends movies users may enjoy.

TMDB API improves the project by providing movie posters and additional movie details.

This project demonstrates how machine learning can be used in real-world recommendation systems like Netflix and YouTube.
