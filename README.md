# Music Recommendation Based on Genre Recognition

In this project we extended the work from [jsalbert](https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning) and adapted it to use genre recognition for music recommendation.
The model takes as an input the spectogram of music frames and analyzes the image using a Convolutional Neural Network (CNN) plus a Recurrent Neural Network (RNN). The output of the system is a vector of predicted genres for the song.

Using this genre recognition model, the user is able to create two different music lists - ideally one with their favorite music and another one with unknown songs from different genres.
The model will take the first list **list_example.txt** and iterate through all of the songs contained in it, predicting the genres for each of those. Based on these predictions, the model will determine which genre is more likely to be the user's favorite.
Then it will take the second list **list_recommendation.txt** and iterate again, finding the songs that belong to that genre and simulating a recommendation engine (a very basic example of how recommendation on music platforms works).

In order to analyze the signal, the model needs to apply some sound processing techniques. Generating the spectrum of the given signal and creating an image that will work as the input for the neural network.
These spectograms will be divided into frames and then they will be evaluated to classify the signals by genres, and furthermore analyze these results to suggest other songs that the user could like based on previous recognitions.

The model was trained using song samples from 10 different music genres:
1) Blues
2) Classical
3) Country
4) Disco
5) Hip hop
6) Jazz
7) Metal
8) Pop
9) Reggae
10) Rock

- Poster

### How to test it

You need to install the following [requirements](https://github.com/MrManu96/Music-Recommendation-System/blob/main/requirements.txt). We suggest you create a virtual environment and install the requirements by using:
**pip install -r requirements.txt**.

After you install all the requirements and download our project, fill the music folder with songs you like and fill list_example.txt with their names.
Also fill the playlist folder with songs you don't know and fill list_recommendation.txt with their names too.

Run using **python quick_test.py**
