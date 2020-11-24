from keras import backend as K
import os
import time
import h5py
import sys
from tagger_net import MusicTaggerCRNN
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from music_tagger_cnn import MusicTaggerCNN
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, extract_melgrams
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# Parameters to set
TEST = 1

LOAD_MODEL = 0
LOAD_WEIGHTS = 1
MULTIFRAMES = 1
time_elapsed = 0

# GTZAN Dataset Tags
tags = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
tags = np.array(tags)
generos = []
recomendaciones = []

# COUNTERS FOR MUSIC RECOMENDATION
counter = [0,0,0,0,0,0,0,0,0,0]

# Paths to set
model_name = "example_model"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

test_songs_list = 'list_example.txt'

# Clear screen
os.system('clear')

# Initialize model
model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if LOAD_WEIGHTS:
    model.load_weights(weights_path+'crnn_net_gru_adam_ours_epoch_40.h5')

#model.summary()

X_test, num_frames_test= extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')

num_frames_test = np.array(num_frames_test)

t0 = time.time()

print '\n--------- Predicting ---------','\n'

results = np.zeros((X_test.shape[0], tags.shape[0]))
predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
predicted_labels_frames = np.zeros((X_test.shape[0], 1))

song_paths = open(test_songs_list, 'r').read().splitlines()

previous_numFrames = 0
n=0
for i in range(0, num_frames_test.shape[0]):
    print 'Song number #' +str(i+1)+ ': ' + song_paths[i]
    cancion = song_paths[i]

    num_frames=num_frames_test[i]
    print 'Number of 30-seconds frames: ', str(num_frames),'\n'

    results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
        X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])

    s_counter = 0
    for j in range(previous_numFrames, previous_numFrames+num_frames):
        #normalize the results
        total = results[j,:].sum()
        results[j,:]=results[j,:]/total
        print '\nPercentage of genre prediction for seconds '+ str(20+s_counter*30) + ' to ' \
            + str(20+(s_counter+1)*30) + ': '
        sort_result(tags, results[j,:].tolist())

        predicted_label_frames=predict_label(results[j,:])
        predicted_labels_frames[n]=predicted_label_frames
        s_counter += 1
        n+=1

    print '\n', 'Mean genre of the song: '
    results_song = results[previous_numFrames:previous_numFrames+num_frames]

    mean=results_song.mean(0)
    sort_result(tags, mean.tolist())

    predicted_label_mean=predict_label(mean)

    predicted_labels_mean[i]=predicted_label_mean
    print '\n','The predicted music genre for the song is', str(tags[predicted_label_mean]),'!\n'

    previous_numFrames = previous_numFrames+num_frames
    generos.append(tags[predicted_label_mean])

    colors = ['b','g','c','r','m','k','y','#ff1122','#5511ff','#44ff22']
    fig, ax = plt.subplots()
    index = np.arange(tags.shape[0])
    opacity = 1
    bar_width = 0.2
    #print mean
    #for g in range(0, tags.shape[0]):
    plt.bar(left=index, height=mean, width=bar_width, alpha=opacity, color=colors)

    plt.xlabel('Genres')
    plt.ylabel('Probability')
    plt.title(cancion[6:])
    plt.xticks(index + bar_width / 2, tags)
    plt.tight_layout()
    fig.autofmt_xdate()
    graph_title = 'Graphs/prediction_'+str(i+1)+'.png'
    plt.savefig(graph_title)

    print '***************************************************************'

for i in range(0, num_frames_test.shape[0]):
    song_name = song_paths[i]
    print '\n','The predicted music genre for song #'+str(i+1),'('+song_name[6:]+') =\t',str(generos[i])
    #print 'Probability of: '+
    if str(generos[i]) == "Blues":
        counter[0]+=1
    else:
        if str(generos[i]) == "Classical":
            counter[1]+=1
        else:
            if str(generos[i]) == "Country":
                counter[2]+=1
            else:
                if str(generos[i]) == "Disco":
                    counter[3]+=1
                else:
                    if str(generos[i]) == "Hip-Hop":
                        counter[4]+=1
                    else:
                        if str(generos[i]) == "Jazz":
                            counter[5]+=1
                        else:
                            if str(generos[i]) == "Metal":
                                counter[6]+=1
                            else:
                                if str(generos[i]) == "Pop":
                                    counter[7]+=1
                                else:
                                    if str(generos[i]) == "Reggae":
                                        counter[8]+=1
                                    else:
                                        if str(generos[i]) == "Rock":
                                            counter[9]+=1

print '\n***************************************************************\n'

recommend = raw_input('Would you like some music recommendation from me? (Y = Yes / N = No) ')
while recommend != 'Y' and recommend != 'N' and recommend != 'y' and recommend != 'n':
    print '\nWrong option! Please try again.'
    recommend = raw_input('Would you like some music recommendation from me? (Y = Yes / N = No) ')

if recommend == 'Y' or recommend == 'y':
    highest = counter[0]
    indice = 0
    for p in range(1,9):
        if counter[p] == highest:
            highest = highest
            indice = indice
        else:
            if counter[p] > highest:
                highest = counter[p]
                indice = p
    # Clear screen
    os.system('clear')

    test_songs_list = 'list_recommendation.txt'

    # Initialize model
    model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if LOAD_WEIGHTS:
        model.load_weights(weights_path+'crnn_net_gru_adam_ours_epoch_40.h5')

    #model.summary()

    X_test, num_frames_test= extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')

    num_frames_test = np.array(num_frames_test)

    t0 = time.time()

    print '\nLooking for Recommendations..\n'

    results = np.zeros((X_test.shape[0], tags.shape[0]))
    predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
    predicted_labels_frames = np.zeros((X_test.shape[0], 1))

    song_paths = open(test_songs_list, 'r').read().splitlines()

    previous_numFrames = 0
    n=0
    for i in range(0, num_frames_test.shape[0]):
        #print 'Song number #' +str(i+1)#+ ': ' + song_paths[i]

        num_frames=num_frames_test[i]
        #print 'Number of 30-seconds frames: ', str(num_frames),'\n'

        results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
            X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])

        s_counter = 0
        for j in range(previous_numFrames, previous_numFrames+num_frames):
            #normalize the results
            total = results[j,:].sum()
            results[j,:]=results[j,:]/total
            #print '\nPercentage of genre prediction for seconds '+ str(20+s_counter*30) + ' to ' \
            #    + str(20+(s_counter+1)*30) + ': '
            sort_result(tags, results[j,:].tolist())

            predicted_label_frames=predict_label(results[j,:])
            predicted_labels_frames[n]=predicted_label_frames
            s_counter += 1
            n+=1

        #print '\n', 'Mean genre of the song: '
        results_song = results[previous_numFrames:previous_numFrames+num_frames]

        mean=results_song.mean(0)
        sort_result(tags, mean.tolist())

        predicted_label_mean=predict_label(mean)

        predicted_labels_mean[i]=predicted_label_mean
        #print '\n','The predicted music genre for the song is', str(tags[predicted_label_mean]),'!\n'
        if predicted_label_mean == indice:
            recomendaciones.append(song_paths[i])

        previous_numFrames = previous_numFrames+num_frames
        #generos.append(tags[predicted_label_mean])

        print '\nLooking for Recommendations..'

    print '\nBased on previous music recognitions, your favorite genre is: '+str(tags[indice])
    print 'With '+str(highest)+' previous predictions.\n'
    print '\nI think you could like these songs. Give them a shot!\n'
    #print recomendaciones
    longitud = len(recomendaciones)
    for i in range(0,longitud):
        song_name = recomendaciones[i]
        print '* '+song_name[9:]+'\n'
        #print 'Its working'


print '\n************* END OF PROCESSING. Have a good day! *************\n\n'



