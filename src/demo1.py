import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.utils import shuffle

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 50) 

newdata = pd.read_csv('annotations_final.csv', sep="\t")
newdata.head(5)
newdata.info()
newdata.columns
newdata[["clip_id", "mp3_path"]]

# Extract clip_id and mp3_path as a matrix.
clip_id, mp3_path = newdata[["clip_id", "mp3_path"]].as_matrix()[:,0], newdata[["clip_id", "mp3_path"]].as_matrix()[:,1]

synonyms = [
    ['beat', 'beats'],
    ['chant', 'chanting'],
    ['choir', 'choral'],
    ['classical', 'clasical', 'classic'],
    ['drum', 'drums'],
    ['electro', 'electronic', 'electronica', 'electric'],
    ['fast', 'fast beat', 'quick'],
    ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
    ['flute', 'flutes'],
    ['guitar', 'guitars'],
    ['hard', 'hard rock'],
    ['harpsichord', 'harpsicord'],
    ['heavy', 'heavy metal', 'metal'],
    ['horn', 'horns'],
    ['india', 'indian'],
    ['jazz', 'jazzy'],
    ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
    ['no beat', 'no drums'],
    ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
    ['opera', 'operatic'],
    ['orchestra', 'orchestral'],
    ['quiet', 'silence'],
    ['singer', 'singing'],
    ['space', 'spacey'],
    ['string', 'strings'],
    ['synth', 'synthesizer'],
    ['violin', 'violins'],
    ['vocal', 'vocals', 'voice', 'voices'],
    ['strange', 'weird']
    ]

# Merge the synonyms and drop all other columns than the first one.
# Merge 'beat', 'beats' and save it to 'beat'.
for synonym_list in synonyms:
    newdata[synonym_list[0]] = newdata[synonym_list].max(axis=1)
    newdata.drop(synonym_list[1:], axis=1, inplace=True)

newdata.info()
newdata.head()

newdata.drop('mp3_path', axis=1, inplace=True)
data = newdata.sum(axis=0)
data
data.sort_values(axis=0, inplace=True)
topindex, topvalues = list(data.index[84:]), data.values[84:]
del(topindex[-1])
topvalues = np.delete(topvalues, -1)
topindex
topvalues
rem_cols = data.index[:84]
len(rem_cols)
newdata.drop(rem_cols, axis=1, inplace=True)
newdata.info()
backup_newdata = newdata
newdata = shuffle(newdata)
newdata.reset_index(drop=True)
newdata.info()
final_columns_names = list(newdata.columns)
del(final_columns_names[0])
final_columns_names

# Here, binary 0's and 1's from each column is changed to 'False' and 'True' by using '==' operator on the dataframe.
final_matrix = pd.concat([newdata['clip_id'], newdata[final_columns_names]==1], axis=1)

# Rename all the mp3 files to their clip_id and save it into one folder named 'dataset_clip_id_mp3' in the same folder.
root = os.getcwd()
os.mkdir(root + "/dataset_clip_id_mp3/")

# Iterate over the mp3 files, rename them to the clip_id and save it to another folder.
for id in range(25863):
    # print(clip_id[id], mp3_path[id])
    src = root + "/" + mp3_path[id]
    dest = root + "/dataset_clip_id_mp3/" + str(clip_id[id]) + ".mp3"
    shutil.copy2(src,dest)
    # print(src, dest)

# Rename all the mp3 files to their clip_id and save it into one folder named 'dataset_clip_id_mp3' in the same folder.
root = os.getcwd()
os.mkdir(root + "/dataset_clip_id_mp3/")

# Iterate over the mp3 files, rename them to the clip_id and save it to another folder.
for id in range(25863):
    # print(clip_id[id], mp3_path[id])
    src = root + "/" + mp3_path[id]
    dest = root + "/dataset_clip_id_mp3/" + str(clip_id[id]) + ".mp3"
    shutil.copy2(src,dest)
    # print(src, dest)

# Convert all the mp3 files into their corresponding mel-spectrograms (melgrams).
# Audio preprocessing function
def compute_melgram(audio_path):
    ''' 
    Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), 
    where 96 == #mel-bins and 1366 == #time frame
    
    parameters
    ----------
    audio_path: path for the audio file. 
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # make it 1366 frame
    src, _ = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    ret = librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS
            )**2, 
        ref=1.0
        )
    ret = ret[np.newaxis, np.newaxis, :]
    return ret

# Get the absolute path of all audio files and save it to audio_paths array
audio_paths = []
files_that_dont_work=[]
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
root = os.getcwd()
os.chdir(root + '/dataset_clip_id_mp3/')
for audio_path in os.listdir('.'):
    # audio_paths.append(os.path.abspath(fname))
    if os.path.isfile(root + '/dataset_clip_id_melgram/' + str(os.path.splitext(audio_path)[0]) + '.npy'):
        # print("exists")
        continue
    else:
        if str(os.path.splitext(audio_path)[1]) == ".mp3":
            try:
                melgram = compute_melgram(os.path.abspath(audio_path))
                dest = root + '/dataset_clip_id_melgram/' + str(os.path.splitext(audio_path)[0])
                np.save(dest, melgram)
            except EOFError:
                files_that_dont_work.append(audio_path)
                continue
                
mp3_available = []
for mp3 in os.listdir('/home/cc/notebooks/MusicProject/MagnaTagATune/dataset_clip_id_mp3/'):
     mp3_available.append(int(os.path.splitext(mp3)[0]))

melgram_available = []
for melgram in os.listdir('/home/cc/notebooks/MusicProject/MagnaTagATune/dataset_clip_id_melgram/'):
     melgram_available.append(int(os.path.splitext(melgram)[0]))

new_clip_id = final_matrix['clip_id']

# Let us see which all files have not been converted into melspectrograms.
set(list(new_clip_id)).difference(melgram_available)

# Saw that these clips were extra 35644, 55753, 57881. Removing them.
final_matrix = final_matrix[final_matrix['clip_id'] != 35644]
final_matrix = final_matrix[final_matrix['clip_id'] != 55753]
final_matrix = final_matrix[final_matrix['clip_id'] != 57881]

final_matrix.info()
final_matrix.to_pickle('final_Dataframe.pkl')
training_with_clip = final_matrix[:19773]
validation_with_clip = final_matrix[19773:21294]
testing_with_clip = final_matrix[21294:]
training_with_clip
testing_with_clip
validation_with_clip

# Extract the corresponding clip_id's
training_clip_id = training_with_clip['clip_id'].values
validation_clip_id = validation_with_clip['clip_id'].values
testing_clip_id = testing_with_clip['clip_id'].values
training_clip_id

os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/final_dataset/')
np.save('train_y.npy', training_with_clip[final_columns_names].values)
np.save('valid_y.npy', validation_with_clip[final_columns_names].values)
np.save('test_y.npy', testing_with_clip[final_columns_names].values)

np.savetxt('train_x_clip_id.txt', training_with_clip['clip_id'].values, fmt='%i')
np.savetxt('test_x_clip_id.txt', testing_with_clip['clip_id'].values, fmt='%i')
np.savetxt('valid_x_clip_id.txt', validation_with_clip['clip_id'].values, fmt='%i')

# Now to combine the melgrams according to the clip_id. 
# Variable to store melgrams.
train_x = np.zeros((0, 1, 96, 1366))
test_x = np.zeros((0, 1, 96, 1366))
valid_x = np.zeros((0, 1, 96, 1366))

root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,valid_clip in enumerate(list(validation_clip_id)):
    if os.path.isfile(str(valid_clip) + '.npy'):
        # print(i, valid_clip)
        melgram = np.load(str(valid_clip) + '.npy')
        valid_x = np.concatenate((valid_x, melgram), axis=0)

os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('valid_x.npy', valid_x)
print("Validation file created")

root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,test_clip in enumerate(list(testing_clip_id)):
    if os.path.isfile(str(test_clip) + '.npy'):
        # print(i, test_clip)
        melgram = np.load(str(test_clip) + '.npy')
        test_x = np.concatenate((test_x, melgram), axis=0)
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('test_x.npy', test_x)
print("Testing file created")

root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,train_clip in enumerate(list(training_clip_id)):
    # if os.path.isfile(str(train_clip) + '.npy'):
        # print(i, train_clip)
    melgram = compute_melgram(str(train_clip) + '.mp3')
    # melgram = np.load(str(train_clip) + '.npy')
    train_x = np.concatenate((train_x, melgram), axis=0)
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('train_x.npy', train_x)
print("Training file created.")