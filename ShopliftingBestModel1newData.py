#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json

# Kaggle credentials
kaggle_credentials = {
    "username": "ayushichouhan1973",
    "key": "b545d8e88cd4cd3c22894044fc1e8b06"
}

# Create the .kaggle directory if it doesn't exist
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Write the credentials to kaggle.json
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json_path, "w") as f:
    json.dump(kaggle_credentials, f)

# Set permissions to secure the file
os.chmod(kaggle_json_path, 0o600)

print("Kaggle credentials set up successfully!")


# In[2]:


get_ipython().system('pip install kagglehub')


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("mateohervas/dcsass-dataset")

print("Path to dataset files:", path)


# In[4]:


import os

# Path to the main directory
dataset_path = '/home/ec2-user/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1'

# List contents of the directories
folders = os.listdir(dataset_path)
print("Folders inside the dataset:", folders)

# Now list files in each folder
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  # Check if it's a folder
        files_in_folder = os.listdir(folder_path)
        print(f"Files in '{folder}':", files_in_folder)


# In[5]:


import os

# Path to the 'DCSASS Dataset' folder
dataset_path = '/home/ec2-user/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1/DCSASS Dataset'

# Path to the 'Labels' folder inside it
labels_folder_path = os.path.join(dataset_path, 'Labels')

# List the files in the 'Labels' folder
if os.path.isdir(labels_folder_path):
    label_files = os.listdir(labels_folder_path)
    print(f"Files inside 'Labels' folder:", label_files)


# In[6]:


import pandas as pd
import os

# Path to the 'Labels' folder
labels_folder_path = '/home/ec2-user/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1/DCSASS Dataset/Labels'

# Path to 'Shoplifting.csv' inside the 'Labels' folder
shoplifting_file_path = os.path.join(labels_folder_path, 'Shoplifting.csv')

# Check if the file exists
if os.path.exists(shoplifting_file_path):
    # Read the CSV file into a DataFrame
    shoplifting_df = pd.read_csv(shoplifting_file_path)
    print(shoplifting_df.head())  # Display the first few rows of the Shoplifting dataset
else:
    print("Shoplifting.csv file not found!")


# In[7]:


# Organising data
folder_path = "Shoplifting"
folder_names = ["0","1"]

if not os.path.exists(folder_path):
    os.mkdir(folder_path)
else:
    print("Folder already exists...")

for folder_name in folder_names:
    if not os.path.exists(os.path.join(folder_path,folder_name)):
        os.mkdir(os.path.join(folder_path,folder_name))
    else:
        print("folder already exists...")

dataset = pd.read_csv(shoplifting_file_path)

# the datasets column names are also the part of the dataset
# first we will append that data into our dataframe and then rename the columns

data = [dataset.columns[0],dataset.columns[1],int(dataset.columns[2])]

dataset.loc[len(dataset)] = data

dataset.rename(columns={"Shoplifting001_x264_0":"clipname","Shoplifting":"Shoplifting","0":"Action"},inplace=True)


# In[8]:


ROOT_DIR = r"/home/ec2-user/.cache/kagglehub/datasets/mateohervas/dcsass-dataset/versions/1/DCSASS Dataset/Shoplifting"


# In[9]:


import os


base_path = "/home/ec2-user/Shoplifting"

# Create the directories
os.makedirs(os.path.join(base_path, "new_0"), exist_ok=True)
os.makedirs(os.path.join(base_path, "1"), exist_ok=True)

# Check if directories were created
print("Directories created:", os.listdir(base_path))


# In[10]:


DESTINATION_ROOT = r"/home/ec2-user/Shoplifting"
DESTINATION_PATH_0 = r"/home/ec2-user/Shoplifting/new_0"
DESTINATION_PATH_1 = r"/home/ec2-user/Shoplifting/1"


# In[11]:


import shutil
import os
directories = os.listdir(ROOT_DIR)

for dir in directories:
    for d in os.listdir(os.path.join(ROOT_DIR,dir)):
            row = dataset.loc[dataset['clipname'] == d[:-4]]
            if row['Action'].iloc[0] == 0:
                shutil.copy(os.path.join(ROOT_DIR,dir,d),os.path.join(DESTINATION_PATH_0,d))
            else:
                shutil.copy(os.path.join(ROOT_DIR,dir,d),os.path.join(DESTINATION_PATH_1,d))


# In[12]:


import os
import random

# Define the destination directory
destination_directory = r"/home/ec2-user/Shoplifting/new_0"

# List all video files in the destination directory
video_files = [f for f in os.listdir(destination_directory) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Ensure there are more than 160 videos to delete the excess
if len(video_files) > 160:
    # Randomly select 160 files to keep
    files_to_keep = random.sample(video_files, 160)

    # Determine which files to delete (all except the ones selected to keep)
    files_to_delete = set(video_files) - set(files_to_keep)

    # Delete the files that are not in the files_to_keep list
    for video in files_to_delete:
        video_path = os.path.join(destination_directory, video)
        os.remove(video_path)  # Delete the file
        print(f"Deleted: {video}")

    print(f"Successfully deleted excess files. {len(files_to_delete)} files were removed.")
else:
    print(f"Error: There are not more than 160 videos. Found {len(video_files)} videos.")


# In[13]:


print("Count of number of video clips with 0 and 1 :-")
print(dataset['Action'].value_counts())
print("---------------------------------------------------------------------------------------")
print("Video clips present in 0 and 1 :-")
print("no shoplifting count : ",len(os.listdir(DESTINATION_PATH_0)))
print("shoplifting count : ",len(os.listdir(DESTINATION_PATH_1)))


# In[14]:


import numpy as np
import tensorflow as tf

seed_constant = 27
np.random.seed(seed_constant)
tf.random.set_seed(seed_constant)


# In[16]:


dim = 128  # Increase for better quality
IMAGE_HEIGHT = dim
IMAGE_WIDTH = dim

SEQUENCE_LENGTH = 20  # Reduce if GPU resources are tight
CLASSES_LIST = ["new_0", "1"]


# In[17]:


import cv2 
import os
def frame_extraction(video_path):
    frame_list = []

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frame_list  # Return empty list on failure

    # Get the total frame count
    video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count < SEQUENCE_LENGTH:
        print(f"Warning: Video {video_path} has fewer frames ({video_frame_count}) than SEQUENCE_LENGTH ({SEQUENCE_LENGTH}).")
        return frame_list  # Return empty list if not enough frames

    # Calculate the frame skip window
    skip_frame_window = max(video_frame_count // SEQUENCE_LENGTH, 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        # Calculate the exact frame position
        frame_position = frame_counter * skip_frame_window
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

        success, frame = video_capture.read()
        if not success:
            print(f"Warning: Frame {frame_counter} could not be read from {video_path}.")
            break

        # Resize the frame to fixed dimensions
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize the frame (vectorized)
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        # Append to the frame list
        frame_list.append(normalized_frame)

    # Release the video capture object
    video_capture.release()

    # Check if the required number of frames was collected
    if len(frame_list) != SEQUENCE_LENGTH:
        print(f"Error: Only {len(frame_list)} frames could be extracted from {video_path}.")
        return []

    return frame_list


# In[20]:


SEQUENCE_LENGTH = 20  # Updated to match the new parameter
CLASSES_LIST = ["new_0", "1"]

def create_dataset():
    features = []
    labels = []
    video_file_paths = []

    # Iterate through each class
    for cls_index, cls in enumerate(CLASSES_LIST):
        class_dir = os.path.join(DESTINATION_ROOT, cls)
        
        if not os.path.exists(class_dir):
            print(f"Error: Directory {class_dir} does not exist.")
            continue
        
        # Get all valid video files in the directory
        file_list = [
            file for file in os.listdir(class_dir) 
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        if not file_list:
            print(f"Warning: No valid video files found in {class_dir}.")
            continue

        for file in file_list:
            video_file_path = os.path.join(class_dir, file)

            # Extract frames from the video
            frames = frame_extraction(video_file_path)
            
            # Ensure the extracted frames match the required sequence length
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(cls_index)
                video_file_paths.append(video_file_path)
            else:
                print(f"Skipping {file}: Insufficient frames ({len(frames)}/{SEQUENCE_LENGTH}).")

    # Convert lists to numpy arrays
    if features:
        features = np.asarray(features, dtype=np.float32)
    else:
        features = np.empty((0, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        print("Warning: No valid videos processed.")

    labels = np.array(labels, dtype=np.int32)

    return features, labels, video_file_paths


# In[22]:


features, labels, video_paths = create_dataset()
print(f"Features Shape: {features.shape}")
print(f"Labels Shape: {labels.shape}")
print(f"Sample Video Paths: {video_paths[:3]}")


# In[23]:


from tensorflow.keras.utils import to_categorical

one_hot_encoded_labels = to_categorical(labels)


# In[24]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features,one_hot_encoded_labels,test_size=0.25,shuffle=True,random_state=seed_constant)


# In[25]:


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model




input_layer = Input(shape=(SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,3))

convlstm_1 = ConvLSTM2D(filters=4,kernel_size=(3,3),activation='tanh',data_format='channels_last',recurrent_dropout=0.2,return_sequences=True)(input_layer)
pool1 = MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last')(convlstm_1)
timedistributed_1 = TimeDistributed(Dropout(0.2))(pool1)

convlstm_2 = ConvLSTM2D(filters=8,kernel_size=(3,3),activation='tanh',data_format='channels_last',recurrent_dropout=0.2,return_sequences=True)(timedistributed_1)
pool2 = MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last')(convlstm_2)
timedistributed_2 = TimeDistributed(Dropout(0.2))(pool2)

convlstm_3 = ConvLSTM2D(filters=16,kernel_size=(3,3),activation='tanh',data_format='channels_last',recurrent_dropout=0.2,return_sequences=True)(timedistributed_2)
pool3 = MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last')(convlstm_3)
timedistributed_3 = TimeDistributed(Dropout(0.2))(pool3)

convlstm_4 = ConvLSTM2D(filters=32,kernel_size=(3,3),activation='tanh',data_format='channels_last',recurrent_dropout=0.2,return_sequences=True)(timedistributed_3)
pool4 = MaxPooling3D(pool_size=(1,2,2),padding='same',data_format='channels_last')(convlstm_4)

flatten = Flatten()(pool4)

output = Dense(units=len(CLASSES_LIST), activation='softmax')(flatten)



model = Model(input_layer,output)


# In[26]:


model.summary()


# In[27]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(), 
              metrics=['accuracy'])


# In[35]:


history = model.fit(x_train, 
                    y_train, 
                    epochs=30, 
                    batch_size=4, 
                    shuffle=True, 
                    validation_split=0.2)


# In[36]:


model.save("model_2.h5")


# In[37]:


loss,accuracy = model.evaluate(x_test,y_test)

print("Loss : ",loss)
print("Accuracy : ",accuracy)


# In[38]:


import matplotlib.pyplot as plt

def plot_curve(model_training_history,metric_name_1,metric_name_2,plot_name):
    
    metric1 = model_training_history.history[metric_name_1]
    metric2 = model_training_history.history[metric_name_2]
    plt.plot(metric1,color='blue',label=metric_name_1)
    plt.plot(metric2,color='red',label=metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.show()


# In[39]:


plot_curve(history,'loss','val_loss',"Total loss vs Total validation loss")


# In[32]:


from sklearn.metrics import precision_recall_fscore_support

predictions = model.predict(x_test)

# Assuming predictions are in probability form and you need to convert them to binary labels
binary_predictions = (predictions > 0.5).astype('int32')

# Calculate precision, recall, and F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, binary_predictions, average=None)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# In[33]:


from sklearn.metrics import confusion_matrix

predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)  # Assuming test_Y is one-hot encoded
conf_matrix = confusion_matrix(true_labels, predicted_labels)


# In[34]:


import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




