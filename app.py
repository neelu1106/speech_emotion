import gradio as gr
import librosa
import numpy as np  
import tensorflow as tf
import html

from huggingface_hub import hf_hub_download
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

#model_id = "ZionC27/EMO_20_82"
#model_d = os.path.join(model_CLSTM_20_89.h5)

from keras.initializers import Orthogonal
model = tf.keras.models.load_model("model_CLSTM_20_89.h5", custom_objects={'Orthogonal': Orthogonal})

categories = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']

def prepare_data(audio_path):
    raw_audio, sr = librosa.load(audio_path,sr=16000)
    raw_audio, _ = librosa.effects.trim(raw_audio, top_db=25, frame_length=256, hop_length=64)
    audio_duration=len(raw_audio)/sr
    if audio_duration > 4:
        raw_audio=raw_audio[:4*sr]
    else:
        raw_audio = np.pad(raw_audio, (0, (4*sr)-len(raw_audio)), 'constant')

    zcr_list = []
    rms_list = []
    mfccs_list = []


    FRAME_LENGTH = 400
    HOP_LENGTH = 160
    sr=16000

    y = raw_audio

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)

    zcr_list.append(zcr)
    rms_list.append(rms)
    mfccs_list.append(mfccs)
    pda = np.concatenate((zcr_list,rms_list,mfccs_list),axis=1)
    pda = pda.astype('float32')
    return pda

def runner(audio_path):

    features = prepare_data(audio_path)

    pr = model.predict(features) 
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']
    predicted_emotion = emotion_labels[np.argmax(pr)]

    confidences = {categories[i]:np.round(float(pr[0, i]), 3) for i in range(len(categories))}

    return confidences

gui_params = {
    "fn":runner, 
    "title" : html.unescape(
        "<h1 style='text-align: center;'>Speech Emotion Recognition</h1>"
    ), 
    "inputs":gr.Audio(label="Audio file", type="filepath"),
    "outputs" : "label",
    "examples" : "examples",
    "description" : html.unescape(
        "<h2>Try uploading a WAV audio file or recording an audio clip for emotion recognition. You may also utilize the example files for testing. </h2>"
        )
}

demo = gr.Interface(**gui_params)

 
    demo.launch()