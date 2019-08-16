import vggish_slim
import vggish_params
import vggish_input
import soundfile as sf
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import wave
import librosa

# Reference: https://colab.research.google.com/drive/1TbX92UL9sYWbdwdGE0rJ9owmezB-Rl1C#scrollTo=_7t20mo27zKf
def CreateVGGishNetwork(sess, hop_size=0.96):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """
  vggish_slim.define_vggish_slim()
  checkpoint_path = 'vggish_model.ckpt'
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size
  
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)

  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
    
  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }

# def ProcessWithVGGish(sess, vgg, file_name):
#   '''Run the VGGish model, starting with a sound (x) at sample rate
#   (sr). Return a whitened version of the embeddings. Sound must be scaled to be
#   floats between -1 and +1.'''

#   # Produce a batch of log mel spectrogram examples. (MFCC)
#   input_batch = vggish_input.wavfile_to_examples(file_name)
#   # print('Log Mel Spectrogram example: ', input_batch[0])

#   [embedding_batch] = sess.run([vgg['embedding']],
#                                feed_dict={vgg['features']: input_batch})

#   return embedding_batch


def ProcessWithVGGish(sess, vgg, file_name, start=0, stop=None):
  '''Run the VGGish model, starting with a sound (x) at sample rate
  (sr). Return a whitened version of the embeddings. Sound must be scaled to be
  floats between -1 and +1.'''

  # Produce a batch of log mel spectrogram examples. (MFCC)
  input_batch = vggish_input.wavfile_to_examples(file_name, start, stop)
  # print('Log Mel Spectrogram example: ', input_batch[0])
  [embedding_batch] = sess.run([vgg['embedding']],
                               feed_dict={vgg['features']: input_batch})

  return embedding_batch, input_batch


def get_number_of_samples_to_read(duration, sample_rate):
    return duration*sample_rate
        
def audio_length_calculator(data_shape, sample_rate):
    return (data_shape/sample_rate)

# Reshape the classification labels depending on the pre-processing parameters
def reshape_labels(label_path, audio_length, y_length, header=0):
    # Read the file and create the data frame
    dataframe = pd.read_csv(label_path, header=header)
    dataset = dataframe.values
    
    # Initialise the response array with 0
    y = np.zeros((y_length, 2), dtype=int)
    y[:, 1] = range(0, y_length) # Add the index of each row (useful to track the "hit" time)
    
    # Set the "hit" indices to 1
    for i in dataset:
        start_index = math.floor((i[0]*y_length/audio_length)-1)
        end_index = math.ceil(i[1]*y_length/audio_length)
        y[start_index:end_index, 0] = 1
    
    return y

# Get the time from index
def get_time_from_index(audio_data_length, audio_length, index):
    index = np.sort(index)
    times = []
    for i in index:
        times.append(round(i*audio_length/audio_data_length, 1))

    return np.unique(times)


# Read audio file using librosa
def read_audio_file_librosa(path, sample_rate=22050):
    # Read the audio file
    data, sample_rate = librosa.load(path, sr=sample_rate)
    # If stereo, convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sample_rate
