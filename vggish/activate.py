import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import pandas as pd

import mel_features
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import glob

def load_audio(file_path, sample_rate=vggish_params.SAMPLE_RATE, min_length=1.0):
    audio, sr = sf.read(file_path, dtype='int16')
    print(f'Original sample rate: {sr} Hz')
    duration = len(audio) / sr
    print(f'Input signal length: {len(audio)} samples')
    print(f'Input signal duration: {duration:.2f} seconds')

    if len(audio) < sample_rate * min_length:
        pad_length = int(sample_rate * min_length) - len(audio)
        audio = np.pad(audio, (0, pad_length), 'constant', constant_values=0)
    if sr != sample_rate:
        audio = resampy.resample(audio, sr, sample_rate)
    return audio

def extract_vggish_features(audio_path):
    audio = load_audio(audio_path)
    examples_batch = vggish_input.waveform_to_examples(audio, vggish_params.SAMPLE_RATE)

    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

    pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    postprocessed_batch = pproc.postprocess(embedding_batch)

    return postprocessed_batch

path="../data/"
files=glob.glob(path+"pure/*.wav")
for file in files:
    output_file = path+"csvs/"+file.split("\\")[-1][:-3]+"csv"
    print(output_file)
    features = extract_vggish_features(file)
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)
    print(features)