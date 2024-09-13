import glob
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
import tqdm
import soundfile as sf
files=glob.glob("./data/raw/*.mp3")
wavPath="./data/wav/"
exportPath="./data/pure/"
for file in tqdm.tqdm(files):
    sound = AudioSegment.from_mp3(file)
    sound.export(wavPath+file.split("\\")[-1][:-3]+'wav',format="wav")
wavs=glob.glob("./data/wav/*.wav")
for file in tqdm.tqdm(wavs):
    sound,sr=librosa.load(file,mono=False)
    sound=librosa.to_mono(sound)
    soundres=librosa.resample(sound,sr,16000)
    sf.write(exportPath+file.split("\\")[-1][:],soundres,16000)
    audio=AudioSegment.from_wav(exportPath+file.split("\\")[-1][:])
    normalized_audio=normalize(audio)
    normalized_audio.export(exportPath+file.split("\\")[-1][:],format="wav")