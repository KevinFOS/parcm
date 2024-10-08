# PARCM: VGGish-Based "Public Acceptance" Rating of Classical Music

PARCM is a model designed to rate the "public acceptance" rate of classical music. It is based on the pre-trained [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset/vggish).

## References

**"Public Acceptance" Rating Model of Classical Music**, by Yikang Hong.

## Environment

```
conda create -n env_name python=3.11
conda activate env_name
pip install -r requirements.txt
```

Place all the documents of the [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), along with the [model checkpoints](https://storage.googleapis.com/audioset/vggish_model.ckpt) and the [embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz) in the `./vggish/` folder.

## Data

Place the raw MP3 form of the audios to be rated in the `./data/raw/` folder. Note that each of the audios must have sample rate **at least 16kHz**, and duration **at least 64 seconds**.

## Preprocess

```
python preprocess.py
```

By activating the `preprocess.py` in the main branch, the resampled and normalized WAV files are generated in the `./data/pure/` folder.

## Feature Extraction

```
cd vggish
python activate.py
cd ..
```

This activates the VGGish, exporting the features in csv form to `./data/csvs/`.

## Rating

```
python rating.py
```

The rating module returns a rating of "public acceptance", an integer ranging from 0 to 100 and information of the closest music piece in the standard dataset.
