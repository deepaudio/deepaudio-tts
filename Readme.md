## What is deepaudio-tts?
Deepaudio-tts is a framework for training neural network based Text-to-Speech (TTS) models. It inlcudes or will include  popular neural network architectures for tts and vocoder models. 

To make it easy to use various functions such as mixed-precision, multi-node training, and TPU training etc, I introduced PyTorch-Lighting and Hydra in this framework. *It is still in development.*


## Training examples
1. Preprocess you data. (Scripts comming soon, or you can follow the tutorial of paddle speech for this step.)
2. Train the model. You can choose one experiment in deepaudio/tts/cli/configs/experiment. Then train the model with following lines:
```
$ export PYTHONPATH="${PYTHONPATH}:/dir/of/this/project/"
$ python -m deepaudio.tts.cli.train experiment=tacotron2 datamodule.train_metadata=/you/path/to/train_metadata datamodule.dev_metadata=/you/path/to/dev_metadata
```

## Supported Models
1. Tacotron2
2. FastSpeech2
3. Transformer TTS
4. Parallel WaveGAN
5. HiFiGAN
6. VITS

## Future plan
### clean code
1. Remove redundant codes.
2. make deepaudio.tts.models more clean.
### Models
1. Other models.
2. Pretrained models. 
### Deployment
1. onnx
2. jit
## How to contribute to deepaudio-tts

It is a personal project. So I don't have enough gpu resources to do a lot of experiments. 
This project is still in development. 
I appreciate any kind of feedback or contributions. Please feel free to make a pull requsest for some small issues like bug fixes, experiment results. If you have any questions, please [open an issue](https://github.com/deepaudio/deepaudio-tts/issues).

## Acknowledge
I borrowed a lot of codes from [espnet](https://github.com/espnet/espnet) and [paddle speech](https://github.com/PaddlePaddle/PaddleSpeech)