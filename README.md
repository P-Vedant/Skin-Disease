# Skin-Disease
AI
# Skin-Disease-Detection-AI

## Project Info

### Overview
This project is intended for the public to use to identify a variety of common skin conditions. By booting the flask server, you can upload a picture to the database to help continue to train the model, as well as pictures to test the model. This model is not intended to replace the opinion of a medical professional. Instead, we recommend you use this model for reasearch purposes. The datasets we used bias towards certain skin tones, and may provide some in-look into the extensive effects biased datasets have on the efficacy of an A.I. model.

### Technical explanation
This project includes a simple model for detecting a variety of skin diseases using Computer Vision. The model uses both tensorflow convolution layers, as well as tensorflow depth layers. The model is trained by minimizing an adam optomized sparse categorical crossentropy loss function. Our reasoning for using sparse categorical crossentropy opposed to regular categorical crossentropy is that sparse categorical crossentropy bypasses the need for one-hot vectors. Such vectors take up a signifigant amount of memory and computational time, and additionally are arguably more difficult to implement. Due to our algorithm selection, it should be noted that the final layers activation algorithm **must** be softmax for the program to function, as this ensures the output to be probabilistic in nature. The convolution layers utilize an ReLU algorithm for increased pattern complexity recognition. The convolution layers are actually all configured to be Leaky ReLU layers however, as this allows for a quick fix to the dead ReLU problem if need be.

## Dependencies Summary
- Tensorflow:
  - abstracted GPU level math
  - model building, compiling, training, and running
- OS:
  - environmental variable editing
- Flask:
  - web server hosting
- Gdown:
  - dynamically accessing training data
## Dependency Installation Guide
### Tensorflow
On windows, you will need to have an NVIDIA GPU to take advantage of GPU acceloration. Additionally, you will need [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) installed. If you do not have an NVIDIA GPU, you will not be able to use tensorflow on the GPU and instead will need to use it on the CPU.
```
pip install tensorflow
```

*Note: you may need to enable long path in the windows registery editor, for help to do so see [here](https://www.elevenforum.com/t/enable-long-file-path-names-in-windows-11.28659/)* \
\
On mac:
```
pip install tensorflow-macos
pip install tensorflow-metal
```

\
On Linux, you will need to have an NVIDIA GPU to take advantage of GPU acceloration. Additionally, you will need [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) installed. If you do not have an NVIDIA GPU, you will not be able to use tensorflow on the GPU and instead will need to use it on the CPU.
```
pip install tensorflow
```

### Flask
On windows:
```
pip install flask
```

\
On mac:
```
pip install flask
```

\
On linux:
```
pip install flask
```

### Gdown:


## Dataset Sources:

We used the [Kaggle Dataset here](https://www.kaggle.com/datasets/lysaapriani/skin-disease-and-normal-skin-dataset) to train our model.

Cancer Dataset: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?select=Skin+cancer+ISIC+The+International+Skin+Imaging+Collaboration
