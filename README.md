# Fast style transfer with dynamic channels

This is a fast style implementation using dynamic channels that can be used to shorten the training and inference time of the model.

# Dataset

we used COCO 2014 Training images dataset [80K/13GB] [(download)](https://cocodataset.org/#download).

## Usage

Train model

```
python neural_style\neural_style.py train --dataset PATH_TO_DATASETDIR --style-image PATH_TO_STYLEJPG --save-model-dir PATH_TO_MODELDIR --epochs 2 --cuda 1 --dynamic-channel 1 --sort-channel 1
```

- `--dataset`: path to dataset.
- `--style-image`: path to style-image.
- `--save-model-dir`: path to folder where trained model will be saved.
- `--cuda`: set it to 1 for running on GPU, 0 for CPU.
- `--dynamic-channel`: set it to 1 for running with dynamic channel, 0 for full channel.
- `--sort-channel`: set it to 1 for sorting channels before trainning.

Stylize image

```
python neural_style/neural_style.py eval --content-image PATH_TO_CONTENTIMG --output-image PATH_TO_OUTPUTIMG --cuda 1
```

- `--content-image`: path to content image.
- `--output-image`: path to output image.

