# HackHarvard2024

This is the Github Repo for our HackHarvard Project


# Spectrogram Model
We use a pretrained MobileNetV2 model with imagenet weights to appropriately differentiate the different spectrograms, allowing us to create a more accurate model in the hackathons time constraint. We use the AVID speech dataset and convert each .wav file to an appropriate spectrogram so the model can use a CNN to interpret this encoding from audio to image. The way this spectrogram is encoded will indicate to the model whether the speaker is using a soft, normal, loud, or very loud voice. 