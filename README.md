# Image Captioning with Deep Learning

This project focuses on generating captions for images using deep learning techniques. It leverages a combination of pre-trained models (MobileNetV3Large and BLIP) and custom neural networks to generate descriptive captions for images. The project also includes text-to-speech functionality to listen to the generated captions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [License](#license)

---

## Project Overview

The goal of this project is to generate meaningful captions for images using a combination of computer vision and natural language processing techniques. The project uses:
- **MobileNetV3Large** for image feature extraction.
- **Custom LSTM-based model** for caption generation.
- **BLIP (Bootstrapped Language-Image Pretraining)** for advanced captioning.
- **gTTS (Google Text-to-Speech)** for converting captions into audio.

---

## Features

- **Image Feature Extraction**: Extracts features from images using MobileNetV3Large.
- **Caption Generation**: Generates captions using a custom LSTM-based model or BLIP.
- **Text-to-Speech**: Converts generated captions into audio using gTTS.
- **Word Cloud Visualization**: Visualizes the most frequent words in the dataset.
- **Interactive Display**: Displays images with their corresponding captions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mo7239/image-caption-using-LSTM-CNN-and-Transformers.git
   cd image-caption-using-LSTM-CNN-and-Transformers

## Usage

1. **Extract Image Features**:
   - Run the script to extract features from images using MobileNetV3Large:
     ```python
     model = MobileNetV3Large(weights='imagenet', include_top=True)
     model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
     ```

2. **Generate Captions**:
   - Use the custom LSTM-based model or BLIP to generate captions:
     ```python
     def predict_caption(model, image, tokenizer, max_length):
         # Caption generation logic
         pass
     ```

3. **Display Results**:
   - Display images with their captions and listen to the generated captions:
     ```python
     def generate_caption(image_name):
         # Display image and caption
         pass
     ```

4. **Visualize Word Cloud**:
   - Generate a word cloud of the most frequent words in the dataset:
     ```python
     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))
     plt.imshow(wordcloud)
     plt.axis('off')
     plt.show()
     ```


   
