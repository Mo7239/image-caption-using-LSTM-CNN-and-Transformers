# Image Captioning with Deep Learning

This project focuses on generating captions for images using deep learning techniques. It leverages a combination of pre-trained models (MobileNetV3Large and BLIP) and custom neural networks to generate descriptive captions for images. The project also includes text-to-speech functionality to listen to the generated captions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Contact](#contact)
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

## Model Architecture

The custom model architecture consists of:
- **Image Feature Extraction**: MobileNetV3Large.
- **Text Processing**: Embedding layer followed by Bidirectional LSTM.
- **Attention Mechanism**: Dot-product attention to focus on relevant parts of the image and text.
- **Decoder**: Dense layers to generate the final caption.

```python
inputs1 = Input(shape=(1000,),name='image')  
fe1 = BatchNormalization()(inputs1)
fe2 = Dense(512, activation='relu')(fe1)
fe2_projected = RepeatVector(max_length)(fe2)

inputs2 = Input(shape=(max_length,), name='text')
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = BatchNormalization()(se1)
se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)

attention = Dot(axes=[2, 2])([fe2_projected, se3])
attention = Activation('softmax')(attention)
context_vector = Dot(axes=[1, 1])([attention, se3])
context_vector = BatchNormalization()(context_vector)

context_vector = tf.keras.layers.Flatten()(context_vector)
decoder1 = Concatenate()([context_vector, fe2])
decoder2 = Dense(512, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
```   
## Results

- **Sample Captions**:
  - Generated captions are displayed alongside images.
  - Captions are converted to audio using gTTS.

- **Word Cloud**:
  - Visualizes the most frequent words in the dataset.

## Acknowledgments

- [MobileNetV3Large](https://arxiv.org/abs/1905.02244) for image feature extraction.
- [BLIP](https://arxiv.org/abs/2201.12086) for advanced caption generation.
- [gTTS](https://gtts.readthedocs.io/) for text-to-speech functionality.

## 📬 Contact  
For any inquiries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/mohamed-wasef-789743233/)
