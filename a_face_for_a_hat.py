# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# !pip install gradio -q
# %tensorflow_version 1.x for colab only

import gradio as gr
import tensorflow as tf
import urllib.request
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

cnn_model = tf.keras.models.load_model("model.h5")
"""
labels = "./labels.txt"
labels = labels.read()
labels = labels.decode('utf-8').split("\n")[:-1]
"""
labels = ["hat", "face"]

def predict_shape(img):
  img = tf.math.divide(img, 255)
  preds = cnn_model.predict(img.reshape(-1, 28, 28, 1))[0]
  return {label: float(pred) for label, pred in zip(labels, preds)}


output = gr.outputs.Label(num_top_classes=2)
input = gr.inputs.Image(image_mode='L',
                        source='canvas',
                        shape=(28, 28),
                        invert_colors=True,
                        tool= 'select')



title="An 'Intelligent' Machine Who Mistook a Face for a Hat!"
description=" A project inspired from the case 'The Man Who Mistook His Wife for a Hat' by Oliver Sacks. A convolution neural network model is trained on Google's QuickDraw dataset." \
            " Start by drawing a face or a hat!"


gr.Interface(fn = predict_shape,
             inputs = input,
             outputs = output,
             live = True,
             title=title,
             description = description,
             capture_session=True).launch(share=True)

