import os

import PIL

import tensorflow as tf
from tensorflow import keras

import numpy as np
from copy import deepcopy
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import defaultdict

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


def decode_preds(data, model, n_iter=1000):
    
    """
    input_shape_from_model = model.layers[0].get_input_at(0).get_shape().as_list()[1:]
    target_shape = tuple([-1] + input_shape_from_model)
    
    data = np.reshape(data, target_shape)
    
    for idx in range(data.shape[0]):
        data[idx, :, :, :] = (data[idx, :, :, :] - np.amin(data[idx, :, :, :])) / (
                np.amax(data[idx, :, :, :]) - np.amin(data[idx, :, :, :]))
    """
    results = []
    for idx in range(n_iter):
        pred = model.predict(data)
        results.append(pred)
    
    results = np.asarray(results)
    predictions = np.mean(results, axis=0)
    return predictions



if __name__ == '__main__':
    
    images_path = "/home/leitherer/Real_space_AI_STEM/AI4STEM/scripts/FFT"
    images_path = "/home/leitherer/Real_space_AI_STEM/AI4STEM/scripts/fft_pngs"
    all_images = [_ for _ in os.listdir(images_path) if _.endswith('.png')]
    
    
    model = keras.models.load_model('../data/model_cnn_model.h5')
    last_conv_layer_name = "leaky_re_lu_6"
    
    class_list = ["bcc100", "bcc110", "bcc111","fcc100",
                  "fcc110", "fcc111", "fcc211", "hcp0001", "hcp10m10",
                   "hcp11m20", "hcp2m1m10"]
    numerical_to_text_label = dict(zip(range(len(class_list)), class_list))
    
    
    
    # GRADCAM
    truncated_model = deepcopy(model)
    truncated_model.layers[-1].activation = None
    
    
    n_iter = 100
    
    results_dict = defaultdict(dict)
    
    fig, axs = plt.subplots(len(all_images), 2, figsize=(5, 15))
    for idx, training_image in enumerate(all_images):
        img_path = os.path.join(images_path, training_image)
        im_frame = PIL.Image.open(img_path)
        img_array = np.array(im_frame)
        img_array = img_array[:, :, :3]
        # normalization:
        for j in range(img_array.shape[-1]):
            img_array[:, :, j] = (img_array[:, :, j] - np.amin(img_array[:, :, j])) / (
                            np.amax(img_array[:, :, j]) - np.amin(img_array[:, :, j]))
        #img_array = img_array[32:96, 32:96, :3]
        print(training_image, img_array.shape)
        axs[idx, 0].imshow(img_array[:, :, 0], cmap='gray')
        
        img_array = np.reshape(img_array, (1, 64, 64, 3))
        
        preds = decode_preds(data=img_array, model=model, n_iter=n_iter)
        print("Predicted:", numerical_to_text_label[np.argmax(preds, axis=-1)[0]])
        pred_label = numerical_to_text_label[np.argmax(preds, axis=-1)[0]]
        pred_prob = round( 100 * preds.flatten()[np.argmax(preds, axis=-1)[0]], 2)                    
        
        heatmaps = []
        for i in range(n_iter):
            heatmap = make_gradcam_heatmap(img_array, truncated_model, last_conv_layer_name)
            heatmaps.append(heatmap)
        heatmaps = np.asarray(heatmaps)
        averaged_heatmaps = np.mean(heatmaps, axis=0)
        axs[idx, 1].imshow(averaged_heatmaps, cmap='coolwarm')
        axs[idx, 0].set_title(training_image + 'Prediction: ({}, {}%)'.format(pred_label, pred_prob), fontsize=5)
        results_dict[training_image]['orig_image'] = img_array
        results_dict[training_image]['heatmap'] = averaged_heatmaps
        results_dict[training_image]['predictions'] = training_image + ' Prediction: ({}, {}%)'.format(pred_label, pred_prob)
    plt.tight_layout()
    plt.savefig('../results/gradcam.png', dpi=200)
    plt.close()
    
    for key in results_dict:
        heatmap = results_dict[key]['heatmap']
        image_1 = results_dict[key]['orig_image']
        prediction = results_dict[key]['predictions']
    
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        plt.imshow(heatmap)
        plt.savefig('heatmap.png')
        plt.close()
        
        #upsample = resize(heatmap, (64,64),preserve_range=True)
        heatmap = keras.preprocessing.image.array_to_img(np.expand_dims(heatmap, axis=0), 
                                                         data_format='channels_first')
        upsample = heatmap.resize((64, 64))
        
        
        plt.imshow(image_1[0, :, :, 0], cmap='gray')
        plt.imshow(upsample,alpha=0.5, cmap='coolwarm')
        plt.title(prediction)
        plt.savefig('../results/overlay_' + key, dpi=200)
        plt.close()