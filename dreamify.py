import os
import argparse
import urllib.request
from glob import glob

from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--depth", default=3)
    parser.add_argument("--n_images", default=1)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--max_coeff", default=10)
    parser.add_argument("--max_loss", default=20)

    args = parser.parse_args()
    path = str(args.path)
    url = str(args.url)
    name = str(args.name)
    depth = int(args.depth)
    n_images = int(args.n_images)
    seed = int(args.seed)
    max_coeff = float(args.max_coeff)
    max_loss = float(args.max_loss)

    if not path and not url:
        print("Insert path with \"--path\" or url with \"--url\".")
        return
    
    if url:
        if not name:
            print("Please provide a name for the image folder with \"--name\".")
            return
        path = "Images/" + name + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        path += "original" + url[-4:]
        urllib.request.urlretrieve(url, path)

    model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
    model._layers[0].batch_input_shape = (None, None, None, 3)
    
    generate_dreams(path, model, depth, n_images, seed, max_coeff, max_loss)

def preprocess_image(img):
    """ Preprocess image for inception_v3 """
    
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = inception_v3.preprocess_input(img)
    
    return img

def deprocess_image(x):
    """ Deprocess image from inception_v3 tensor """
    
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception_v3 preprocess
    x /= 2.
    x += 0.5
    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

def compute_loss(input_image, feature_extractor, layer_settings):
    """
    Computes loss based on a feature extractor and the corresponding layers weights.
    
    Parameters:
        input_image (np.ndarray): numpy array of an image.
        feature_extractor (tf.Model): feature extractor with outputs corresponding to the layers.
        layer_settings (dict): dict with tf model layer names and corresponding weights in the loss function.
        
    Returns:
        loss (float): loss of the image w.r.t. the layers specified in layer_settings.
    """
    
    # Computes features from image
    features = feature_extractor(input_image)
    
    loss = tf.zeros(shape=())
    for key in features.keys():
        # Coefficient of the layer
        coeff = layer_settings[key]
        # Activation of the layer
        activation = features[key]
        # Adds to loss (avoid artifacts by removing borders)
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), 'float32'))
        #loss += coeff * tf.reduce_sum(tf.square(activation[:,2:-2,2:-2,:])) / scaling
        loss += coeff * tf.reduce_sum(tf.square(activation)) / scaling
    
    return loss

@tf.function
def gradient_ascent_step(img, feature_extractor, layer_settings, learning_rate):
    """
    Performs a gradient ascent step on an image.
    
    Parameters:
        img (np.ndarray): numpy array of an image.
        feature_extractor (tf.Model): feature extractor with outputs corresponding to the layers.
        layer_settings (dict): dict with tf model layer names and corresponding weights in the loss function.
        learning_rate (float): learning rate for the gradient ascent step.
        
    Returns:
        loss (float32): loss of the image w.r.t. the layers specified in layer_settings.
        img (np.ndarray): numpy array of the modified image
    """
    
    # Computes loss with GradientTape
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, feature_extractor, layer_settings)
    # Computes gradients and normalize
    grads = tape.gradient(loss, img)
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    # Gradient ascent step
    img += learning_rate * grads
    
    return loss, img

def gradient_ascent_loop(img, feature_extractor, layer_settings, iterations, learning_rate, octave, shape, destination_filepath=None, max_loss=None):
    """
    Performs the gradient ascent loop on an image.
    
    Parameters:
        img (np.ndarray): numpy array of an image.
        feature_extractor (tf.Model): feature extractor with outputs corresponding to the layers.
        layer_settings (dict): dict with tf model layer names and corresponding weights in the loss function.
        iterations (int): number of iterations for the loop.
        learning_rate (float): learning rate for the gradient ascent steps.
        max_loss (float): maximum loss before interruption (default=None).
        
    Returns:
        img (np.ndarray): numpy array of the modified image
    """
    
    # gradient ascent loop
    losses = []
    t = tqdm(range(iterations))
    for i in t:
        loss, img = gradient_ascent_step(img, feature_extractor, layer_settings, learning_rate)
        losses.append(loss)
        
        description = ""
        if destination_filepath:
            description += destination_filepath + " - "
        description += "Octave: {0:d}, Shape: {1:s}, Loss: [{2:.2f}, {3:.2f}]".format(octave, str(shape), np.min(losses), np.max(losses))
        t.set_description(description)
        t.refresh()
        if max_loss is not None and loss > max_loss:
            break
        #print("Loss at step %d: %.2f" % (i, loss))
    #print("Min loss: %.2f - Max loss: %.2f" % (np.min(losses), np.max(losses)))
    return img

def dreamify(source_filepath,
             destination_filepath,
             model,
             layer_settings,
             learning_rate = 0.01,
             num_octave = 3,
             octave_scale = 1.5,
             iterations = 20,
             max_loss = None):
    """
    Dreamifies an image.
    Returns nothing, the image is automatically saved at the requested destination.
    
    Parameters:
        image_path (str): path to the image folder (must end with "/").
        source_filename (str): name of the original file (with file format).
        destination_filename (str): name of the destination file (without file format).
        model (tf.Model): model to be used.
        layer_settings (dict): dict with tf model layer names and corresponding weights in the loss function.
        learning_rate (float): learning rate for the gradient ascent steps (deault=0.01).
        num_octave (int): number of subsampling octaves (default=3).
        octave_scale (float): scale of each subsampling (deault=1.5).
        iterations (int): number of iterations for the gradient ascent loop at each scale (deafault=20).
        max_loss (float): maximum loss before interruption on a loop (default=15.0).
    """
    
    # Dict of output layers
    outputs_dict = dict([(layer.name, layer.output)
                         for layer in [model.get_layer(key)
                                       for key in layer_settings.keys()]])
    # Feature extractor from model input layer and dict of output layers
    feature_extractor = keras.Model(inputs = model.inputs,
                                    outputs = outputs_dict)
    
    # Loads image, preprocesses it and gets its shape
    original_img = keras.preprocessing.image.load_img(source_filepath)
    original_img = preprocess_image(original_img)
    original_shape = original_img.shape[1:3]
    
    # Creates the list of successive shapes to use
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1] # Invert
    
    # Image at minimum shape
    shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])
    # Image to be modified
    img = tf.identity(original_img)
    
    for i, shape in enumerate(successive_shapes):
        #print("Octave %d with shape %s" % (i, shape))
        # Resizes at current shape
        img = tf.image.resize(img, shape)
        # Performs the gradient ascent loop on the current shape
        img = gradient_ascent_loop(img, 
                                   feature_extractor, 
                                   layer_settings,
                                   iterations=iterations,
                                   learning_rate=learning_rate,
                                   octave=i,
                                   shape=shape,
                                   destination_filepath=destination_filepath,
                                   max_loss=max_loss)
        # Restores lost details
        upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
        same_size_original = tf.image.resize(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        # Resizes the minimum shape to the current shape for next loop
        shrunk_original_img = tf.image.resize(original_img, shape)
        
    # Stores the resulting image
    keras.preprocessing.image.save_img(destination_filepath + '.png', deprocess_image(img.numpy()))
    
def single_layer_dreamify(source_filepath, model, layer_index, coeff, max_loss=None):
    """ 
    Dreamifies an image using a single layer.
    Returns nothing, the image is automatically saved at the requested destination.
    
    Parameters:
        source_filepath (str) = filepath of the source file.
        layer_index (int) = index of the layer to use.
        coeff (float) = coefficient of the layer in the layer_settings dict.
        max_loss (float) = maximum loss before breaking loop (default=None).
    """
    
    layer_settings = {model.layers[layer_index].name : coeff}
    
    destination_filepath = source_filepath[:-4] + "-" + str(layer_index) + "=" + str(np.around(coeff, decimals=2))
    
    dreamify(source_filepath, destination_filepath, model, layer_settings, max_loss=max_loss)
        
    return destination_filepath + '.png'

def generate_dreams(file_path, model, depth, n_images, seed=0, max_coeff=10, max_loss=20):
    
    np.random.seed(seed)
    
    for n in range(n_images):
        
        image_path = file_path
        
        for d in range(depth):

            layer = np.random.randint(0, len(model.layers))
            coeff = np.random.uniform(high=max_coeff)
            
            image_path = single_layer_dreamify(image_path, model, layer, coeff, max_loss)
              
if __name__ == "__main__":
    main()