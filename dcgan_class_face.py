from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Reshape, LeakyReLU, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import argparse
import shutil
import copy
import math
import os

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_2 (Dense)              (None, 36864)             9474048
_________________________________________________________________
batch_normalization_1 (Batch (None, 36864)             147456
_________________________________________________________________
activation_2 (Activation)    (None, 36864)             0
_________________________________________________________________
reshape_1 (Reshape)          (None, 12, 12, 256)       0
_________________________________________________________________
dropout_5 (Dropout)          (None, 12, 12, 256)       0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 24, 24, 256)       0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 24, 24, 128)       819328
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 128)       512
_________________________________________________________________
activation_3 (Activation)    (None, 24, 24, 128)       0
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 48, 48, 128)       0
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 48, 48, 64)        204864
_________________________________________________________________
batch_normalization_3 (Batch (None, 48, 48, 64)        256
_________________________________________________________________
activation_4 (Activation)    (None, 48, 48, 64)        0
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 96, 96, 64)        0
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 96, 96, 32)        51232
_________________________________________________________________
batch_normalization_4 (Batch (None, 96, 96, 32)        128
_________________________________________________________________
activation_5 (Activation)    (None, 96, 96, 32)        0
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 96, 96, 3)         2403
_________________________________________________________________
activation_6 (Activation)    (None, 96, 96, 3)         0
=================================================================
Total params: 10,700,227
Trainable params: 10,626,051
Non-trainable params: 74,176
'''
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=256, output_dim=256*12*12))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((12, 12, 256), input_shape=(256*12*12,)))
    model.add(Dropout(0.4))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(32, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    #model.add(Activation('sigmoid'))
    model.summary()
    plot_model(model, to_file='./generator_model.png', show_shapes=True, show_layer_names=True)
    return model

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 48, 48, 64)        4864
_________________________________________________________________
dropout_1 (Dropout)          (None, 48, 48, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 128)       204928
_________________________________________________________________
dropout_2 (Dropout)          (None, 24, 24, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 256)       819456
_________________________________________________________________
dropout_3 (Dropout)          (None, 12, 12, 256)       0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 512)         3277312
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 6, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 18432)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 18433
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0
=================================================================
Total params: 4,324,993
Trainable params: 4,324,993
Non-trainable params: 0
'''
def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            strides=2,
            padding='same',
            input_shape=(96, 96, 3),
            activation=LeakyReLU(alpha=0.2))
            )
    model.add(Dropout(0.4))
    model.add(
            Conv2D(128, (5, 5), 
            strides=2, 
            padding='same', 
            activation=LeakyReLU(alpha=0.2))
            )
    model.add(Dropout(0.4))
    model.add(
            Conv2D(256, (5, 5), 
            strides=2, 
            padding='same', 
            activation=LeakyReLU(alpha=0.2))
            )
    model.add(Dropout(0.4))
    model.add(
            Conv2D(512, (5, 5), 
            strides=2, 
            padding='same', 
            activation=LeakyReLU(alpha=0.2))
            )
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    plot_model(model, to_file='./discriminator_model.png', show_shapes=True, show_layer_names=True)
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def noise_random(number, size, method="uniform"):
    if method == 'uniform':
        #print('uniform')
        return np.random.uniform(-1, 1, size=(number, size))
    elif method == 'standard':
        #print('standard')
        return np.random.standard_normal(size=(number, size))
    elif method == 'test':
        ret = np.zeros((number, size), dtype=float)
        base = 1.0
        seed = np.linspace(-base, base, number, endpoint=True)
        divid = (base/number) * 50
        for i in range (number):
            #ret[i] = seed[i]
            ret[i] = np.random.uniform(seed[i]-divid, seed[i]+divid, size=(size,))
            #ret[i] = np.random.standard_normal(size=(size, ))
        print(ret)
        return ret
    else:
        assert()

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def combine_images_rgb(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = \
            img[:, :, :]
    return image

def extract_by_index(x_dataset, y_dataset, index=-1):
    decorated = [(y_dataset[i], i, x) for i, x in enumerate(x_dataset)]
    decorated.sort()
    x_dataset = np.array([x for y, i, x in decorated if (index==-1 or index==y)])
    y_dataset = np.array([y for y, i, x in decorated if (index==-1 or index==y)])
    return x_dataset, y_dataset

def double_samples(x_dataset, y_dataset):
    x_temp = copy.deepcopy(x_dataset)
    y_temp = copy.deepcopy(y_dataset)
    #image = combine_images_rgb(x_dataset[:1, :, :, :])
    #Image.fromarray(image.astype(np.uint8)).save("L.png")
    width = x_dataset.shape[1]
    hight = x_dataset.shape[2]
    for i in range(width):
        x_temp[:, :, width-1-i] = x_dataset[:, :, i]
    #image = combine_images_rgb(x_temp[:1, :, :, :])            
    #Image.fromarray(image.astype(np.uint8)).save("R.png")
    return np.concatenate((x_dataset,x_temp), axis=0),  np.concatenate((y_dataset,y_temp), axis=0)

def train(BATCH_SIZE, SOURCE, SPLIT=1, INDEX=-1, DOUBLE=False):
    if SOURCE == 'face':
        X_train, y_train = load_data_face() 

    if (SPLIT > 1): 
        num_split = int(X_train.shape[0]/SPLIT)
        start_split = np.random.randint(X_train.shape[0] - num_split)
        X_train, y_train = X_train[start_split:start_split+num_split], y_train[start_split:start_split+num_split]

    if (INDEX != -1):
        X_train, y_train = extract_by_index(X_train, y_train, INDEX)
    print("Select class: "+str(INDEX))
    
    if DOUBLE:
        X_train, y_train = double_samples(X_train, y_train)
	  
    print("Train on "+str(np.shape(y_train)[0])+" samples")
    print("Generate original image for reference")	
    image = combine_images_rgb(X_train[:BATCH_SIZE, :, :, :])
    Image.fromarray(image.astype(np.uint8)).save("Original_"+str(INDEX)+'_'+str(BATCH_SIZE)+".png")
    #image = combine_images_rgb(X_train[(int)(X_train.shape[0]/2):(int)(X_train.shape[0]/2)+BATCH_SIZE, :, :, :])
    #Image.fromarray(image.astype(np.uint8)).save("Originar_"+str(INDEX)+'_'+str(BATCH_SIZE)+".png")

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, :]
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    d_optim = RMSprop(lr=0.0004)
    dg_optim = RMSprop(lr=0.0002)
    d_on_g.compile(loss='binary_crossentropy', optimizer=dg_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    try:
        d.load_weights('discriminator')
        g.load_weights('generator')
    except:
    	  print("Can't load weight")

    epochs = int(100*10000/(np.shape(y_train)[0]))
    batchs = int(X_train.shape[0]/BATCH_SIZE)
    i_stat = np.zeros(epochs * batchs, dtype=int)
    d_loss_stat = np.zeros(epochs * batchs, dtype=float)
    g_loss_stat = np.zeros(epochs * batchs, dtype=float)
    i = 0
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", batchs)
        for index in range(batchs):
            noise = noise_random(BATCH_SIZE, 256, 'standard')
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images_rgb(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            #y = [1] * BATCH_SIZE + [0] * BATCH_SIZE #1 True, 0 Fake, try to maximum exact
            y = np.concatenate((np.random.uniform(0.8, 1.0, size=(BATCH_SIZE,)), np.random.uniform(0, 0.2, size=(BATCH_SIZE,)))) #~1 True, ~0 Fake, try to maximum exact
            loop = 0
            d_loss = d.train_on_batch(X, y)
            while d_loss > 1 and loop < 5: 
                d_loss = d.train_on_batch(X, y)
                loop += 1
            print("batch %d d_loss : %f" % (index, d_loss))
            #d_loss_r = d.train_on_batch(image_batch, [1] * BATCH_SIZE) #try to split true and fake into different batch, but cause the model crash
            #d_loss_f = d.train_on_batch(generated_images, [0] * BATCH_SIZE)
            #d_loss = d_loss_r + d_loss_f
            #print("batch %d d_loss : %f, (r : %f, f : %f)" % (index, d_loss, d_loss_r, d_loss_f))
            noise = noise_random(BATCH_SIZE, 256, 'standard')
            y = np.random.uniform(0.8, 1.0, size=(BATCH_SIZE,))
            d.trainable = False
            loop = 0
            g_loss = d_on_g.train_on_batch(noise, y) #1 True(actually Fake), try to maximum cheat the D
            while g_loss > 1 and loop < 5:
                g_loss = d_on_g.train_on_batch(noise, y) #1 True(actually Fake), try to maximum cheat the D
                loop += 1
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            i_stat[i], d_loss_stat[i], g_loss_stat[i] = i, d_loss, g_loss
            i = i+1
            if index % 10 == 1:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)
            if index % 10 == 6:
                shutil.copyfile('generator', 'generator.bak')
                shutil.copyfile('discriminator', 'discriminator.bak')

        plt.plot(i_stat[:(epoch+1)*batchs], d_loss_stat[:(epoch+1)*batchs], 'r', i_stat[:(epoch+1)*batchs], g_loss_stat[:(epoch+1)*batchs], 'b')
        plt.pause(5)
        plt.savefig("D-G.png")
        plt.close()

    #Save the final stastics on G, D
    plt.plot(i_stat, d_loss_stat, 'r', i_stat, g_loss_stat, 'b')
    plt.savefig("D-G_final.png")
    plt.show()


def generate(BATCH_SIZE, NOISE='uniform', NICE=False):
    g = generator_model()
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim, metrics=['accuracy'])
    g.load_weights('generator')
    if NICE:
        d = discriminator_model()
        d_optim = RMSprop(lr=0.0004)
        d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])
        d.load_weights('discriminator')
        noise = noise_random(BATCH_SIZE*20, 256, NOISE)
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:4], dtype=np.float32)
        #nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, :] = generated_images[idx, :, :, :]
        image = combine_images_rgb(nice_images)
    else:
        noise = noise_random(BATCH_SIZE, 256, NOISE)
        generated_images = g.predict(noise, verbose=1)
        image = combine_images_rgb(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image_"+NOISE+".png")

def preview(BATCH_SIZE, SOURCE):
    if SOURCE == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        INDEX_MAX = 10
    elif SOURCE == 'cifar100_f':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('fine')
        INDEX_MAX = 100
    elif SOURCE == 'cifar100_c':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')
        INDEX_MAX = 20
    else: 
        assert()
    y_train = y_train.flatten()#2D -> 1D

    for INDEX in range(INDEX_MAX):
        X_temp, y_temp = extract_by_index(X_train, y_train, INDEX)
        start = np.random.randint(X_temp.shape[0] - BATCH_SIZE)
        image = combine_images_rgb(X_temp[start:(start+BATCH_SIZE), :, :, :])
        Image.fromarray(image.astype(np.uint8)).save("Preview_"+str(INDEX)+'_'+str(BATCH_SIZE)+".png")
        print('Preview: '+str(INDEX))

def load_data_face(): 
    image_path = r'./face/'
    w = 96
    h = 96
    c = 3

    num = len([image_name for image_name in os.listdir(image_path)])
    print("Load training set: "+str(num))
    x = np.zeros([num, w, h, c])
    i = 0
    for image_name in os.listdir(image_path):
        #print(image_name)
        x[i] = Image.open(image_path+image_name)
        i += 1

    x = np.reshape(x, (num, w, h, c))
    #print(np.shape(x))
    np.random.shuffle(x)
    #print(np.shape(x))
    y = np.zeros(num, dtype=int)
    print("Loaded...")
    return x, y   

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--source", type=str, default="face")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--double", dest="double", action="store_true")
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--noise", type=str, default='standard')
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(NICE=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, SOURCE=args.source, SPLIT=args.split, DOUBLE=args.double, INDEX=args.index)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, NOISE=args.noise, NICE=args.nice)
    elif args.mode == "preview": 
    	  preview(BATCH_SIZE=args.batch_size, SOURCE=args.source)
