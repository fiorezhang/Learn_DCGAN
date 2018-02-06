from keras.models import Sequential
from keras.layers import Dense, Reshape, LeakyReLU, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist, fashion_mnist, cifar10
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import argparse
import math


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=256*8*8))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((8, 8, 256), input_shape=(256*8*8,)))
    model.add(Dropout(0.4))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(32, (5, 5), padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    #model.add(Activation('sigmoid'))
    model.summary()
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            strides=2,
            padding='same',
            input_shape=(32, 32, 3),
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
            strides=1, 
            padding='same', 
            activation=LeakyReLU(alpha=0.2))
            )
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


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

def train(BATCH_SIZE, SOURCE, SPLIT, INDEX):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()#2D -> 1D

    if (SPLIT > 1):
        num_split = int(X_train.shape[0]/SPLIT)
        start_split = np.random.randint(X_train.shape[0] - num_split)
        X_train, y_train = X_train[start_split:start_split+num_split], y_train[start_split:start_split+num_split]

    if (INDEX != -1):
        X_train, y_train = extract_by_index(X_train, y_train, INDEX)
    print("Select class: "+str(INDEX))
	  
    print("Train on "+str(np.shape(y_train)[0])+" samples")
    print("Generate original image for reference")	
    image = combine_images_rgb(X_train[:BATCH_SIZE, :, :, :])
    Image.fromarray(image.astype(np.uint8)).save("Original_"+str(INDEX)+'_'+str(BATCH_SIZE)+".png")
    

    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, :]
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    d_optim = RMSprop(lr=0.0002, decay=6e-8)
    dg_optim = RMSprop(lr=0.0001, decay=3e-8)
    d_on_g.compile(loss='binary_crossentropy', optimizer=dg_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    epochs = 200
    batchs = int(X_train.shape[0]/BATCH_SIZE)
    i_stat = np.zeros(epochs * batchs, dtype=int)
    d_loss_stat = np.zeros(epochs * batchs, dtype=float)
    g_loss_stat = np.zeros(epochs * batchs, dtype=float)
    i = 0
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", batchs)
        for index in range(batchs):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images_rgb(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE #1 True, 0 Fake, try to maximum exact
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE) #1 True(actually Fake), try to maximum cheat the D
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            i_stat[i], d_loss_stat[i], g_loss_stat[i] = i, d_loss, g_loss
            i = i+1
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

        plt.plot(i_stat[:(epoch+1)*batchs], d_loss_stat[:(epoch+1)*batchs], 'r', i_stat[:(epoch+1)*batchs], g_loss_stat[:(epoch+1)*batchs], 'b')
        plt.pause(5)
        plt.close()

    #Save the final stastics on G, D
    plt.plot(i_stat, d_loss_stat, 'r', i_stat, g_loss_stat, 'b')
    plt.show()


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g_optim = RMSprop(lr=0.0001, decay=3e-8)
    g.compile(loss='binary_crossentropy', optimizer=g_optim, metrics=['accuracy'])
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d_optim = RMSprop(lr=0.0002, decay=6e-8)
        d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images_rgb(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images_rgb(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--source", type=str, default="mnist")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, SOURCE=args.source, SPLIT=args.split, INDEX=args.index)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
