import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

import keras
# from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D,BatchNormalization,Activation,AveragePooling2D
from keras.models import load_model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# !pip install kerassurgeon
from kerassurgeon import identify 
from kerassurgeon.operations import delete_channels,delete_layer
from kerassurgeon import Surgeon
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def my_get_all_conv_layers(model , first_time):

    '''
    Arguments:
        model -> your model
        first_time -> type boolean 
            first_time = True => model is not pruned 
            first_time = False => model is pruned
    Return:
        List of Indices containing convolution layers
    '''

    all_conv_layers = list()
    for i,each_layer in enumerate(model.layers):
        if (each_layer.name[0:6] == 'conv2d'):
            all_conv_layers.append(i)
    return all_conv_layers if (first_time==True) else all_conv_layers[1:]


def my_get_all_dense_layers(model):
    '''
    Arguments:
        model -> your model        
    Return:
        List of Indices containing fully connected layers
    '''
    all_dense_layers = list()
    for i,each_layer in enumerate(model.layers):
        if (each_layer.name[0:5] == 'dense'):
            all_dense_layers.append(i)
    return all_dense_layers


def my_get_weights_in_conv_layers(model,first_time):

    '''
    Arguments:
        model -> your model
        first_time -> boolean variable
            first_time = True => model is not pruned 
            first_time = False => model is pruned
    Return:
        List containing weight tensors of each layer
    '''
    weights = list()
    all_conv_layers = my_get_all_conv_layers(model,first_time)
    layer_wise_weights = list() 
    for i in all_conv_layers:
          weights.append(model.layers[i].get_weights()[0])  
    return weights


def my_get_l1_norms_filters_per_epoch(weight_list_per_epoch):

    '''
    Arguments:
        List
    Return:
        Number of parmaters, Number of Flops
    '''
    num_layers = len(weight_list_per_epoch)
    num_filters = [np.array(weight_list_per_epoch[i]).shape[-1] for i in range(num_layers)]
    sorted_filter_pair_sum = [{} for _ in range(num_layers)]

    filter_pair_similarities = [{f'{i+1}, {j+1}': 0.0 for i, j in combinations(range(_), 2)} for _ in num_filters]

    for layer_index in range(len(weight_list_per_epoch)):
        for epochs in weight_list_per_epoch[layer_index]:
            epochs = np.array(epochs)
            num_filter = num_filters[layer_index]
            flattened_filters = epochs.reshape(-1, num_filter).T
            cosine_sim = cosine_similarity(flattened_filters)
            for (i, j) in combinations(range(num_filter), 2):
                filter_pair_similarities[layer_index][f'{i+1}, {j+1}'] += cosine_sim[i, j]

    for layer_index in range(num_layers):  
        sorted_filter_pair_sum[layer_index] = dict(sorted(filter_pair_similarities[layer_index].items(), key=lambda item: item[1], reverse=True))

    return sorted_filter_pair_sum

def find_pruning_indices(model, weight_list_per_epoch, first_time, percentage):
    '''
    Arguments:
        model: The model from which weights are extracted
        weight_list_per_epoch: List of weight arrays per epoch for each layer
        first_time: Boolean indicating if it's the first time to extract weights
    Return:
        List of indices of filters to be pruned for each layer
    '''
    sorted_filter_pair_sums = my_get_l1_norms_filters_per_epoch(weight_list_per_epoch)
    
    all_layer_filter_pairs = []
    
    # Iterate through each layer's sorted filter pair sums
    for layer_index, sorted_filter_pair_sum in enumerate(sorted_filter_pair_sums):
        filter_pairs = []
        for key in sorted_filter_pair_sum.keys():
            filter1, filter2 = map(int, key.split(','))
            filter1 -= 1  
            filter2 -= 1 
            filter_pairs.append([filter1, filter2])
        all_layer_filter_pairs.append(filter_pairs)
    
    l1_norm_matrix_list = l1_norms(model, first_time)
    
    all_layer_pruning_indices = []
    for layer_index, filter_pairs in enumerate(all_layer_filter_pairs):
        pruning_indices = my_get_filter_pruning_indices(filter_pairs, l1_norm_matrix_list[layer_index], percentage)
        all_layer_pruning_indices.append(pruning_indices)
    
    num_filter_pairs_to_prune = int(len(filter_pairs) * percentage / 100 / 2)
    return all_layer_pruning_indices, all_layer_filter_pairs[:num_filter_pairs_to_prune]


def my_get_filter_pruning_indices(filter_pairs, l1_norms, prune_percentage):
    """
    Arguments:
        episodes_for_all_layers: List of selected filter pairs for one layer
        l1_norms: List of L1 norms of the filters for that layer
        prune_percentage: The percentage of filters to be pruned based on the most similar filter pairs (default is 10%)
    Return:
        filter_pruning_indices: List of indices of filters to be pruned for that layer
    """

    # Calculate the number of filters to prune based on the percentage
    num_filter_pairs_to_prune = int(len(filter_pairs) * prune_percentage / 100 / 2)
    
    filter_pruning_indices = set()
    
    # Iterate through the top `num_filter_pairs_to_prune` pairs
    for i in range(num_filter_pairs_to_prune):
        filter1 = filter_pairs[i][0]
        filter2 = filter_pairs[i][1]
        l1_norm_filter_1 = l1_norms[filter1]
        l1_norm_filter_2 = l1_norms[filter2]
        
        if l1_norm_filter_1 > l1_norm_filter_2:
            filter_pruning_indices.add(filter2)
        else:
            filter_pruning_indices.add(filter1)

    return list(filter_pruning_indices)


def l1_norms(model, first_time):
    conv_layers = my_get_all_conv_layers(model, first_time)
    l1_norms = list()
    for index, layer_index in enumerate(conv_layers):
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = weights.shape[-1]
        layer_l1_norms = []
        for i in range(num_filters):
            weights_sum = np.sum(np.abs(weights[:, :, :, i]))  # L1 norm is the sum of absolute values
            layer_l1_norms.append(weights_sum)
        l1_norms.append(layer_l1_norms)

    return l1_norms

    
def my_delete_filters(model,weight_list_per_epoch,percentage,first_time):
    filter_pruning_indices, _ = find_pruning_indices(model, weight_list_per_epoch, first_time, percentage)
    all_conv_layers = my_get_all_conv_layers(model,first_time)

    surgeon = Surgeon(model)
    for index,value in enumerate(all_conv_layers):
        print(index,value,filter_pruning_indices[index])
        surgeon.add_job('delete_channels',model.layers[value],channels = filter_pruning_indices[index])
    model_new = surgeon.operate()
    return model_new    


def count_model_params_flops(model,first_time):

    '''
    Arguments:
        model -> your model
        first_time -> boolean variable
        first_time = True => model is not pruned 
        first_time = False => model is pruned
    Return:
        Number of parmaters, Number of Flops
    '''

    total_params = 0
    total_flops = 0
    model_layers = model.layers
    for index,layer in enumerate(model_layers):
        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            
            params = layer.count_params()
            flops = conv_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            
            params = layer.count_params()
            flops = dense_flops(layer)
            print(index,layer.name,params,flops)
            total_params += params
            total_flops += flops

    return total_params, int(total_flops)


def dense_flops(layer):
    output_channels = layer.units
    input_channels = layer.input_shape[-1]
    return 2 * input_channels * output_channels


def conv_flops(layer):
    output_size = layer.output_shape[1]
    kernel_shape = layer.get_weights()[0].shape
    return 2 * (output_size ** 2) * (kernel_shape[0] ** 2) * kernel_shape[2] * kernel_shape[3]


class Get_Weights(Callback):
    def __init__(self,first_time):
        super(Get_Weights, self).__init__()
        self.weight_list = [] #Using a list of list to store weight tensors per epoch
        self.first_time = first_time
    def on_epoch_end(self,epoch,logs=None):
        if epoch == 0:
            all_conv_layers = my_get_all_conv_layers(self.model,self.first_time)
            for i in range(len(all_conv_layers)):
                self.weight_list.append([]) # appending empty lists for later appending weight tensors 
        
        for index,each_weight in enumerate(my_get_weights_in_conv_layers(self.model,self.first_time)):
                self.weight_list[index].append(each_weight)


#######################################################################
###################  Model Building
#######################################################################

model = keras.Sequential()

model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D())

model.add(Conv2D(filters=50, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=500, activation='relu'))

model.add(Dense(units=10, activation = 'softmax'))


def train(model,epochs,first_time, learning_rate=0.001):
    """
    Arguments:
        model:model to be trained
        epochs:number of epochs to be trained
        first_tim:
    Return:
        model:trained/fine-tuned Model,
        history: accuracies and losses (keras history)
        weight_list_per_epoch = all weight tensors per epochs in a list
    """

    lr = learning_rate

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 

    gw = Get_Weights(first_time)
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            callbacks=[gw],
            validation_data=(x_test, y_test))

    return model,history,gw.weight_list


#######################################################################
###################  Model Training
#######################################################################

model,history,weight_list_per_epoch = train(model,1,True)
initial_flops = count_model_params_flops(model,True)[1]
log_dict = dict()
log_dict['train_loss'] = []
log_dict['train_acc'] = []
log_dict['val_loss'] = []
log_dict['val_acc'] = []
log_dict['total_params'] = []
log_dict['total_flops'] = []
log_dict['filters_in_conv1'] = []
log_dict['filters_in_conv2'] = []

best_acc_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
log_dict['train_loss'].append(history.history['loss'][best_acc_index])
log_dict['train_acc'].append(history.history['accuracy'][best_acc_index])
log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
log_dict['val_acc'].append(history.history['val_accuracy'][best_acc_index])
a,b = count_model_params_flops(model,True)
log_dict['total_params'].append(a)
log_dict['total_flops'].append(b)
log_dict['filters_in_conv1'].append(model.layers[0].get_weights()[0].shape[-1])
log_dict['filters_in_conv2'].append(model.layers[2].get_weights()[0].shape[-1])
al = history

from keras import backend as K
def custom_loss(lmbda , regularizer_value):
  def loss(y_true , y_pred):
    # print(type(K.categorical_crossentropy(y_true ,y_pred)),K.categorical_crossentropy(y_true ,y_pred),regularizer_value)
    return K.categorical_crossentropy(y_true ,y_pred) + lmbda * regularizer_value
  return loss


def my_get_l1_norms_filters(model,first_time):
    """
    Arguments:
        model:

        first_time : type boolean 
            first_time = True => model is not pruned 
            first_time = False => model is pruned
        Return:
            l1_norms of all filters of every layer as a list
    """
    conv_layers = my_get_all_conv_layers(model, first_time)
    cosine_sums = list()
    for index, layer_index in enumerate(conv_layers):
        cosine_sums.append([])
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = len(weights[0,0,0,:])
        filter_vectors = [weights[:,:,:,i].flatten() for i in range(num_filters)]
        
        for i in range(num_filters):
            similarities = cosine_similarity([filter_vectors[i]], filter_vectors)[0]
            cosine_sum = np.sum(similarities) - 1
            cosine_sums[index].append(cosine_sum)
            
    return cosine_sums


def my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time):
    """
    Arguments:
        model:initial model
        weight_list_per_epoch:weight tensors at every epoch
        percentage:percentage of filter to be pruned
        first_time:type bool
    Return:
        regularizer_value
    """
    _, filter_pairs = find_pruning_indices(model, weight_list_per_epoch, first_time, percentage)
    l1_norms = my_get_l1_norms_filters(model,first_time)
    regularizer_value = 0
    for layer_index,layer in enumerate(filter_pairs):
        for episode in layer:
            regularizer_value += abs(l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]])  # Sum of abs differences between the episodes in all layers
    regularizer_value = np.exp((regularizer_value))
    print(regularizer_value)    
    return regularizer_value
    

def optimize(model,weight_list_per_epoch,epochs,percentage,first_time):
    """
    Arguments:
        model:inital model
        weight_list_per_epoch: weight tensors at every epoch
        epochs:number of epochs to be trained on custom regularizer
        percentage:percentage of filters to be pruned
        first_time:type bool
    Return:
        model:optimized model
        hisory: accuracies and losses over the process keras library
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    regularizer_value = my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time)
    print("INITIAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    model_loss = custom_loss(lmbda= 0.1 , regularizer_value=regularizer_value)
    # print('model loss',model_loss)
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=model_loss,optimizer=adam,metrics=['accuracy'])

    history = model.fit(x_train , y_train,epochs=epochs,batch_size = batch_size,validation_data=(x_test, y_test),verbose=1)
    print("FINAL REGULARIZER VALUE ",my_get_regularizer_value(model,weight_list_per_epoch,percentage,first_time))
    return model,history

count_model_params_flops(model,True)

print('Validation accuracy ',max(history.history['val_accuracy']))

validation_accuracy = max(history.history['val_accuracy'])
print("Initial Validation Accuracy = {}".format(validation_accuracy) )
max_val_acc = validation_accuracy
count = 0
all_models = list()
a,b = count_model_params_flops(model,False)
print(a,b)

#######################################################################
###################  Model Pruning
#######################################################################

while validation_accuracy - max_val_acc >= -0.01:


    print("ITERATION {} ".format(count+1))
    all_models.append(model)
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy
        
    if count < 1:
        optimize(model,weight_list_per_epoch,1,5,True)
        model = my_delete_filters(model,weight_list_per_epoch,5,True)
        model,history,weight_list_per_epoch = train(model,1,False)
   
    elif count < 2:
        optimize(model,weight_list_per_epoch,1,7,False)
        model = my_delete_filters(model,weight_list_per_epoch,7,False)
        model,history,weight_list_per_epoch = train(model,1,False)

    elif count < 3:
        optimize(model,weight_list_per_epoch,1,9,False)
        model = my_delete_filters(model,weight_list_per_epoch,9,False)
        model,history,weight_list_per_epoch = train(model,1,False)

    elif count < 4:
        optimize(model,weight_list_per_epoch,1,11,False)
        model = my_delete_filters(model,weight_list_per_epoch,11,False)
        model,history,weight_list_per_epoch = train(model,1,False)

    elif count < 5:
        optimize(model,weight_list_per_epoch,1,13,False)
        model = my_delete_filters(model,weight_list_per_epoch,13,False)
        model,history,weight_list_per_epoch = train(model,1,False)

    elif count < 10:
        optimize(model,weight_list_per_epoch,1,15,False)
        model = my_delete_filters(model,weight_list_per_epoch,15,False)
        model,history,weight_list_per_epoch = train(model,1,False)

    else:
        optimize(model,weight_list_per_epoch,10,35,False)
        model = my_delete_filters(model,weight_list_per_epoch,35,False)
        model,history,weight_list_per_epoch = train(model,10,False)

    a,b = count_model_params_flops(model,False)
    print(a,b)
    
    # al+=history
    validation_accuracy = max(history.history['val_accuracy'])
    best_acc_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
    log_dict['train_loss'].append(history.history['loss'][best_acc_index])
    log_dict['train_acc'].append(history.history['accuracy'][best_acc_index])
    log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
    log_dict['val_acc'].append(history.history['val_accuracy'][best_acc_index])
    a,b = count_model_params_flops(model,False)
    log_dict['total_params'].append(a)
    log_dict['total_flops'].append(b)
    log_dict['filters_in_conv1'].append(model.layers[1].get_weights()[0].shape[-1])
    log_dict['filters_in_conv2'].append(model.layers[3].get_weights()[0].shape[-1])
    print("VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(count+1,validation_accuracy))
    count+=1

model.summary()

model,history,weight_list_per_epoch = train(model,60,False,learning_rate=0.001)

best_acc_index = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
log_dict['train_loss'].append(history.history['loss'][best_acc_index])
log_dict['train_acc'].append(history.history['accuracy'][best_acc_index])
log_dict['val_loss'].append(history.history['val_loss'][best_acc_index])
log_dict['val_acc'].append(history.history['val_accuracy'][best_acc_index])
a,b = count_model_params_flops(model,False)
log_dict['total_params'].append(a)
log_dict['total_flops'].append(b)
log_dict['filters_in_conv1'].append(model.layers[1].get_weights()[0].shape[-1])
log_dict['filters_in_conv2'].append(model.layers[3].get_weights()[0].shape[-1])
print("Final Validation Accuracy = ",(max(history.history['val_accuracy'])*100))

log_df = pd.DataFrame(log_dict)
log_df

log_df.to_csv(os.path.join('.', 'results', 'lenet5_2.csv'))

