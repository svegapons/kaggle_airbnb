""" 
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Classifier based on Keras code.
"""

import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import log_loss
from letor_metrics import ndcg_score
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import MaskedLayer
from keras import initializations
from keras.optimizers import SGD
import theano


class MyEarlyStopping(Callback):
    '''
    Stop training when a monitored quantity has stopped improving.
    I reimplemented this class in order to add a variable self.best_epoch
    with is the epoch number at which the computation is stopped.
    '''
    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1

def relu(x):
    """
    Implementation of relu..
    """
    return theano.tensor.switch(x<0, 0, x)

class MyPReLU(MaskedLayer):
    '''
    Implementation of PReLU
    '''
    def __init__(self, init='zero', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.alphas = None
        super(MyPReLU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alphas = self.init(input_shape)
        self.params = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        pos = relu(X)
        neg = self.alphas * (X - abs(X)) * 0.5
        return pos + neg

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "init": self.init.__name__}
        base_config = super(MyPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def clf_keras(data, cl_weight=None, random_state=0, ext_name=""):
    """
    Keras MLP classifier.
    The function applies the classifier twice:
    - First: Fit the classifier to (X_train, y_train) and predict on (X_valid).
             The prediction is stored in 'save/valid' folder.
    - Second: Fit the classifier to (X, y) = (X_train + X_valid, y_train + y_valid)
             and predict on (X_test). The prediction is stored in 'save/test' 
             folder.
             
    Parameters:
    ----------
    data: list
         [X_train, y_train, X_valid, y_valid, X_test]
    cl_weight: None or Dictionary
         Class weights, e.g. {0:1, 1:1.5, 2:1.6...} => weight for class 0 is 1, 
         for class 1 is 1.5, for class 2 is 1.6, and so on.
    random_state: numpy RandomState
         RandomState used for reproducibility
    ext_name: string
         Extra string to be used in the name of the stored prediction, e.g. it 
         can be used to identify specific parameter values that were used.
         
    Result:
    ------
    y_valid_pred: numpy ndarray shape=(n_samples_validation, n_classes)
              Labels of the predictions for the validation set.
    y_test_pred: numpy ndarray shape=(n_samples_test, n_classes)
              Labels of the predictions for the test set.
              
    Save:
    ----
    y_valid_pred: it is stored in save/valid folder
    y_test_pred: it is stored in save/test folder 
    """    
    X_train, y_train, X_valid, y_valid, X_test = data  
    
    ###Working on (X_Train => X_Valid)###
    #Centering and scaling the data
    ss = StandardScaler()
    XX_train = ss.fit_transform(X_train)
    XX_valid = ss.transform(X_valid)   
    
    #Computing binary labels (required by keras)
    lb = LabelBinarizer()
    yb_train = lb.fit_transform(y_train)
    yb_valid = lb.transform(y_valid)
    
    #Defining the network
    dims = XX_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = Sequential()
    model.add(Dropout(0.15, input_shape=(dims,)))
    model.add(Dense(input_dim=dims, output_dim=1000, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.35))    
    model.add(Dense(input_dim=1000, output_dim=650, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))    
    model.add(Dense(input_dim=650, output_dim=350, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.15))   
    model.add(Dense(input_dim=350, output_dim=n_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    #These two callback objects are used to stop the training at the best
    #iteration based on validation log_loss and to save that model
    es = MyEarlyStopping(monitor='val_loss', patience=30, verbose=1)
    mch = ModelCheckpoint('save/aux_keras_model', monitor='val_loss', 
                          save_best_only=True)
    #Training the model
    if cl_weight == None:
        model.fit(XX_train, yb_train, nb_epoch=1000, batch_size=512, 
                  validation_data=(XX_valid, yb_valid), verbose=2,
                  callbacks=[mch,es])
    else:
        model.fit(XX_train, yb_train, nb_epoch=1000, batch_size=512, 
                  validation_data=(XX_valid, yb_valid), verbose=2,
                  callbacks=[mch,es], class_weight=cl_weight)        
    #Loading the weights of the best epoch        
    model.load_weights('save/aux_keras_model')
    print es.best_epoch
    #Predicting the labels of X_valid
    y_valid_pred = model.predict_proba(XX_valid, batch_size=512, verbose=2)
    #Computing the scores
    ndcg_ke = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in \
    zip(yb_valid.tolist(), y_valid_pred.tolist())])
    logloss_ke = log_loss(y_valid, y_valid_pred)
    print ndcg_ke, logloss_ke
    
    #Saving the result
    rnd = random_state.randint(1000, 9999)
    pickle.dump(y_valid_pred, open('save/valid/v_KE_%s_%s_%s_%s'%(ext_name, 
                rnd, round(ndcg_ke, 4), round(logloss_ke, 4)), 'w'))
    
    ###Working on X => X_test###
    X = np.vstack((X_train, X_valid))
    y = np.hstack((y_train, y_valid))
    yb = lb.fit_transform(y)
    
    XX = ss.fit_transform(X)
    XX_test = ss.transform(X_test)
    #Defining the network
    model = Sequential()
    model.add(Dropout(0.15, input_shape=(dims,)))
    model.add(Dense(input_dim=dims, output_dim=1000, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.35))    
    model.add(Dense(input_dim=1000, output_dim=650, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))    
    model.add(Dense(input_dim=650, output_dim=350, init='glorot_normal'))
    model.add(MyPReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.15))   
    model.add(Dense(input_dim=350, output_dim=n_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    #Is it necessary just to compile??
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #Number of epochs is set to the optimun n epochs in validation
    n_epochs = es.best_epoch + 5
    
    if cl_weight == None:
        model.fit(XX, yb, nb_epoch=n_epochs, batch_size=512)
    else:
        model.fit(XX, yb, nb_epoch=n_epochs, batch_size=512,
                  class_weight=cl_weight)
                  
    y_test_pred = model.predict_proba(XX_test, batch_size=512, verbose=2)
        
    pickle.dump(y_test_pred, open('save/test/t_KE_%s_%s'%(ext_name, rnd), 'w'))
    
    return y_valid_pred, y_test_pred
    
    
          
          
          