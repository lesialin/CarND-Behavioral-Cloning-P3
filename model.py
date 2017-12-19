import os
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Conv2D,Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import input

batch_size = 128

samples = input.make_samples('data')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# compile and train the model using the generator function
train_generator = input.generator(train_samples, batch_size=batch_size)
validation_generator = input.generator(validation_samples, batch_size=batch_size)
drop_rate = 0.2

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
#input dim = (160,320,3), output dim = (66,320,3)
model.add(Cropping2D(cropping=((70,24),(0,0))))
# input dim = (66,320,3), output dim = (31,158,24)
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(drop_rate))
# input dim = (31,158,24), output dim = (14,77,36)
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(drop_rate))
# input dim = (14,77,36), output dim = (5,36,48)
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(drop_rate))
# input dim = (5,36,48), output dim = (3,34,64)
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Dropout(drop_rate))
# input dim = (3,34,64), output dim = (1,32,64)
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Dropout(drop_rate))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
# checkpoint
filepath="model-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit_generator(train_generator,samples_per_epoch= \
        len(train_samples),validation_data=validation_generator, \
        nb_val_samples=len(validation_samples),callbacks=callbacks_list,nb_epoch=40,verbose=1)
model.save('model.h5')
#print log
print(history.history.keys())
#plot history loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean square error loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'], loc='upper right')
plt.show()


