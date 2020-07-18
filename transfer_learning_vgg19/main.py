from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import matplotlib.pyplot as pltimg = load_img(train + "COVID-19/COVID-19 (1).png")
img = img_to_array(img)

input_shape = img.shape
print("input shape : ",input_shape)

num_classes = glob(test + "/*")
num_classes = len(num_classes)
print("number of classes : ",num_classes)

from glob import glob


img = load_img(train + "COVID-19/COVID-19 (1).png")
img = img_to_array(img)

input_shape = img.shape
print("input shape : ",input_shape)

num_classes = glob(test + "/*")
num_classes = len(num_classes)
print("number of classes : ",num_classes)


vgg = VGG16()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

test_data = ImageDataGenerator(1./255).flow_from_directory(test,
                                                     target_size=(224,224))

train_data = ImageDataGenerator().flow_from_directory(train,
                                                     target_size=(224,224))
vgg_layer_list = vgg.layers

model = Sequential()

for lyr in range(len(vgg_layer_list)-1):
  model.add(vgg_layer_list[lyr])

for layers in model.layers:
  layers.trainable = False

model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy',f1_m,precision_m, recall_m])

batch_size = 32

hist = model.fit_generator(train_data,
                           steps_per_epoch = 100,
                           epochs=50,
                           validation_data=test_data,
                           validation_steps = 20)



print(hist.history.keys)

plt.plot(hist.history['loss'],label= 'traininh loss')
plt.plot(hist.history['val_loss'],label= 'validation loss')
plt.plot(hist.history['accuracy'],label= 'validation accuracy')
plt.plot(hist.history['val_accuracy'],label = 'validation accuracy')
plt.legend()
plt.show()

plt.plot(hist.history['val_f1_m'],label= 'validation  f1_m')
plt.plot(hist.history['val_precision_m'],label= 'validation  precision_m')
plt.plot(hist.history['val_accuracy'],label= 'validation  recall_m')

plt.legend()
plt.show()


plt.plot(hist.history['f1_m'],label= 'train  f1_m')
plt.plot(hist.history['precision_m'],label= 'train precision_m')
plt.plot(hist.history['accuracy'],label= 'train  recall_m')

plt.legend()
plt.show()

