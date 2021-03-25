# Gerekli kütüphaneler yüklenir
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

#Veri seti yuklenir
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#verinin boyutlarını inceleyelim.
print("Eğitim setinin boyutu: ")
print(train_images.shape)
print("Test setinin boyutu: ")
print(test_images.shape)

# Eğitim ve test setindeki eşsiz sınıf sayısı
unique_classes,u_counts = np.unique(np.concatenate([train_labels,test_labels]),return_counts=True)
print(unique_classes)
print(u_counts)


class_names = ['Tişört / Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket',
               'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Çizme']
#Her sınıftan kaç örnek göstereceğimiz bilgisi
num_of_samples_per_class = 10
#Kaç sınıfımız var
num_classes = len(unique_classes)


#verisetimizdeki ilk 10 resmi görelim
#Bu daha hoş duruyor diye ekledim.
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Modelimizi olusturalim
from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import Model

model=keras.Sequential()
model.add(Conv2D(16, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=2,  activation='relu'))
model.add(Conv2D(64, kernel_size=2,  activation='relu'))
model.add(Flatten())
model.add(Dense(20,  activation='softmax'))



#Modelimizi inceleyelim
model.summary()

#Modeli derleme
model.compile(optimizer= 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

#Eğitim ve test
train_images = train_images.reshape(-1,28, 28, 1) 
test_images = test_images.reshape(-1,28, 28, 1) 

train_images = train_images / 255.0
test_images = test_images / 255.0


#Modeli eğitelim
model.fit(train_images, train_labels, epochs=60)

#Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test seti \"accuracy\" değeri {:.2f}".format(test_acc))


