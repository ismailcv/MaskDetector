# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:38:41 2022

@author: User
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#başlangıç ​​öğrenme oranını, eğitilecek dönem sayısını belirleyen parametreler.
INIT_LR = 1e-4
EPOCHS = 1 #bundan öğrenme sayısını belirliyoruz ne kadar arttırırsak kesinlik o kadar artacaktır.
BS = 32

#burada maskeli ve maskesiz insanların bulunduğu dosyayı
# bu sözlüğe aktarıyoruz ve dosya isimlerinide kategoriye attık.
DIRECTORY = r"C:\Users\User\OneDrive\Masaüstü\FaceMaskDetector-main\dataset"
CATEGORIES = ["with_mask", "without_mask"]

#veri kümesi dizinimizdeki görüntülerin listesini alın, ardından veri listesini (yani görüntüler) ve sınıf görüntülerini başlatın

print("[INFO] loading images...")

data = []
labels = []


# bu kısımda maskeli ve maskesiz resimleri karşılaştırma işlemi yapılıyor
# burada aslında bir ön işlem gerçekleşiyor.
# Aşşağıda resimlerin porgrama entegre edilme boyutuda verilmektedir
# Aşşağıda resimleri direk listeye dönüştürme işlemide gerçekleşiyor.

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

#burada data listesine resimleri ekliyoruz
#labels listesi kısmınada katergoriyi ekliyoruz 
        
    	data.append(image)
    	labels.append(category)

# etiketlerde tek sıcak kodlama gerçekleştirin
# makine öğrenmesinde sınıflandırma için kullanılır tek sıcak kodlama
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# veri büyütme için eğitim görüntü oluşturucusunu oluşturuyoruz
# burada önce en yakındaki işleme alarak yakınlaştırma yapıyoruz.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 ağını yükleyip, baş FC katman setlerinin kapalı kalmasını sağlayın
# burada 3 yazmamızın sebebi 3 ana rengi kullanmamız.
# Red Green Blu.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# temel modelin üstüne yerleştirilecek modelin başını oluşturoruz
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#kafa FC modelini temel modelin üstüne yerleştirin (bu, eğiteceğimiz gerçek model olacaktır)
model = Model(inputs=baseModel.input, outputs=headModel)

# temel modeldeki tüm katmanlar üzerinde döngü yapın ve ilk eğitim sürecinde güncellenmemeleri için onları dondurun

for layer in baseModel.layers:
	layer.trainable = False
        
# modelimizi derleyin
#adam optimizasyonu derin öğrenmede de kullanılan eğitim verilerine dayalı
#yinelemler için kullanılır.
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# ağın başını eğitmek

print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# info bilgi veriliyor anlamında kullanılıyor.
# test setiyle ilgili tahminlerde bulunun
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# test setindeki her bir görüntü için, karşılık gelen en büyük tahmin edilen olasılığa sahip etiketin dizinini bulmamız gerekir

predIdxs = np.argmax(predIdxs, axis=1)

# güzel biçimlendirilmiş bir sınıflandırma raporu gösterin

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save("mask_detector.model")

# eğitim grafiği çıkarır
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")