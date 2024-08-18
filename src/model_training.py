import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# load the preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

#creating sequential model
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#creating optimizer
adam = keras.optimizers.Adam(learning_rate=0.001)

#compile the model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

#train the model
history = model.fit(X_train, y_train, epochs=100)



#evaluate the model
loss_and_metrics = model.evaluate(X_test, y_test)
print(loss_and_metrics)
print('Loss = ',loss_and_metrics[0])
print('Accuracy = ',loss_and_metrics[1])

#see predictions
preds = model.predict(X_test)
preds = np.round(preds)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('./images/confusion_matrix.png')
print(classification_report(y_test, preds))

#graphing the loss and accuracy
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='red')
ax1.plot(history.history['loss'], color='red', label = 'Loss')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()

ax2.set_ylabel('Accuracy', color='blue')
ax2.plot(history.history['accuracy'], color='blue', label = 'Accuracy')
ax2.tick_params(axis='y', labelcolor='blue')

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.legend(loc='upper right')

plt.title('Loss and Accuracy')
plt.savefig('./images/loss_accuracy.png')

