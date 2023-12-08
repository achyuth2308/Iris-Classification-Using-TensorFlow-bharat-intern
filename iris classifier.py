# %%
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
import random
import matplotlib.pyplot as plt

# %% [markdown]
# # Create Dataset

# %%
dataset=load_iris()
_x=dataset["data"]
_y=dataset["target"]
labels=dataset["target_names"]
feature_names=dataset["feature_names"]
x=[]
y=[]
assert len(_x)==len(_y)
train=[]
for i,j in zip(_x,_y):
    train.append([i,j])
random.shuffle(train)
for i,j in train:
    x.append(i)
    y.append(j)
x=np.array(x)
y=np.array(y)

# %% [markdown]
# # Create Model

# %%
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,input_shape=(4,),activation="relu"),
    tf.keras.layers.Dense(3,activation="softmax")
])

# %% [markdown]
# # Get Summary Of Model

# %%
model.summary()

# %% [markdown]
# # Compile The Model

# %%
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# %% [markdown]
# # Train The Model

# %%
history=model.fit(x,y,epochs=100)

# %% [markdown]
# # Epoch Graphs

# %% [markdown]
# ### Accuracy

# %%
plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train"],loc="best")

# %% [markdown]
# ### Loss

# %%
plt.plot(history.history["loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train"],loc="best")

# %% [markdown]
# # Predict

# %% [markdown]
# ### Versicolor

# %%
labels[np.argmax(model.predict([x])[5])]

# %%
labels[y[5]]

# %% [markdown]
# ### Virginica

# %%
labels[np.argmax(model.predict([x])[7])]

# %%
labels[y[7]]

# %% [markdown]
# ### Setosa 

# %%
labels[np.argmax(model.predict([x])[20])]

# %%
labels[y[20]]

# %% [markdown]
# # Save The Model

# %%
model.save("iris_classification.h5")

# %% [markdown]
# # Convert Model To TFLite Model For Edge Devices

# %%
tf_lite_converter=tf.lite.TFLiteConverter.from_keras_model(model)
with open("iris_classification.tflite","wb")as f:
    f.write(tf_lite_converter.convert())


