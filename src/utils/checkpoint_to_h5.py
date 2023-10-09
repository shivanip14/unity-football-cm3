from ray.rllib.policy.policy import Policy
from tensorflow.python.keras.saving.saved_model import load
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
import keras
from keras import layers

checkpoint_path = "D:/MAI/TFM/codebases/gatech-unity-football-cm3/checkpoints/PPO_SoccerTwosRR_2023-09-08_14-41-4694jpdz3w/checkpoint_001000/policies/BluePlayer"
pre_trained_policy = Policy.from_checkpoint(checkpoint_path)

model = keras.Sequential()
model.add(layers.Dense(512, activation="tanh", input_shape=(338,)))
model.add(layers.Dense(512, activation="tanh"))
model.add(layers.Dense(9, activation="softmax"))
model.set_weights()
pre_trained_policy.export_model("./exported_model")
tf_model = tf.saved_model.load("./exported_model")
keras_model = load.load(tf_model)
keras_model.save("./exported_model/pre_trained_model.h5")
i = 0