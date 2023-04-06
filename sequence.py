import pickle
import os

with open("calib.pkl", "rb") as f:
    obj, img = pickle.load(f)

print(img)