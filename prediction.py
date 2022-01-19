import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784",version=1,return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 10,train_size = 7500,test_size = 2500)

x_test_scaled = x_test / 255
x_train_scaled = x_train / 255

lr = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(x_train_scaled,y_train)

def get_prediction(img):
    img_pil = Image.open(img)

    img_bw = img_pil.convert("L")
    img_resized = img_bw.resize((28,28),Image.ANTIALIAS)
    # img_resized_inverted = PIL.ImageOps.invert(img_resized)
    pixel_filter = 20
    minpixel = np.percentile(img_resized,pixel_filter)
    img_resized_inverted_scaled = np.clip(img_resized-minpixel,0,255)
    max_pixel = np.max(img_resized)
    img_resized_inverted_scaled = np.asarray(img_resized_inverted_scaled) / max_pixel
    test_sample = np.array(img_resized_inverted_scaled).reshape(1,784)
    test_predict = lr.predict(test_sample)

    return test_predict[0]