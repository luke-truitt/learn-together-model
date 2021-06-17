import model_factory as mf
import data_preprocessing as dp
import keras
import numpy as np

def load_trained_model(classes):

    trained_model = mf.train_model(classes, save_model=True, fname="test3")

    return trained_model

def predict(file, model):

    img = dp.process_img(file)

    pd = model.predict(img)

    return pd

def load_model(fname):
    model = keras.models.load_model(fname)
    model.summary()
    X, y, enc = dp.get_test(["cat", "dog"])
    y_enc = enc.transform(y)
    i = 0
    correct = 0
    total = 0
    for x in X:
        if i % 5 == 0:
            x = np.reshape(x, (1,300, 300, 3))
            y_pred = int(model.predict(x)[0][0])
            if y_pred == y_enc[i]:
                correct += 1
                print("correct", y_pred, y_enc[i], y[i])
            else:
                print("wrong", y_pred, y_enc[i], y[i])
            total += 1
        i = i+1
    print(correct/total)
    return model

print(load_trained_model(["cat", "dog"]))
# load_model("models/test1")
