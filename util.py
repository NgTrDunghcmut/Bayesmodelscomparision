"""
This files helps you read data from data files
"""

import pickle
import gzip
import glob
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def load_npy(file_name):
    """load_npy
    Load numpy data file. This is needed as python 2.7 pickle uses ascii as default encoding method but python 3.x uses utf-8.abs

    :param file_name: npy file path

    :return obj: loaded numpy object
    """

    if sys.version_info[0] >= 3:
        obj = np.load(file_name, encoding="latin1", allow_pickle=True)
    elif sys.version_info[0] >= 2:
        obj = np.load(file_name)

    return obj


def load_list(file_name):
    """load_list
    Load a list object to file_name.

    :param file_name: string, file name.
    """
    end_of_file = False
    list_obj = []
    f = open(file_name, "rb")
    python_version = sys.version_info[0]
    while not end_of_file:
        try:
            if python_version >= 3:
                list_obj.append(pickle.load(f, encoding="latin1"))
            elif python_version >= 2:
                list_obj.append(pickle.load(f))
        except EOFError:
            end_of_file = True
            print("EOF Reached")

    f.close()
    return list_obj


def save_list(list_obj, file_name):
    """save_list
    Save a list object to file_name

    :param list_obj: List of objects to be saved.
    :param file_name: file name.
    """

    f = open(file_name, "wb")
    for obj in list_obj:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def get_vehicle_data():
    """
    Load vehicle data and return it as a list: [train_x, train_y, test_x, test_y]
    """
    print("Reading vehicle data...")
    train_x, train_y, test_x, test_y = load_list(".//vehicles.dat")
    train_x = np.transpose(train_x, (2, 0, 1))
    test_x = np.transpose(test_x, (2, 0, 1))
    # print(x for x in train_x)
    # # print(train_x.s,type(train_x))
    # mean_per_pix = np.sum(train_x, axis=0) / train_x.shape[0]
    # std_per_pix = np.sqrt(
    #     np.sum((x - mean_per_pix) ** 2 for x in train_x) / train_x.shape[0]
    # )
    # print(mean_per_pix)
    # print(std_per_pix)
    # mean_all_pix = np.full_like(train_x[0], np.mean(train_x))
    # std_all_pix = np.std(train_x)
    # print(mean_all_pix, np.shape(mean_all_pix), std_all_pix, type(std_all_pix))
    # std_train_x_rc  = np.sqrt(np.sum(x - mean_train_x_rc for x in train_x))

    print("Done reading")
    return train_x, train_y, test_x, test_y


def read_mnist_gz(data_path, offset):
    with gzip.open(data_path, "rb") as f:
        dataset = np.frombuffer(f.read(), dtype=np.uint8, offset=offset)

    return dataset


from sklearn.metrics import roc_curve, roc_auc_score


def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, "r:")
    plt.plot(fpr[idx], tpr[idx], "ro")
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):

    data = {"y_Actual": target, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Predicted", "y_Actual"])
    confusion_matrix = pd.crosstab(
        df["y_Predicted"], df["y_Actual"], rownames=["Predicted"], colnames=["Actual"]
    )

    if perc:
        sns.heatmap(
            confusion_matrix / np.sum(confusion_matrix),
            annot=True,
            fmt=".2%",
            cmap="Blues",
        )
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.show()


def get_mnist_data(sampling_step=20):
    print("Reading fashion MNIST data...")
    train_x = read_mnist_gz(".//fashion-mnist/train-images-idx3-ubyte.gz", 16)
    train_y = read_mnist_gz(".//fashion-mnist/train-labels-idx1-ubyte.gz", 8)
    test_x = read_mnist_gz(".//fashion-mnist/t10k-images-idx3-ubyte.gz", 16)
    test_y = read_mnist_gz(".//fashion-mnist/t10k-labels-idx1-ubyte.gz", 8)
    num_train = len(train_y)
    num_test = len(test_y)

    train_x = train_x.reshape((num_train, 28 * 28))
    test_x = test_x.reshape((num_test, 28 * 28))
    # print(train_x[105, :])

    val_x = train_x[50000:, :]
    val_y = train_y[50000:]
    train_x = train_x[:50000, :]
    train_y = train_y[:50000]

    train_x = train_x[0::sampling_step, :]
    train_y = train_y[0::sampling_step]
    val_x = val_x[0::sampling_step, :]
    val_y = val_y[0::sampling_step]
    test_x = test_x[0::sampling_step, :]
    test_y = test_y[0::sampling_step]
    print(type(train_y))
    # For debugging purpose
    # train_x = train_x.reshape((num_train, 28, 28))
    # test_x = test_x.reshape((num_test, 28, 28))
    # plt.ion()
    # for i in range(0, num_train, 1000):
    #     img = train_x[i, :, :]
    #     plt.clf()
    #     plt.imshow(img, cmap="gray")
    #     plt.show()
    #     plt.pause(0.1)
    #     print(i)

    # for i in range(0, num_test, 100):
    #     img = test_x[i, :, :]
    #     plt.clf()
    #     plt.imshow(img, cmap="gray")
    #     plt.show()
    #     plt.pause(0.1)
    #     print(i)
    print("Done reading")
    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)
    print("val_x shape:", val_x.shape)
    print("val_y shape:", val_y.shape)
    print("test_x shape:", test_x.shape)
    print("test_y shape:", test_y.shape)
    # print("train_x shape:",train_x.shape)
    return (
        train_x.astype(np.float32),
        train_y,
        val_x.astype(np.float32),
        val_y,
        test_x.astype(np.float32),
        test_y,
    )


# print(get_mnist_data())

if __name__ == "__main__":
    get_vehicle_data()
