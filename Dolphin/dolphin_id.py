# Might make your life easier for appending to lists
import os
from collections import defaultdict

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lib.buildmodels import build_model as bm, build_model
# our libraries
from lib.partition import split_by_day
import lib.file_utilities as util
from tensorflow.keras.layers import Dense, Input

# Any other modules you create


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()   # enable interactive plotting

    use_onlyN = np.Inf  # debug, only read this many files for each species

    raise NotImplementedError


if __name__ == "__main__":
     data_directory = "C:/Users/ADMIN/Documents/AI/AIA4/features/Gg"  # root directory of data
     data_directoryLO = "C:/Users/ADMIN/Documents/AI/AIA4/features/Lo/A"
     feature_getfiles = []
     feature_getfiles_Lo =[]
     feature_parsefiles = []
     feature_parsefiles_Lo =[]
     feature_getfiles = util.get_files(data_directory)
     feature_getfiles_Lo = util.get_files(data_directoryLO)
     feature_parsefiles = util.parse_files(feature_getfiles)
     feature_parsefiles_Lo = util.parse_files(feature_parsefiles_Lo)
     gg_by_dates = split_by_day(feature_parsefiles)
     lo_by_dates = split_by_day(feature_parsefiles_Lo)
     gg_by_dates_key= (list(gg_by_dates.keys()))
     gg_by_dates_values=(list(gg_by_dates.values()))
     lo_by_dates_key = (list(lo_by_dates.keys()))
     lo_by_dates_values = (list(lo_by_dates.values()))
     print(lo_by_dates_values)
     """
     xtrain_ggkey = train_test_split(gg_by_dates_key,test_size=0.7)
     ytrain_ggdata = train_test_split(gg_by_dates_values,test_size=0.7)
     xtest_ggkey = train_test_split(gg_by_dates_key,test_size=0.3)
     ytest_ggdata = train_test_split(gg_by_dates_values,test_size=0.3)
     xtrain_lokey = train_test_split(lo_by_dates_key, test_size=0.7)
     ytrain_lodata = train_test_split(lo_by_dates_values, test_size=0.7)
     xtest_lokey = train_test_split(lo_by_dates_key, test_size=0.3)
     ytest_lodata = train_test_split(lo_by_dates_values, test_size=0.3)
     array_gg_train = np.hstack(xtrain_ggkey)
     array_lo_train = np.hstack(xtrain_lokey)
     array_gg_lo = np.concatenate(array_gg_train,array_lo_train)


    
    Create a specification list consisting of features which will be fed as each layer
    """
specification_list =[(Dense, [20], {'activation':'relu', 'input_dim': 8}),
                     (Dense, [20], {'activation':'relu', 'input_dim':20}),
        (Dense, [20], {'activation': 'relu', 'input_dim': 20}),
     (Dense, 3, {'activation':'softmax', 'input_dim':20})
     ]
model = build_model(specification_list)
#compile mode
#predict model
#print Confusion matrix
