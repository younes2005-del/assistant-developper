

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib
import nltk
from past.builtins import execfile
import h5py


import os
cmd_lst = []
cmd = input("Hey I'm the AV, thaks to my dad,Younes I may serve you : ")
#main loop
while cmd != "quit":
    cmd_lst = []
    opt_lst = []
    cmd_lst.append(cmd)
    cmd_lst = np.array(cmd_lst)
    list_of_files = os.listdir('models')
    for m in list_of_files:
        path_model = 'models/'+m
        #print(path_model)
     
        model = tf.keras.models.load_model(path_model, custom_objects={'KerasLayer': hub.KerasLayer})
  
        prediction = model.predict(cmd_lst)
        print(m ,prediction)
        #print(prediction)
        opt_lst.append(prediction)
        
        '''
        if prediction[0][0] > 0.5:
                print("je suis rentré !!")
                str_lst = m.split('.')
                module = str_lst[0]
                execfile('modules/'+ module +'.py')
                main_func()
                break
        '''
    max_val = max(opt_lst)
    if max_val > 0.5 :
        index = opt_lst.index(max_val)#à verifier
        md = list_of_files[index]
        print(md)
        str_lst = md.split('.')
        module = str_lst[0]
        execfile('modules/'+ module +'.py')
        main_func()
    else :
        print("Je ne suis pas sûr de comprendre ce que vous voulez dire par" + "\"" + cmd + "\"")
        
       
            #executer main_function()
    cmd = input("what may I do for you: ")
