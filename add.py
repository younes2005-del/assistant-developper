def main_func():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_hub as hub
    import numpy as np
    import h5py as h5
    import matplotlib.pyplot as plt
    import os
          
    import json

    training = []
    outputs = []
    train_val = []
    out_val = []
    
    name = input("what is your mode")

    x = input('type something : ')        

    for _ in range(20):
        y =1.0
        training.append(x)
        outputs.append(y)
        x = input('type something : ')        

    print(training)
    print(outputs)
    #x = "spotify"
#training = ['Ajoute pour moi un nouveau module', 'Aide moi à ajouter mes propres fonctionnalités ', 'ajoute une alarme à 16heures', 'prend moi un rendez vous avec le docteur', 'Aide moi à ajouter une nouvelle fonctionnalité', 'lance le processus de création de modules', 'Comment il fait dehors ?', "c'est quoi la vie ?", 'comment on ajoute un nouveau module', 'lance la douche', 'lance moi une musique sur spotify', 'aide moi à créer mon propre module', "c'est quoi le scrapping", '', 'Hey', 'Salut', 'regarde moi comment je pourrais créer mon module', 'crée un nouveau module', 'ça va ?', 'comment je me porte']
#outputs = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    movel = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" 
    hub_layer = hub.KerasLayer(movel, output_shape=[20], input_shape=[],
                               dtype=tf.string, trainable=True)
#training = ['ajoute un module', 'lance le process de création de module', 'regarde comment je pourrais créer un module', 'Crée un module', 'Ajoute pour moi une nouvelle fonction', 'ajoute pour moi une nouvelle fonction', 'Lance le process de création de fonctionnalité', 'aide moi à personnaliser tes fonctions', 'ajoute pour moi un nouveau module', 'Module', 'Fonction', 'Fonctionnalite', "ajoute une fonctionnalité pour moi s'il te plait", 'aide moi à ajouter plus de fonctions', 'Crée une nouvelle fonctionnalité', 'aide moi à créer une fonctionnalité', "S'il te plait crée pour moi une fonction", "Ajoute plus de fonctions s'il te plait", 'Comment on peut te perssonaliser', 'Merci de lancer le process de perssonalisation']
#outputs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    training_train = np.array(training[:10])
    outputs_train = np.array(outputs[:10])
    training_test = np.array(training[10:20])
    outputs_test = np.array(outputs[10:20])
    
    #training = ['spotify', 'hey', 'what the weather look like', 'run me a music ', 'hey', 'how are you doing ?',
                #"I'm 14", 'start a playlist on spotify', 'play Rap caviar playlist']
    #outputs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]




    model = tf.keras.Sequential()
    model.add(hub_layer)

    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(training_train,
                        outputs_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(training_test, outputs_test),
                        verbose=1)


    # prediction = model.predict("spotify")
    # print(prediction)
    model.save("models/" +name + '.h5')
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss') 
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    print("Votre model est prêt il a une erreur de", loss,"")
    
    with open("DB/DTB.json", "r") as f:
        DBT = json.load(f)
        print(DBT)
        f.close()
    for output in training:
        DBT.update({output : name})
    print(DBT)
    with open("DB/DTB.json","w") as file:    
        json.dump(DBT,file)
        file.close()
        
        
    lst_files = os.listdir("models")
    for file in lst_files:
        a = file.split(".")
        train = []
        op = []
        global_name = a[0]
        for db in DBT.items():
            train.append(db[0])
            if db[1] == name:
                op.append(1.0)
            else:
                op.append(0.0)
        train = np.array(train)
        op = np.array(op)
        print("Training the " + global_name + "model with the next data")
        print("inputs :", train)
        print("outputs :" , op)
    

        model_second = tf.keras.Sequential()
        model_second.add(hub_layer)

        model_second.add(tf.keras.layers.Dense(256, activation="relu"))
        model_second.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model_second.compile(optimizer='adam',
                    loss=tf.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = model_second.fit(train,
                        op,
                        epochs=20,
                        batch_size=512,
                        validation_data=(train, op),
                        verbose=1)
        history_dict = history.history
        history_dict.keys()

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        model.save("models/" + global_name + ".h5")
        

    
    
    
        
          
