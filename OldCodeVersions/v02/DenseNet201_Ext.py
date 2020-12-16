
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List, Dict
import os
import sklearn

print(f'Tensorflow version: {tf.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')

#print(tf.python.client.device_lib.list_local_devices())

os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf.device('/gpu:1')

############################### Functions ################################

def getClassWeightsFromLabels(labels: List[int]):# -> Dict[int]:
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return {'0':weights[0], '1':weights[1]}

def loadPartingExt(fname: str, extPath: str, sampleWeights: bool =False) -> pd.DataFrame:

    root = os.getcwd()
    print('root:',root)    
    columns=['filename','class','A1','A2','A3','A4','A5']
    if sampleWeights:
        columns.append('sampleWeights')
    data={c:[] for c in columns}

    
    f=open(os.path.join(root, extPath, fname),'r')
    lines=f.readlines()
    f.close()
    
    for line in lines:
        
        path, c = line.strip().split()
        atrs = ('.'.join(path.split('.')[:-1])).split('/')[-1].split('_')
        path=os.path.join(root, extPath, 'PATCHES', path)

        data[columns[0]].append(path)
        data[columns[1]].append(c)
        data[columns[2]].append(atrs[0])
        data[columns[3]].append(atrs[1])
        data[columns[4]].append(atrs[2])
        data[columns[5]].append(atrs[3])
        data[columns[6]].append(atrs[4])
        if sampleWeights:
            data[columns[7]].append(0)

    if sampleWeights:
        weights=getClassWeightsFromLabels(data[columns[1]])
        for i in range(len(lines)):
            data[columns[7]][i]=weights[data[columns[1]][i]]
        
    df = pd.DataFrame(data)
    print('len(df.index)',len(df.index))

    return df


### Model ###

def makeModel(inputShape: Tuple[int], modelName:str ='') -> keras.Model:
    """
    Source: https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model
    
    Note that I tend to prefer the super-explicit (if somewhat verbose) style. 
    This style is technically unnecessary, but it helps with readability.

    Load model by inputing the name to modelName
    Options are "Simple_ResNet", "SimpleNet", "InceptionResNetV2", "MobileNetV2", "ResNet50V2", "DenseNet121", "DenseNet201", and "NASNetLarge"
    """
    input = keras.Input(shape=inputShape, name="Input")
    x=None
    if modelName == "Simple_ResNet":

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(input)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        block_1_output = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

    elif modelName in ["InceptionResNetV2","MobileNetV2","ResNet50V2","DenseNet121","DenseNet201","NASNetLarge"]:
        
        if modelName=="InceptionResNetV2":
            baseModel = keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(150,150,3))(input)
        elif modelName=="MobileNetV2":
            baseModel = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(150,150,3))(input)
        elif modelName=="ResNet50V2":
            baseModel = keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(150,150,3))(input)
        elif modelName=="DenseNet121":
            baseModel = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(150,150,3))(input)
        elif modelName=="DenseNet201":
            baseModel = keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(150,150,3))(input)
        elif modelName=="NASNetLarge":
            x = tf.keras.layers.experimental.preprocessing.Resizing(height=331, width=331)(input)
            baseModel = keras.applications.NASNetLarge(include_top=False, weights="imagenet", input_shape=(331,331,3))(x)
        else:
            raise("Model Name Not In Recognized Keras Models")
        
        baseModel.trainable = False
        x = layers.Flatten()(baseModel)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

    elif modelName=="SimpleNet":
        x = layers.AveragePooling2D(pool_size=(50, 50))(input)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

    if x!=None:
        output=layers.Dense(2, activation="softmax")(x)
        return keras.Model(inputs=input, outputs=output, name=modelName)
    else:
        raise("Model Name Not Recognized")


### Data Parameters ###

patience = 3
batchSize = 128
# batchSize = 80


### Callbacks ###

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=1,
        restore_best_weights=True,
        min_delta=0.001
    ),
    # tf.keras.callbacks.TensorBoard(
    #     log_dir="logs",
    #     write_graph=True,
    #     write_images=True,
    # )
]


### Load Data ###

train_df=loadPartingExt('LABELS/train.txt','CSC790/Project/CNR-EXT-Patches-150x150/',True)
val_df=loadPartingExt('LABELS/val.txt','CSC790/Project/CNR-EXT-Patches-150x150/',True)
test_df=loadPartingExt('LABELS/test.txt','CSC790/Project/CNR-EXT-Patches-150x150/')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
)
trainDataset = datagen.flow_from_dataframe(
        train_df,
        target_size=(150, 150),
        batch_size=batchSize,
        weight_col='sampleWeights',
        class_mode='categorical',
)
validationDataset = datagen.flow_from_dataframe(
        val_df,
        target_size=(150, 150),
        batch_size=batchSize,
        weight_col='sampleWeights',
        class_mode='categorical',
)
testDataset = datagen.flow_from_dataframe(
        test_df,
        target_size=(150, 150),
        batch_size=batchSize,
        class_mode='categorical',
)


### Build and Train Model ###

### Premade models
# model_name="InceptionResNetV2"
# model_name="MobileNetV2"
# model_name="ResNet50V2"
# model_name="DenseNet121"
model_name="DenseNet201"
# model_name="NASNetLarge"

### Custom models
# model_name="Simple_ResNet"
# model_name="SimpleNet"

### Or load an existing model
#model_name = 'DenseNet201_acc98.87.h5'

if not model_name.endswith('.h5'):
    model = makeModel(inputShape=trainDataset[0][0][0].shape, modelName=model_name)

    opt=tf.optimizers.Adam()

    model.compile(
        optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()

    keras.utils.plot_model(model=model, to_file=model_name+".png", show_shapes=True)

    model.fit(
        trainDataset,
        epochs=20,
        callbacks=callbacks,
        validation_data=validationDataset,
        shuffle=True,
    )
else:
    model = keras.models.load_model(model_name)
    model.summary()

### Test Results ###

loss, accuracy = model.evaluate(trainDataset)
print('train:\tloss =', loss,'\taccuracy =', str(round(accuracy*100,2)))

loss, accuracy = model.evaluate(testDataset)
print('test:\tloss =', loss,'\taccuracy =', str(round(accuracy*100,2)))

if model_name == None:
    model.save(model.name+'_acc'+str(round(accuracy*100,2))+'.h5')