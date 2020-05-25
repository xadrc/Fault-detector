# FAULT DETECTOR



### DESCRIPTION

Computer Vision application to automatically detect cracks or splits in video-monitored mechanical components in a factory

### ACHIEVEMENTS

* Ingestion of ~1600 close-up images of good and defect mechanical parts 
* Data augmentation
* Trained a CNN with `~96%` accuracy to automatically detect mechanical parts which need replacement

### DATA

1596 close-up pictures of monitored mechanical parts, of which 928 picturing defect parts and 668 without faults (non-public)

### CNN ARCHITECTURE

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 36992)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               4735104   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 4,828,481
Trainable params: 4,828,481
Non-trainable params: 0

```

### RESULTS

```
Epoch 20/25
62/62 [==============================] - 61s 981ms/step - loss: 0.0710 - acc: 0.9771 - val_loss: 0.1027 - val_acc: 0.9594
Epoch 21/25
62/62 [==============================] - 61s 985ms/step - loss: 0.0566 - acc: 0.9803 - val_loss: 0.1810 - val_acc: 0.9469
Epoch 22/25
62/62 [==============================] - 61s 987ms/step - loss: 0.0881 - acc: 0.9703 - val_loss: 0.1492 - val_acc: 0.9469
Epoch 23/25
62/62 [==============================] - 61s 980ms/step - loss: 0.0667 - acc: 0.9772 - val_loss: 0.0823 - val_acc: 0.9688
Epoch 24/25
62/62 [==============================] - 61s 979ms/step - loss: 0.0543 - acc: 0.9823 - val_loss: 0.0889 - val_acc: 0.9688
Epoch 25/25
62/62 [==============================] - 61s 981ms/step - loss: 0.0541 - acc: 0.9798 - val_loss: 0.1310 - val_acc: 0.9563
```

#

###### 2019
