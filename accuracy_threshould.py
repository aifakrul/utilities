# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:57:59 2021

@author: Md Fakrul Islam
"""

ACCURACY_THRESHOLD = 0.65
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):   
          print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
          self.model.stop_training = True

callbacks = myCallback()

#callbacks=[callbacks, lr_sched]
#start_time = time.time()
#history = model_ResNet152V2.fit(training_dataset.repeat(), steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[callbacks, lr_sched], 
                    #validation_data=validation_dataset.repeat(), validation_steps=validation_steps, shuffle=True)