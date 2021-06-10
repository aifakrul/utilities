# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:59:16 2021

@author: Md Fakrul Islam
"""

import numpy as np
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)

#callbacks=[callbacks, lr_sched]
#start_time = time.time()
#history = model_ResNet152V2.fit(training_dataset.repeat(), steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[callbacks, lr_sched], 
                    #validation_data=validation_dataset.repeat(), validation_steps=validation_steps, shuffle=True)