# Imports
import time
# let's also import the abstract base class for our callback
from keras.callbacks import Callback


# for plots
import matplotlib.pyplot as plt
from IPython.display import clear_output


# ==============================================================================
# =============================== Shutdown Timer ===============================
# ==============================================================================
# Daniel Möller: [Make best use of a Kernel's limited uptime (Keras)
# https://www.kaggle.com/danmoller/make-best-use-of-a-kernel-s-limited-uptime-keras


# defining the callback
class TimerCallback(Callback):
    def __init__(self, maxExecutionTime, byBatch=False, on_interrupt=None):

        # Arguments:
        #     maxExecutionTime (number): Time in minutes. The model will keep training
        #                                until shortly before this limit
        #                                (If you need safety, provide a time with a certain tolerance)

        #     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
        #                             If False, will try to interrupt the model at the end of each epoch
        #                            (use `byBatch = True` only if each epoch is going to take hours)

        #     on_interrupt (method)          : called when training is interrupted
        #         signature: func(model,elapsedTime), where...
        #               model: the model being trained
        #               elapsedTime: the time passed since the beginning until interruption

        self.on_interrupt = on_interrupt
        self.maxExecutionTime = maxExecutionTime * 60

        # the same handler is used for checking each batch or each epoch
        if byBatch == True:
            # on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            # on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler

    # Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0  # time taken by the longest epoch or batch
        # time when the last trained epoch or batch was finished
        self.lastTime = self.startTime

    # this is our custom handler that will be used in place of the keras methods:
    # `on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
    def on_end_handler(self, index, logs):

        currentTime = time.time()
        self.elapsedTime = currentTime - self.startTime  # total time taken until now
        thisTime = currentTime - self.lastTime  # time taken for the current epoch
        # or batch to finish

        self.lastTime = currentTime

        # verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime

        # if the (assumed) time taken by the next epoch or batch is greater than the
        # remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:

            self.model.stop_training = True  # this tells Keras to not continue training
            print(
                "\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: "
                + str(self.elapsedTime / 60.) + " minutes )\n\n")

            # if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)


# ==============================================================================
# =============================== Plot Learning ================================
# ==============================================================================

# Live loss plots in Jupyter Notebook for Keras
# https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12,6))

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        plt.grid()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        plt.grid()

        plt.show()


class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure( figsize=(12,6) )

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.grid()
        plt.show()
