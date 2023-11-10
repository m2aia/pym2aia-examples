
import m2aia as m2
import tensorflow as tf
import numpy as np


# Running variance
def running_variance_update(existingAggregate, newValue):
    (count, mean, deltaM2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    deltaM2 += delta * delta2
    return (count, mean, deltaM2)

def running_variance_finalize(existingAggregate):
    (count, mean, deltaM2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, deltaM2 / count, deltaM2 / (count - 1))
    return (mean, variance, sampleVariance)

class BatchSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset: m2.Dataset.BaseDataSet , batch_size: int, shuffle: bool=True):
        super().__init__()
        self.gen = m2.BatchGenerator(dataset, batch_size, shuffle)
    
    def __len__(self):
        return self.gen.__len__()

    def on_epoch_end(self):
        self.gen.on_epoch_end()
    
    def __getitem__(self, index):
        X, Y = self.gen.__getitem__(index)
        return X, Y


def image_variance(I : m2.ImzMLReader):
    # initialize running variance calculation
    existingAggregate = (0, np.zeros_like(I.GetXAxis()), np.zeros_like(I.GetXAxis()))

    # update the variance
    for i in range(I.GetNumberOfSpectra()):
        _, ys = I.GetSpectrum(i)
        existingAggregate = running_variance_update(existingAggregate, ys)

    # finalize running variance
    _, var, _ = running_variance_finalize(existingAggregate)
    return var

def image_list_variance(images):
    
    # initialize running variance calculation
    existingAggregate = (0, np.zeros_like(images[0].GetXAxis()), np.zeros_like(images[0].GetXAxis()))
    for I in images:
        # update the variance
        for i in range(I.GetNumberOfSpectra()):
            _, ys = I.GetSpectrum(i)
            existingAggregate = running_variance_update(existingAggregate, ys)

    # finalize running variance
    _, var, _ = running_variance_finalize(existingAggregate)
    return var
