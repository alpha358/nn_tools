import glob
import keras
import numpy as np
import cv2

# ==============================================================================
# ============================== Get paths from folders =========================
# ==============================================================================


def get_dir_paths_with_label(folder_path, label):
    '''
    Just read all paths for specific directory
    Set constant label for all examples
    '''
    paths = glob.glob(folder_path + '/*.png')

    # (example_index, prob_index)
    labels = np.zeros([len(paths), 2])

    # set all labels to 1
    labels[:, label] = 1

    return paths, labels



def get_examples_with_label(folder_path, label, img_size):
    '''
    Read images from folder, assign constant label
    '''

    paths, labels = get_dir_paths_with_label(folder_path, label)

    X = np.zeros([len(paths), img_size[0], img_size[1], 3], dtype=np.float32)

    n = 0
    for path in paths:
        X[n, :, :, :] = load_image(path, img_size)
        n += 1
    
    return X, labels

def load_image(path, img_size):
    '''
    Read and resize rgb image
    '''
    img = cv2.cvtColor(
        cv2.imread(path),
        cv2.COLOR_BGR2RGB
    )

    img = np.float32(
        cv2.resize(
            img, (img_size[0], img_size[1])
        )
    )

    return img


# ==============================================================================
# ============================== Data Generator ===============================
# ==============================================================================


# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# credits: Eimantas

class SimpleDataGenerator(keras.utils.Sequence):

    '''
    labels --- one hot labels for each example: (example_index, prob_index)
    '''

    def __init__(self, paths, labels, batch_size, shape, shuffle=False, augmenter=None,
                 augment_fraction=0.5, use_cache=False, ram_GB=2):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.augmenter = augmenter  # augmenter is instance of Keras ImageDataGenerator
        self.augment_fraction = augment_fraction
        self.use_cache = use_cache
        if use_cache:
            # single image size in bytes (assuming float32)
            png_weight = shape[0] * shape[1] * shape[2] * 4
            self.cache_size = min(
                int(ram_GB * 1e9 / png_weight), paths.shape[0])
            self.cache = np.empty((self.cache_size, *shape), dtype='float32')
            for i in range(self.cache_size):
                self.cache[i] = self.__load_image(paths[i])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx *
                               self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.empty((paths.shape[0], *self.shape), dtype='float32')
        # Generate data
        for i, path in enumerate(paths):
            if self.use_cache == True:
                if indexes[i] < self.cache_size:
                    X[i] = self.cache[indexes[i]]
                else:
                    X[i] = self.__load_image(path)
            else:
                X[i] = self.__load_image(path)

            # Optional augmentation
            if self.augmenter is not None:
                if np.random.rand() < self.augment_fraction:
                    X[i] = self.augmenter.random_transform(X[i])

        y = self.labels[indexes]

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_image(self, path):
        '''
        Read and resize rgb image
        '''

        img = cv2.cvtColor(
            cv2.imread(path),
            cv2.COLOR_BGR2RGB
        )

        img = np.float32(
            cv2.resize(
                img, (self.shape[0], self.shape[1])
            )
        )

        return img
