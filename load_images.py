import cv2
import numpy as np
import os

def load_images(paths, N_max=1000, alpha_chnl=False):
    '''
        Purpose: load images from paths.
                    for drones use alpha channel
    '''

    if len(paths) > N_max:
        paths = paths[0: N_max]

    # imgs = np.zeros((len(paths), *img_shape, 3), dtype=np.uint8)
    imgs = []

    for num, img_path in enumerate(paths):
        if alpha_chnl:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return np.array(imgs)


def get_image_paths(folder):
    '''
    Purpose: get image paths from a folder
    Return: a list of paths
    '''
    # df = pd.DataFrame(columns=['path'])
    paths = []
    for i, fname in enumerate(os.listdir(folder)):
        if fname[-3:] in ['png', 'jpg', 'JPG', 'PNG']:
            # df.loc[i] = [os.path.join(folder, fname)]
            paths.append(os.path.join(folder, fname))

    return paths
