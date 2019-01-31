import numpy as np
import cv2
import os


def rotate_img(img, angle):

    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    cols_rot = int(rows * abs_sin + cols * abs_cos)
    rows_rot = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += (cols_rot - cols) / 2
    M[1, 2] += (rows_rot - rows) / 2

    return cv2.warpAffine(img, M, (cols_rot, rows_rot))


def insert_subimg(img, subimg, row, col):

    assert img.shape[0] >= subimg.shape[0] + row
    assert img.shape[1] >= subimg.shape[1] + col

    result = np.copy(img)
    
    # grayscale segmentation mask
    segmentation = np.zeros(img.shape[0:2])

    mask = np.stack(
        (subimg[:, :, 3], subimg[:, :, 3], subimg[:, :, 3]), axis=2)
    result[row:row+subimg.shape[0], col:col+subimg.shape[1]] *= (1 - mask)
    result[row:row+subimg.shape[0], col:col +
           subimg.shape[1]] += mask * subimg[:, :, :3]

    # grayscale segmentation mask
    segmentation[row:row+subimg.shape[0],
                 col:col + subimg.shape[1]] = subimg[:, :, 3]

    blured = cv2.GaussianBlur(
        result[row:row+subimg.shape[0], col:col+subimg.shape[1]], (5, 5), 0)
    result[row:row+subimg.shape[0], col:col+subimg.shape[1]] *= (1 - mask)
    result[row:row+subimg.shape[0], col:col+subimg.shape[1]] += mask * blured

    return result, segmentation


def random_insert(img, subimg, size_range, angle_range):

    min_size, max_size = size_range
    min_angle, max_angle = angle_range

    size = np.random.uniform(min_size, max_size)
    size = size * min(img.shape[0], img.shape[1])
    scale = size / max(subimg.shape[0], subimg.shape[1])

    subimg_resc = cv2.resize(
        subimg, (int(subimg.shape[1] * scale), int(subimg.shape[0] * scale)))

    angle = np.random.uniform(min_angle, max_angle)
    subimg_resc = rotate_img(subimg_resc, angle)

    row = np.random.randint(img.shape[0] - subimg_resc.shape[0])
    col = np.random.randint(img.shape[1] - subimg_resc.shape[1])

    return insert_subimg(img, subimg_resc, row, col)


def random_fake_img(bgr_imgs, drone_imgs, size_range=(0.05, 0.5), angle_range=(0, 45)):

    drone_img = np.random.choice(drone_imgs)
    bgr_img = np.random.choice(bgr_imgs)

    return random_insert(bgr_img, drone_img, size_range, angle_range)


def read_imgs(path, formats=['png', 'jpg'], alpha_channel=False):

    if alpha_channel:
        read_format = cv2.IMREAD_UNCHANGED
    else:
        read_format = cv2.IMREAD_COLOR

    imgs = []
    for file in os.listdir(path):
        if file[-3:] in formats:
            img = cv2.imread(path + file, read_format)
            imgs.append(img / 255)

    return imgs


def read_img(file, alpha_channel=False):

    if alpha_channel:
        read_format = cv2.IMREAD_UNCHANGED
    else:
        read_format = cv2.IMREAD_COLOR

    img = cv2.imread(file, read_format) / 255

    return img


# working_dir = "/home/aiserver/drone_detection_data/"

# drone_imgs = read_imgs(working_dir + 'drones/', alpha_channel=True)

# output_dir = working_dir + "fakes/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # bgr_file_names = os.listdir(working_dir + 'DPED_bgr_dataset/iphone/')
# bgr_file_names = [f for f in os.listdir(working_dir + 'DPED_bgr_dataset/iphone/') if f[-3:] in ('png','jpg')]

# n_fakes = 6000
# size_range=(0.07, 0.2)
# angle_range=(-45, 45)
# resized_shape = (640, 480)

# for i in range(n_fakes):

#     bgr_name = np.random.choice(bgr_file_names)
#     bgr_img = read_img(working_dir + 'DPED_bgr_dataset/iphone/' + bgr_name)
# 	bgr_img = cv2.resize(bgr_img, resized_shape)

#     drone_img = np.random.choice(drone_imgs)

#     result = random_insert(bgr_img, drone_img, size_range, angle_range)
#     result = np.uint8(result * 255)

#     cv2.imwrite(output_dir + str(i) + '.png', result)
