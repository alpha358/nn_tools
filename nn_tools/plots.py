import numpy as np
import matplotlib.pyplot as plt


def display_imgs(imgs):
    columns = 3
    rows = 3
    img_nums = np.random.choice(len(imgs), columns * rows)
    img_data = imgs[img_nums]

    fig=plt.figure(figsize = (columns * 5, rows * 4))
    for i in range(rows):
        for j in range(columns):
            idx = i + j * columns
            fig.add_subplot(rows, columns, idx + 1)
            plt.axis('off')
#             img = img_data[idx].astype(np.float32)
            img = img_data[idx]
            plt.imshow(img)

    plt.tight_layout()
    plt.show()



# TODO: plot training history in a grid