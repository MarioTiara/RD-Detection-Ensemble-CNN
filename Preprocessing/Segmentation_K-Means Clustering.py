import numpy as np
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import time
import os

def kmeans_images(img, k):
    data = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    if data.shape[1] > 3:
        data = data[:, :3]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_.reshape(img.shape[0], img.shape[1])
    centers = kmeans.cluster_centers_
    return labels, centers


def creat_image(labels, centers):
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() > 1):
        img /= 255
    return img


if __name__ == "__main__":
    clusters = 10
    pathFolder = "0"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    bvDestinationFolder = "Km/"    
    if not os.path.exists(bvDestinationFolder):
        os.mkdir(bvDestinationFolder)   
    for file_name in filesArray:
        print(pathFolder+'/'+file_name)
        file_name_no_extension = os.path.splitext(file_name)[0]
        img = mpimg.imread(pathFolder+'/'+file_name)
        labels, centers = kmeans_images(img, clusters)
        a=creat_image(labels, centers)
        mpimg.imsave(bvDestinationFolder+file_name_no_extension+"_10_C.jpg",a)             

