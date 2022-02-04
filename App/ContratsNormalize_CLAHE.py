import glob
import cv2

def CLAHE (img):
    img= cv2.imread(img,-1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    b,g,r=cv2.split(img)
    cl_b = clahe.apply(b)
    cl_g = clahe.apply(g)
    cl_r = clahe.apply(r)
    cl_rgb = cv2.merge([cl_b,cl_g,cl_r]);
    #cv2.imwrite('Kamvret.jpg',cl_rgb)
    return cl_rgb

