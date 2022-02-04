import glob
import cv2


path = glob.glob("CLAHE/*.jpeg")
cv_img = []
for img in path:
    img2= cv2.imread(img,-1)
    #print(img)
    #print(img2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    b,g,r=cv2.split(img2)
    cl_b = clahe.apply(b)
    cl_g = clahe.apply(g)
    cl_r = clahe.apply(r)
    cl_rgb = cv2.merge([cl_b,cl_g,cl_r]);
    cv2.imwrite(img,cl_rgb)
