
# IMAGE AUGMENTATION
# IMAGE AUGMENTATION
# IMAGE AUGMENTATION
# IMAGE AUGMENTATION




import cv2
import pandas as pd
import numpy as np
import random

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img


file = pd.read_csv("C:/Users/14oka/Desktop/train.csv")

foto = []
id = []
cins = []
yas = []
bolge = []
diag = []
ben = []
tar = []
adress = "D:/Users/14oka/Downloads/jpeg/trainresized2/"
for i in range(1,len(file)):
    if int(file.loc[i,'target']) == 1:
        image = cv2.imread("D:/Users/14oka/Downloads/jpeg/trainresized2/"+file.loc[i,'image_name']+".jpg", 1)
        for k in range(9):
            id.append(file.loc[i,'patient_id'])
            cins.append(file.loc[i,'sex'])
            yas.append(file.loc[i,'age_approx'])
            bolge.append(file.loc[i,'anatom_site_general_challenge'])
            diag.append(file.loc[i,'diagnosis'])
            ben.append(file.loc[i,'benign_malignant'])
            tar.append(int(1))


        #b = image.copy()
        # set green and red channels to 0
        #b[:, :, 1] = 0
        #b[:, :, 2] = 0
        #cv2.imwrite(adress+file.loc[i,'image_name']+"-B.jpg",b)
        foto.append(file.loc[i,'image_name']+"-B")

        #g = image.copy()
        # set blue and red channels to 0
        #g[:, :, 0] = 0
        #g[:, :, 2] = 0
        #cv2.imwrite(adress+ file.loc[i, 'image_name'] + "-G.jpg", g)
        foto.append(file.loc[i, 'image_name'] + "-G")
        #r = image.copy()
        # set blue and green channels to 0
        #r[:, :, 0] = 0
        #r[:, :, 1] = 0
        #cv2.imwrite(adress + file.loc[i, 'image_name'] + "-R.jpg", r)
        foto.append(file.loc[i, 'image_name'] + "-R")

        #Gblur = cv2.GaussianBlur(image, (5, 5), 0)
        #cv2.imwrite(adress + file.loc[i, 'image_name'] + "-GB.jpg", Gblur)
        foto.append(file.loc[i, 'image_name'] + "-GB")

        #median = cv2.medianBlur(image, 5)
        #cv2.imwrite(adress + file.loc[i, 'image_name'] + "-MB.jpg", median)
        foto.append(file.loc[i, 'image_name'] + "-MB")

        br = image.copy()
        br = brightness(br, 0.5, 3)
        cv2.imwrite(adress + file.loc[i, 'image_name'] + "-BR.jpg", br)
        foto.append(file.loc[i, 'image_name'] + "-BR")

        cs = image.copy()
        cs = channel_shift(cs, 60)
        cv2.imwrite(adress + file.loc[i, 'image_name'] + "-CS.jpg", cs)
        foto.append(file.loc[i, 'image_name'] + "-CS")

        fl = image.copy()
        fl = cv2.flip(fl,1)
        cv2.imwrite(adress + file.loc[i, 'image_name'] + "-FL.jpg", fl)
        foto.append(file.loc[i, 'image_name'] + "-FL")

        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sh = image.copy()
        sh = cv2.filter2D(sh, -1, filter)
        cv2.imwrite(adress + file.loc[i, 'image_name'] + "-SH.jpg", sh)
        foto.append(file.loc[i, 'image_name'] + "-SH")

df1 = pd.DataFrame({"image_name": foto,
                    "sex":cins,
                    "patient_id": id,
                    "age_approx": yas,
                    "anatom_site_general_challenge":bolge,
                    "diagnosis":diag,
                    "benign_malignant":ben,
                    "target":tar})
frames = [file,df1]

result = pd.concat(frames)
result.to_csv(r'D:/Users/14oka/Downloads/jpeg/trainresized2/train2.csv',index = False)

#IMAGE SHUFFLE

from sklearn.utils import shuffle
file = pd.read_csv("D:/Users/14oka/Downloads/jpeg/trainresized2/train2.csv")
file = shuffle(file)
file.to_csv(r'D:/Users/14oka/Downloads/jpeg/trainresized2/train22.csv',index = False)
file = pd.read_csv("D:/Users/14oka/Downloads/jpeg/trainresized2/train22.csv")
file.to_csv(r'D:/Users/14oka/Downloads/jpeg/trainresized2/train22.csv')
