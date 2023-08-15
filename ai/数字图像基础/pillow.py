from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"]="SimHei"
img=Image.open("img/lena.tiff")
# arr_img_gray=np.array(img_gray)

# arr_img_new=255-arr_img_gray

# print("shape",arr_img_gray.shape,"\n")
# print(arr_img_gray)

# img_r,img_g,img_b=img.split()


# img.save("img/lena.jpg")
# img.save("img/lena.bmp")
# img1=Image.open("img/lena.jpg")
# img2=Image.open("img/lena.bmp")

# img_gray=img.convert("L")

plt.figure(figsize=(10,5))
# img_small=img.resize((64,64))

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
img_region=img.crop((100,100,400,400))
plt.imshow(img_region)


# plt.subplot(223)
# plt.axis("off")
# img_r90=img.transpose(Image.ROTATE_90)
# plt.imshow(img_r90)
# plt.title("逆时针旋转90度",fontsize=20)

# plt.subplot(224)
# plt.axis("off")
# img_tp=img.transpose(Image.TRANSPOSE)
# plt.imshow(img_tp)
# plt.title("转置",fontsize=20)

plt.show()
