import cv2


img_path = "/media/rgabhi/ABHINAV2/SIL7175/dataset/doi_10_5061_dryad_612jm649q__v20231226/DS1/1/raw/1_1_1_0.jpg"
# img_path2 = "./dataset/doi_10_5061_dryad_612jm649q__v20231226/DS1/1/SIRE-1_1_1_HT1.bmp"
img = cv2.imread(img_path)

if img is None:
    print("Error!")
else:
    print(f"img dim: {img.shape}")

# resize
new_w = int(img.shape[1]*0.5)
new_h = int(img.shape[0]*0.5)
new_dim = (new_w, new_h)

resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

print(f"resized dim: {resized_img.shape}")
cv2.imwrite("resized_1_1_1_0.jpg", resized_img)