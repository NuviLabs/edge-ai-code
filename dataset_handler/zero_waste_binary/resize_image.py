from PIL import Image
import os
# read an image and resize to 224x224 and save it
save_path = '/home/kaeun.kim/kaeun-dev/android_output4/manual_resize'
def read_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize(input_size, resample=Image.BILINEAR)
    # use basename to get the filename
    filename = os.path.basename(image_path)

    resized_image.save(os.path.join(save_path, filename))
    return resized_image

read_image("/home/kaeun.kim/kaeun-dev/android_output4/manual_crop/zero_0.00_1.00_0.7713272_1733132657811.webp", (224, 224))