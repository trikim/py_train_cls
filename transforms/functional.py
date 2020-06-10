import cv2

def vflip(img):
    return cv2.flip(img, 0)

def hflip(img):
    return cv2.flip(img, 1)

def crop(img, i, j, h, w):
    return img[i:i+h, j:j+w, :]

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def five_crop(img, size):
    h, w = img.shape[:2]
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img[:crop_h, :crop_w, :]
    tr = img[:crop_h, w-crop_w:, :]
    bl = img[h-crop_h:, :crop_w, :]
    br = img[h-crop_h:, w-crop_w:, :]
    center = center_crop(img, (crop_h, crop_w))
    
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)                                                                                     

def ten_crop(img):
