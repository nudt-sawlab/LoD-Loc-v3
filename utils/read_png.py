import cv2
import numpy as np

def calculateCoefficients(zNear, zFar):
    a = zFar / (zFar - zNear)
    b = zFar * zNear / (zNear - zFar)
    return a, b
def inverseZBuffer(zBufferValue, a, b, nBits = 16):
    maxZBufferValue = nBits * 4096 - 1
    z = b / (zBufferValue / maxZBufferValue - a)
    return z


image_path = 'your_image.png'


image = cv2.imread('/home/ubuntu/Documents/code/3DTilesRender-3DTilesRender_dev/python/depthImage-19.tiff', cv2.IMREAD_ANYDEPTH)


if image is None:
    print("无法读取图像，请检查路径是否正确。")
else:

    print(f"图像数据类型: {image.dtype}")


    if image.dtype == np.float32 or image.dtype == np.float64:

        print("这可能是一个16位的图像。")
    elif image.dtype == np.uint16 or image.dtype == np.int16:
        print("这确实是一个16位的图像。")
    else:
        print("这不是一个16位的图像。")

if image is not None:

    min_val = image.min()
    max_val = image.max()
    print(image)

    print(f'Minimum pixel value: {min_val}')
    print(f'Maximum pixel value: {max_val}')
else:
    print('Error: Image not found or unable to read the image.')
    
    
    
    
    
    
    
    
    
