"""
Imports
"""
import cv2 
import numpy as np

# Web Camera Set-Up using OpenCV's CascadeClassifier and VideoCapture
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def apply_mosaic_cv2(img, pixel_size):
    """
    Mosaic Type: Pixelization 
    Method: OpenCV. Utilizes the resize method to scale down the input into 'mini' and then resizes the 
            newly scaled img into the original shape using nearest neighbor interpolation

    Args:
        img: the original image to pixelate
        pixel_size (int): size of the pixel

    Returns:
        mosaic: final altered and pixelated version of the input img
    """
    mini = cv2.resize(img, (img.shape[1] // pixel_size, img.shape[0] // pixel_size), interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(mini, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mosaic

def apply_mosaic_manuel_np(img, pixel_size):
    """
    Mosaic Type: Pixelization 
    Method: Manuel. Utilizes the nearest-neighbor interpolation method to manually scale the input img down and 
                    rescale the altered image to the original dimensions. Uses NumPy to allow for quick, live results.
    
    Args:
        img: the original image to pixelate
        pixel_size (int): size of the pixel

    Returns:
        mosaic: final altered and pixelated version of the input img
    """
    small = img[::pixel_size, ::pixel_size]
    mosaic = np.repeat(np.repeat(small, pixel_size, axis=0), pixel_size, axis=1)
    mosaic = mosaic[:img.shape[0], :img.shape[1]]
    return mosaic # mosaic[:img.shape[0], :img.shape[1]]

def apply_mosaic_manual(img, pixel_size=10):
    """
    Mosaic Type: Pixelization 
    Method: Manuel. Utilizes the nearest-neighbor interpolation method to manually scale the input img down and 
                    rescale the altered image to the original dimensions. 
    
    Args:
        img: the original image to pixelate
        pixel_size (int): size of the pixel

    Returns:
        mosaic: final altered and pixelated version of the input img
  """
    h, w, c = img.shape
    mini_h, mini_w = h // pixel_size, w // pixel_size

    mini_img = np.zeros((mini_h, mini_w, c)) #, dtype=np.uint8
    for i in range(mini_h):
        for j in range(mini_w):
            og_x = min(h - 1, i * pixel_size)
            og_y = min(w - 1, j * pixel_size)
            mini_img[i, j] = img[og_x, og_y]

    mosaic_img = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            og_x = min(mini_h - 1, i // pixel_size)
            og_y = min(mini_w - 1, j // pixel_size)
            mosaic_img[i, j] = mini_img[og_x, og_y]

    return mosaic_img

def detect_face_mask(img):
    """
    Method: Utilized OpenCV's cvtColor and CascadeClassifier's detectMultiScale to 
            detect a face and create a mask for the detected region given the input img

    Args:
        img: the original image to pixelate

    Returns:
        mask: the final, computed mask for the provided img
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.zeros_like(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        mask[y:y+h, x:x+w] = 255

    return mask

def apply_mosaic_select_face(img, mask, pixel_size, use_manual):
    """
    Mosaic Type: Pixelization
    Method: Manual. Applies the mosaic effect selectively to the detected face using
                    either the pure manual or manual with NumPy methods.

    Args:
        img: the original image to pixelate
        mask: the computed mask for the detected face
        pixel_size (int): size of the pixel
        use_manual (bool): determines if the pure manual or manual with NumPy method should be used

    Returns:
        result: final altered and pixelated version of the input img, where only the face is mosaic-ified
    """
    mosaic = apply_mosaic_manual(img, pixel_size) if use_manual else apply_mosaic_manuel_np(img, pixel_size)
    result = img.copy()
    result[mask == 255] = mosaic[mask == 255]
    return result

def add_label_with_background(img, text, position=(50, 50)):
    """
    Method: Utilizes OpenCV's font and text methods to add a text label with 
            partially-transparent background to input img

    Args:
        img: the original image to pixelate
        text (str): the text to be displayed with the background
        position (tuple, optional): coordinates of the location for the text. Defaults to (50, 50).

    Returns:
        img: final altered version of the input img with a text label

    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size

    bg_x, bg_y = position
    bg_w, bg_h = text_w + 10, text_h + 10

    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (bg_x + 5, bg_y + bg_h - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

"""
Method Calls and Initializations
"""
#LINE BELOW FOR LIVE CAMERA FEED
capture = cv2.VideoCapture(0)
# LINE BELOW FOR INPUT IMAGE:
# capture = '/Users/oliviaheng/Desktop/CSCI 1430/finalproject-oliviaaheng/mosiacs/tompkin.png'
use_manual_mosaic = False

while 1: 
    #LINE BELOW FOR LIVE CAMERA FEED
    ret, img = capture.read() 
    # LINE BELOW FOR INPUT IMAGE:
    # img = cv2.imread('/Users/oliviaheng/Desktop/CSCI 1430/finalproject-oliviaaheng/mosiacs/tompkin.png')
    face_mask = detect_face_mask(img)

    """
    Key Toggles
    
    Usage: By clicking the 'm' key on the keyboard while the program is running, 
           the manual mosaic will switch between the pure manual (slow) and the 
           manuel with NumPy (fast)
    """
    key = cv2.waitKey(30) & 0xFF
    if key == ord('m'):
        use_manual_mosaic = not use_manual_mosaic
    elif key == 27:
        break

    mosaic_img = apply_mosaic_cv2(img, pixel_size=10)  
    face_mosaic_img = apply_mosaic_select_face(img, face_mask, pixel_size=15, use_manual=use_manual_mosaic)
    custom_mosaic_img = apply_mosaic_manual(img, pixel_size=10) if use_manual_mosaic else apply_mosaic_manuel_np(img, pixel_size=10)

    img_labeled = add_label_with_background(img.copy(), "Original Image")
    mosaic_img_labeled = add_label_with_background(mosaic_img.copy(), "Full Mosaic")
    face_mask_labeled = add_label_with_background(cv2.cvtColor(face_mask.copy(), cv2.COLOR_GRAY2BGR), "Face Mask Detection")
    
    if use_manual_mosaic:
        custom_mosaic_img_labeled = add_label_with_background(custom_mosaic_img.copy(), "Custom Mosaic (Manual)")
    else:
        custom_mosaic_img_labeled = add_label_with_background(custom_mosaic_img.copy(), "Custom Mosaic (Fast)")

    top_row = np.hstack((img_labeled, mosaic_img_labeled))
    bottom_row = np.hstack((face_mask_labeled, custom_mosaic_img_labeled))
    comparison_screen = np.vstack((top_row, bottom_row))

    cv2.imshow("4-Way Mosaic Comparison with Dynamic Labels", comparison_screen)

# capture.release()
cv2.destroyAllWindows()