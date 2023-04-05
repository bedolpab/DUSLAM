import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
import glob
import sdl2
import sdl2.ext





def get_calib_matrix(directory):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(directory)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    with open('calib.pkl', 'wb') as f:
        pickle.dump([objpoints, imgpoints], f)


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    path = "check-imgs/GO*.jpg" 
    try:
        with open('calib.pkl', 'rb') as f:
            objp, imgp = pickle.load(f)
    except FileNotFoundError:
        get_calib_matrix(path)
        with open('calib.pkl', 'rb') as f:
            objp, imgp = pickle.load(f)

    print(f'OBJ: {objp}')
    print(f'IMG: {imgp}')
    input("Valid (?): ")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, (1280, 720), None, None)
    print(ret)
    input("validate ret")
    
    WIDTH = 1078
    HEIGHT = 578
    sdl2.ext.init()
    window = sdl2.ext.Window("frame", size=(WIDTH, HEIGHT))
    window.show()

    orb = cv2.ORB_create()
    
    fig, ax = plt.subplots()

    while(True):
        ret, frame = vid.read()
        h ,w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        kp = orb.detect(dst, None)
        kp, des = orb.compute(dst, kp)

        img_kp = cv2.drawKeypoints(dst, kp, None, color=(0, 255, 0), flags=0)
        plt.scatter([k.pt[0] for k in kp], [k.pt[1] for k in kp], color='green', marker='.')
        plt.gca().xaxis.tick_top()
        plt.gca().invert_yaxis()
        ax.set_facecolor("black")
        plt.pause(0.01)
        plt.cla()

        # Exit
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        windowArray = sdl2.ext.pixels3d(window.get_surface())
        windowArray[:, :, 0:3] = img_kp.swapaxes(0, 1)
        window.refresh()



        if(cv2.waitKey(1) & 0xFF == ord('q')):
           break

    vid.release()
    cv2.destoryAllWindows()
