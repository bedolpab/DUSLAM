import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys
import glob
import sdl2
import sdl2.ext

FRAME_SIZE = (1280, 720)
CB_SIZE = (8, 6)


class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.paths = self.load_paths(self.directory)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = f'{self.directory}/{self.paths[idx]}'
        image = cv2.imread(item)

        data = {
            "directory": self.directory,
            "path": item,
            "image": image,
            "shape": image.shape,
            "size": image.size,
        }

        return data

    def load_paths(self, folder):
        paths = []
        for image in os.listdir(folder):
            if (image.endswith(".jpg")):
                paths.append(f'{self.directory}/{image}')
        return paths


def calibrate(dataset: Dataset, frame, cb_size):
    chessboardSize = cb_size
    dimensions = frame

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, len(dataset), 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0],
                           0:chessboardSize[1]].T.reshape(-1, 2)

    objPoints = []
    imgPoints = []

    for image in dataset.paths:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:
            objPoints.append(objp)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgPoints.append(corners)

            cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
            cv2.imshow('image', img)
            cv2.waitKey(50)

    cv2.destroyAllWindows()

    with open('calib.pkl', 'wb') as f:
        pickle.dump([objPoints, imgPoints], f)


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    path = "check-imgs"
    data = Dataset(path)
    try:
        with open('calib.pkl', 'rb') as f:
            objp, imgp = pickle.load(f)
    except FileNotFoundError:
        calibrate(data, FRAME_SIZE, CB_SIZE)
        with open('calib.pkl', 'rb') as f:
            objp, imgp = pickle.load(f)

    # Pickle file should save these parameters instead, fix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objp, imgp, (1280, 720), None, None)

    print("Calibrated camer: ", ret)
    print("\nCamera Matrix:\n", mtx)
    print("\nDistortion matrix:\n", dist)
    print("\nRotation Vectors:\n", rvecs)
    print("\nTranslation Vecttors:\n", tvecs)
    input()

    WIDTH = 1125
    HEIGHT = 607
    sdl2.ext.init()
    window = sdl2.ext.Window("frame", size=(WIDTH, HEIGHT))
    window.show()

    orb = cv2.ORB_create(3000)

    fig, ax = plt.subplots()

    while (True):
        ret, frame = vid.read()
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        kp = orb.detect(dst, None)
        kp, des = orb.compute(dst, kp)

        img_kp = cv2.drawKeypoints(dst, kp, None, color=(0, 255, 0), flags=0)
        plt.scatter([k.pt[0] for k in kp], [k.pt[1]
                    for k in kp], color='green', marker='.')
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

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destoryAllWindows()
