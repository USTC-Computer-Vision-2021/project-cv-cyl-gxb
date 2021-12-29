import cv2
import numpy as np
import math
import os
from obj_loader import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES=[100,160,120]
DEFAULT_COLOR1 = (255, 0, 0)
DEFAULT_COLOR2 = (0, 255, 0)


def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf3 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf=[bf1,bf2,bf3]
    # load the reference surface that will be searched in the video stream
    model1 = cv2.imread('D:/Program/augmented-reality-master/src/models/model1.jpg', flags=3)
    model2 = cv2.imread('D:/Program/augmented-reality-master/src/models/model2.jpg', flags=3)
    model3 = cv2.imread('D:/Program/augmented-reality-master/src/models/model3.jpg', flags=3)
    model_list=[model1,model2,model3]
    # Compute model keypoints and its descriptors
    kp_model_list, des_model_list=[],[] 
    for sam in range(len(model_list)):
        kp_model, des_model = orb.detectAndCompute(model_list[sam], None)
        kp_model_list.append(kp_model)
        des_model_list.append(des_model)
    
    match_len_list, match_flag= [[],[],[]], []
    tmp_flag=np.zeros(len(model_list))
    # Load 3D model from OBJ file
    obj1 = OBJ('D:/Program/augmented-reality-master/src/models/fox_color.obj', swapyz=True) 
    obj2 = OBJ('D:/Program/augmented-reality-master/src/models/rat_color.obj', swapyz=True) 
    obj3 = OBJ('D:/Program/augmented-reality-master/src/models/cat_color.obj',swapyz=True)  
    obj_list=[obj1,obj2,obj3]
    # init video capture
    cap = cv2.VideoCapture('D:/Program/augmented-reality-master/src/models/video.MP4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    totalframe=cap.get(cv2.CAP_PROP_FRAME_COUNT)

    out = cv2.VideoWriter('D:/Program/augmented-reality-master/src/result12.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), int(fps), (int(video_w),int(video_h)),True)
    for i in tqdm(range(int(totalframe))):
        # read the current frame
        match_list=[]
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            exit(0)
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        if i ==0:
            for sam in range(len(model_list)):
                # match frame descriptors with model descriptors
                tmp_match=bf[sam].match(des_model_list[sam], des_frame)
                # sort them in the order of their distance
                # the lower the distance, the better the match
                tmp_match = sorted(tmp_match, key=lambda x: x.distance)
                match_list.append(tmp_match)
                while(len(match_len_list[sam])<3):
                    match_len_list[sam].append(len(tmp_match))
                match_flag.append(len(tmp_match))
            match_flag[2] += 30
        else:
            for sam in range(len(model_list)):
                tmp_match=bf[sam].match(des_model_list[sam], des_frame)
                # sort them in the order of their distance
                # the lower the distance, the better the match
                tmp_match = sorted(tmp_match, key=lambda x: x.distance)
                match_list.append(tmp_match)
                # print(sam,len(tmp_match))
                match_len_list[sam].append(len(tmp_match))
                tmp_flag[sam]=len(tmp_match)-match_flag[sam]
            idx=np.argmax(tmp_flag)
            # idx=1
            # print(idx)
            # if i <100: idx=np.argmax(tmp_flag)
            # elif i<300: idx = 0
            # elif i <380: idx=2
            # else: idx=1
            if (match_len_list[idx][i]+match_len_list[idx][i+1]+match_len_list[idx][i+2])/3.0 > MIN_MATCHES[idx]:
                # differenciate between source points and destination points
                src_pts = np.float32([kp_model_list[idx][m.queryIdx].pt for m in match_list[idx]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in match_list[idx]]).reshape(-1, 1, 2)
                # compute Homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # Draw a rectangle that marks the found model in the frame
                h, w, c = model_list[idx].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
                # if a valid homography matrix was found render cube on model plane
                if homography is not None:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    frame = render(frame, obj_list[idx], projection, model_list[idx], True)
                    #frame = render(frame, model, projection)
                # # draw first 10 matches.
                # frame = cv2.drawMatches(model1, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
                # show result
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        out.write(frame)
    plt.figure()
    for sam in range(3):
        plt.plot(range(len(match_len_list[sam])),match_len_list[sam], label=str(sam))
        print(match_len_list[sam])
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    v_colors=obj.colors
    scale_matrix = np.eye(3) * 3
    h, w, c = model.shape
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points_c = np.array([v_colors[vertex - 1] for vertex in face_vertices])
        p_color=np.mean(points_c, axis=0)
        p_color *= 255
        p_color = tuple([int(x) for x in p_color])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillPoly(img, [imgpts], DEFAULT_COLOR1)
        else:
            # print(face)
            # color = hex_to_rgb(face[-1])
            # color = face[-1]  
            # color=color[::-1]
            # reverse
            cv2.fillPoly(img, [imgpts], p_color)
    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
# parser = argparse.ArgumentParser(description='Augmented reality application')

# parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
# parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
# parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
# parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# # TODO jgallostraa -> add support for model specification
# #parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

# args = parser.parse_args()

if __name__ == '__main__':
    main()
