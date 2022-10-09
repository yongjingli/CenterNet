import numpy as np
import cv2


def draw_boxes_in_cv_img(img, boxes, color=(255, 0, 0), thinkness=1):
    for box in boxes:
        x1, y1, w, h = [int(tmp) for tmp in box[:4]]
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thinkness)
    return img


def draw_kps_in_cv_img(img, kps, color=(255, 0, 0), radius=5):
    for kp in kps:
        kp = kp.reshape(-1, 2)
        for _kp in kp:
            x, y = [int(tmp) for tmp in _kp]
            cv2.circle(img, (x, y), radius, color, -1)

        _kp = kp[5]
        x, y = [int(tmp) for tmp in _kp]
        cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)

        _kp = kp[11]
        x, y = [int(tmp) for tmp in _kp]
        cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)

        _kp = kp[13]
        x, y = [int(tmp) for tmp in _kp]
        cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)
    return img


def draw_skeleton_in_cv_img(img, kps, kps_score=None):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    pose_limb_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]
    for i, kp in enumerate(kps):
        kp = kp.reshape(-1, 2)
        if kps_score is not None:
            # print(kps_score.shape)
            kp_score = kps_score[i]

        for i in range(len(skeleton)):
            idx_0 = skeleton[i][0] - 1
            idx_1 = skeleton[i][1] - 1
            limb_color = (int(pose_limb_color[i][0]), int(pose_limb_color[i][1]), int(pose_limb_color[i][2]))
            # if maxval[idx_0] > self.conf and maxval[idx_1] > self.conf:
            cv2.line(img, (int(kp[idx_0][0]), int(kp[idx_0][1])), (int(kp[idx_1][0]),
                     int(kp[idx_1][1])), limb_color, 2)
    return img
