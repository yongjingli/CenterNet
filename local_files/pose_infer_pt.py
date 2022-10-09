import sys
import os
import torch
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, "/userdata/liyj/programs/CenterNet/src/lib")
from models.model import create_model, load_model, save_model
from opts import opts
from models.decode import multi_pose_decode

from local_utils import draw_boxes_in_cv_img
from local_utils import draw_kps_in_cv_img
from local_utils import draw_skeleton_in_cv_img


class PoseInferPt(object):
    def __init__(self, opt):
        self.opt = opt

        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)

        self.model.cuda()
        self.model.eval()

        self.down_scale = 4
        self.num_joints = 17

        # img_preprocess setting
        self.input_w = opt.input_w
        self.input_h = opt.input_h

        self.sub_mean = True
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                        dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                       dtype=np.float32).reshape(1, 1, 3)

        self.trans_input_inv = None
        self.transform_info = None

        # debug
        self.inp_resize = None
        self.img_show = None
        self.output = None

    def infer(self, img):
        inp = self.img_preprocess(img)
        dets = self.process(inp)
        dets = self.pose_process(dets[0])

        outputs = self.scale_boxes(dets)
        return outputs

    def img_preprocess(self, img):
        inp_resize = self.img_trasform(img)
        self.img_show = inp_resize
        self.inp_resize = inp_resize

        inp = (inp_resize.astype(np.float32) / 255.)

        mean = self.mean
        std = self.std

        if self.sub_mean:
            inp = (inp - mean) / std

        inp = inp.transpose(2, 0, 1)
        inp = np.expand_dims(inp, axis=0)
        inp = torch.from_numpy(inp)
        return inp

    def img_trasform(self, img):
        input_size = [img.shape[0], img.shape[1]]
        output_size = [self.input_h, self.input_w]

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)

        # compute scale
        resize_scale = min(output_size[0] / input_size[0], output_size[1] / input_size[1])
        pad_scale = min(output_size[0] / input_size[0], output_size[1] / input_size[1])
        pad_h = (output_size[0] - input_size[0] * pad_scale) // 2
        pad_w = (output_size[1] - input_size[1] * pad_scale) // 2

        pad_top, pad_bottom = pad_h, pad_h
        pad_left, pad_right = pad_w, pad_w

        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        scale = np.array([1.0, 1.0])
        shift = np.array([0.0, 0.0])
        src[0, :] = center + shift * input_size
        src[1, :] = src[0, :] + np.array([0, -input_size[1] / 2 * scale[1]], dtype=np.float32)
        src[2, :] = src[0, :] + np.array([input_size[0] / 2 * scale[0], 0], dtype=np.float32)

        dst[0, :] = [output_size[1] * 0.5, output_size[0] * 0.5]
        dst[1, :] = dst[0, :] + np.array([0, -input_size[1] / 2], dtype=np.float32) * resize_scale
        dst[2, :] = dst[0, :] + np.array([input_size[0] / 2, 0], dtype=np.float32) * resize_scale

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inp_resize = cv2.warpAffine(img, trans, (output_size[1], output_size[0]), flags=cv2.INTER_LINEAR)

        self.transform_info = [pad_scale, pad_top, pad_bottom, pad_left, pad_right]
        return inp_resize

    def process(self, inp):
        inp = inp.cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(inp)[-1]
            output['hm'] = output['hm'].sigmoid_()

            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None

            torch.cuda.synchronize()

            self.output = output
            dets = multi_pose_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        return dets

    def pose_process(self, dets):
        # mask = dets[:, 4] > self.opt.vis_thresh
        mask = dets[:, 4] > 0.2
        dets = dets[mask]

        dets[:, :4] = dets[:, :4] * self.down_scale
        dets[:, 5:39] = dets[:, 5:39] * self.down_scale

        return dets

    def scale_boxes(self, dets):
        scale, pad_top, pad_bottom, pad_left, pad_right = self.transform_info
        dets[:, 0] = (dets[:, 0] - pad_left) / scale
        dets[:, 2] = (dets[:, 2] - pad_left) / scale

        dets[:, 1] = (dets[:, 1] - pad_top) / scale
        dets[:, 3] = (dets[:, 3] - pad_top) / scale

        dets[:, 5:39:2] = (dets[:, 5:39:2] - pad_left) / scale
        dets[:, 6:39:2] = (dets[:, 6:39:2] - pad_top) / scale
        return dets


def show_infer_imgs(infer_model):
    #  proc img data
    src_root = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color"
    dst_root = "/userdata/liyj/data/test_data/depth/debug"

    if dst_root is not None:
        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
        os.mkdir(dst_root)

    img_names = [name for name in os.listdir(src_root) if name.split('.')[-1] in ['png', 'jpg']]
    for img_name in tqdm(img_names):
        # img_name = "image_23_1661767190924.png"
        img_path = os.path.join(src_root, img_name)
        dst_img_path = os.path.join(dst_root, img_name.replace(".png", ".jpg"))

        # img_path = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color/image_157_1662709290128.png"

        img = cv2.imread(img_path)
        img = img[:, :1920, :]

        dets = Pose_Infer_Pt.infer(img)

        kps = dets[:, 5:39]
        kps_score = dets[:, 39:56]
        #
        convert_det_boxes = dets[:, :4]
        convert_det_boxes[:, 2:4] = convert_det_boxes[:, 2:4] - convert_det_boxes[:, 0:2]

        img = draw_boxes_in_cv_img(img, convert_det_boxes, color=(0, 255, 0), thinkness=2)
        img = draw_kps_in_cv_img(img, kps, color=(0, 255, 0), radius=5)
        img = draw_skeleton_in_cv_img(img, kps, kps_score=kps_score)

        if dst_root is not None:
            save_img_path = os.path.join(dst_root, img_name.replace('.png', '.jpg'))
            cv2.imwrite(save_img_path, img)
            # exit(1)
        else:
            plt.imshow(img[:, :, ::-1])
            plt.show()


if __name__ == '__main__':
    print("Start....")
    opt = opts().init()

    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_tmp/pose_0916.pth"
    print("Load Model....")
    Pose_Infer_Pt = PoseInferPt(opt)
    show_infer_imgs(Pose_Infer_Pt)
    print("eND....")




