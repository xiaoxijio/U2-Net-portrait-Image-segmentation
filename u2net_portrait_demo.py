import cv2
import torch
from model.u2net import U2NET
from torch.autograd import Variable
import numpy as np
from glob import glob
import os


def detect_single_face(face_cascade, img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # 检测所有可能的人脸 返回一个包含 (x, y, w, h) 的列表
    if len(faces) == 0:
        print("Warming: no face detection, the portrait u2net will run on the whole image!")
        return None

    # filter to keep the largest face
    wh = 0
    idx = 0
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        if wh < w * h:  # 选择最大人脸
            idx = i
            wh = w * h

    return faces[idx]


# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):
    # no face detected, return the whole image and the inference will run on the whole image
    if face is None:
        return img
    (x, y, w, h) = face

    height, width = img.shape[0:2]
    # 通过人脸的边界框 (x, y, w, h) 裁剪图像，但会为裁剪区域添加一定的边距
    l, r, t, b = 0, 0, 0, 0
    lpad = int(float(w) * 0.4)  # 左边距 (lpad): 人脸宽度的 40%
    left = x - lpad
    if left < 0:
        l = lpad - x
        left = 0

    rpad = int(float(w) * 0.4)  # 右边距 (rpad): 人脸宽度的 40%
    right = x + w + rpad
    if right > width:
        r = right - width
        right = width

    tpad = int(float(h) * 0.6)  # 上边距 (tpad): 人脸高度的 60%
    top = y - tpad
    if top < 0:
        t = tpad - y
        top = 0

    bpad = int(float(h) * 0.2)  # 下边距 (bpad): 人脸高度的 20%
    bottom = y + h + bpad
    if bottom > height:
        b = bottom - height
        bottom = height

    im_face = img[top:bottom, left:right]
    if len(im_face.shape) == 2:
        im_face = np.repeat(im_face[:, :, np.newaxis], (1, 1, 3))
    # 如果越界区域需要填充，使用常数值填充，值为 255（白色）
    im_face = np.pad(im_face, ((t, b), (l, r), (0, 0)), mode='constant',
                     constant_values=((255, 255), (255, 255), (255, 255)))

    # 如果裁剪后的人脸区域不是正方形，通过在较小的一边（宽度或高度）补齐，保持人脸比例不变
    hf, wf = im_face.shape[0:2]
    if hf - 2 > wf:
        wfp = int((hf - wf) / 2)
        im_face = np.pad(im_face, ((0, 0), (wfp, wfp), (0, 0)), mode='constant',
                         constant_values=((255, 255), (255, 255), (255, 255)))
    elif wf - 2 > hf:
        hfp = int((wf - hf) / 2)
        im_face = np.pad(im_face, ((hfp, hfp), (0, 0), (0, 0)), mode='constant',
                         constant_values=((255, 255), (255, 255), (255, 255)))

    # resize to have 512x512 resolution
    im_face = cv2.resize(im_face, (512, 512), interpolation=cv2.INTER_AREA)

    return im_face


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def inference(net, input):
    # normalize the input
    tmpImg = np.zeros((input.shape[0], input.shape[1], 3))
    input = input / np.max(input)

    tmpImg[:, :, 0] = (input[:, :, 2] - 0.406) / 0.225
    tmpImg[:, :, 1] = (input[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (input[:, :, 0] - 0.485) / 0.229

    # convert BGR to RGB
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis, :, :, :]
    tmpImg = torch.from_numpy(tmpImg)

    # convert numpy array to torch tensor
    tmpImg = tmpImg.type(torch.FloatTensor)

    if torch.cuda.is_available():
        tmpImg = Variable(tmpImg.cuda())
    else:
        tmpImg = Variable(tmpImg)

    # inference
    d1, d2, d3, d4, d5, d6, d7 = net(tmpImg)

    # normalization
    pred = 1.0 - d1[:, 0, :, :]
    pred = normPRED(pred)

    # convert torch tensor to numpy array
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    del d1, d2, d3, d4, d5, d6, d7

    return pred


def main():
    # get the image path list for inference
    im_list = glob('./test_data/test_portrait_images/your_portrait_im/*')
    print("Number of images: ", len(im_list))
    # indicate the output directory
    out_dir = './test_data/test_portrait_images/your_portrait_results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load the cascade face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # u2net_portrait path
    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

    # load u2net_portrait model
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # do the inference one-by-one
    for i in range(len(im_list)):
        print("--------------------------")
        print("inferencing ", i+1, "/", len(im_list), im_list[i])

        # load each image
        img = cv2.imread(im_list[i])
        face = detect_single_face(face_cascade, img)   # 检测人脸，并选择一张最大的脸返回
        im_face = crop_face(img, face)  # 裁剪一下，最后输出正方形
        im_portrait = inference(net, im_face)  # 带到模型里

        # save the output
        cv2.imwrite(out_dir + "/" + im_list[i].split(os.sep)[-1], (im_portrait * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
