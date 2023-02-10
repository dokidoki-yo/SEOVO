import os
import time
import cv2
from skimage import segmentation
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
swap1_last = None

def localStd(img):
    # 归一化
    # img = img / 255.0
    # 计算均值图像和均值图像的平方图像
    img_blur = cv.blur(img, (21, 21))
    reslut_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    reslut_2 = cv.blur(img_2, (21, 21))

    reslut = np.sqrt(np.maximum(reslut_2 - reslut_1, 0))
    return reslut


def get_reflect(img, img_illumination):
    # get_img_illumination = get_illumination(img)
    get_img_reflect = (img + 0.001) / (img_illumination + 0.001)
    return get_img_reflect


def enhancement_reflect(img):
    # 通过高斯滤波器
    gaussian_blur_img = cv.GaussianBlur(img, (21, 21), 0)
    enhancement_reflect_img = img * gaussian_blur_img
    return enhancement_reflect_img


def get_enhancment_img(img_enhance_illumination, img_enahnce_reflect):
    img = img_enhance_illumination * img_enahnce_reflect
    img = img.astype('uint8')
    return img


def read_img_from_disk(file_path):
    # 0. 读取图像
    img = cv.imread(file_path, cv.IMREAD_COLOR)
    return img


def get_illumination(img):
    return cv.GaussianBlur(img, (15, 15), 0)


"""
enhancment_illumination 增强反射分量，传入反射分量，返回增强后的反射分量
"""


def enhancment_illumination(img_illumination):
    img_hsv = cv.cvtColor(img_illumination, cv.COLOR_BGR2HSV)
    img_hsv = (img_hsv - np.min(img_hsv)) / (np.max(img_hsv) - np.min(img_hsv))
    h, s, v = cv.split(img_hsv)
    wsd = 5
    gm = np.mean(v) / (1 + wsd * np.std(v)) # 一个数字
    cst = localStd(v)   # 300 * 400 的矩阵
    lm = gm * v /(1 + wsd * cst)    # 300 * 400 的矩阵
    c = np.exp(gm)      # 一个常数
    wg = v ** 0.2       # 300 *400
    wl = 1- wg
    outM = v**c / (v**c +(wl * lm)**c + (wg * gm)**c + 0.001)
    outM = 1.5 * outM - 0.5 * cv.GaussianBlur(outM, (21, 21), 0)
    outM = (outM - np.min(outM))/(np.max(outM) - np.min(outM))
    paramerter = 0.9
    img_illumination[:, :, 0] = outM * (img_illumination[:, :, 0] / (v + 0.01))**paramerter
    img_illumination[:, :, 1] = outM * (img_illumination[:, :, 1] / (v + 0.01))**paramerter
    img_illumination[:, :, 2] = outM * (img_illumination[:, :, 2] / (v + 0.01))**paramerter
    return img_illumination

class Args(object):
    # input_image_path = 'image/woof.jpg'  # image/coral.jpg image/tiger.jpg
    input_image_path = 'image/color_15.png'
    train_epoch = 2 ** 6
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def unevenLightCompensate(gray, blockSize):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)

            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst[dst > 255] = 255
    dst[dst < 0] = 0
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

def run(image, i):
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = cv2.resize(image, (320, 256))

    # modify the illumination of the image
    img_illumination = get_illumination(image)  # 获得高频分量
    img_reflect = get_reflect(image, img_illumination)  # 获得反射分量
    img_enhancement_reflect = enhancement_reflect(img_reflect)  # 增强反射分量
    img_enhancement_illumination = enhancment_illumination(img_illumination)  # 增强照射分量
    img_en = get_enhancment_img(img_enhancement_illumination, img_reflect)  # 照射分量与反射分量融合

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(img_en, scale=64, sigma=0.1, min_size=128)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)] # seg_lab[0]: indexs of pixel that belongs to class 0

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = img_en.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)


    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = img_en.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3)) # produce random nums between 0~255 with size of max_label_num*3
    show = img_en

    '''train loop'''
    start_time1 = time.time()
    model.train()

    return_img_seg = seg_map
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy() # h*w
        # print('img_target_shape', im_target.shape)


        '''refine'''
        for inds in seg_lab: # for indexs of each class
            # print('inds', inds)
            # print(im_target[inds].shape)
            u_labels, hist = np.unique(im_target[inds], return_counts=True) #hist: the num of  u_label in im_target
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)

        loss = criterion(output, target)

        loss_total = loss

        loss_total.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        # print(len(un_label))
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(img_en.shape)
        return_img_seg = im_target
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    # cv2.imshow('seg', show)
    # cv2.imshow('src', img_done)
    # cv2.waitKey(0)
    # print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    # print('write_path', "seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1))
    # cv2.imwrite("seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1), show)
    # cv2.imwrite(
    #     '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/seg_result/process/' + str(i) + '.png', show)
    # cv2.imwrite("seg_%s_%ds.jpg" % (args.input_image_path[6:-4], time1), show)
    # print(return_img_seg.shape)
    # stop
    return return_img_seg, show, img_en

def line_detect(grad,image,show):
    seg_img = image

    # 将图片转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 设置阈值
    lowera = np.array([0, 0, 221])
    uppera = np.array([180, 30, 255])
    mask1 = cv2.inRange(hsv, lowera, uppera)
    kernel = np.ones((3, 3), np.uint8)

    # 对得到的图像进行形态学操作（闭运算和开运算）
    mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel) #闭运算：表示先进行膨胀操作，再进行腐蚀操作
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  #开运算：表示的是先进行腐蚀，再进行膨胀操作

    img_illumination = get_illumination(image)  # 获得高频分量
    img_reflect = get_reflect(image, img_illumination)  # 获得反射分量
    img_enhancement_reflect = enhancement_reflect(img_reflect)  # 增强反射分量
    img_enhancement_illumination = enhancment_illumination(img_illumination)  # 增强照射分量
    img_done = get_enhancment_img(img_enhancement_illumination, img_reflect)  # 照射分量与反射分量融合

    # image = cv.GaussianBlur(image, (3, 3), 15)
    # cv2.imshow('mask', image)

    # 绘制轮廓
    # edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # print(edges.shape) #256,320

    # edges_norm = cv2.Canny(normal_seg, 50, 150, apertureSize=3)
    # 显示图片
    # cv2.imshow("edges", edges_norm)
    # cv2.waitKey(0)
    # 检测白线  这里是设置检测直线的条件，可以去读一读HoughLinesP()函数，然后根据自己的要求设置检测条件
    lines = cv2.HoughLinesP(grad, 1, np.pi / 180, 30, minLineLength=30,maxLineGap=10)
    image_copy1 = image.copy()
    image_copy2 = image.copy()
    lines_return = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            mid_pointy = int((y1 + y2)/2)
            mid_pointx = int((x1 + x2)/2)
            condition1 = x1 > 7 and x1<313 and x2 > 7 and x2<313 and y1 > 7 and y1<248 and y2 > 7 and y2<248
            cv2.line(image_copy1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # if show:
            #     cv2.imshow('current_line', image)
            #     cv2.waitKey(0)
            if x1==x2:
                angle = 90
                rad = np.pi/2.
            else:
                # print(x1,y1,x2,y2)
                k = -(y2 - y1) / (x2 - x1)
                # #       # 求反正切，再将得到的弧度转换为度
                # #       result = np.arctan(k) * 57.29577
                # print('k',k)
                angle = np.arctan(k)* 57.29577
                rad = np.arctan(k)
            # print('angle',angle)
            # print('radiance',rad)
            # x1_tmp = x1
            # y1_tmp = y1
            # x2_tmp = x2
            # y2_tmp = y2
            # if angle <= 90:
            #     if x1 >= x2:
            #         x1_tmp = x1
            #         y1_tmp = y1
            #         x2_tmp = x2
            #         y2_tmp = y2
            #     else:
            #         x1_tmp = x2
            #         y1_tmp = y2
            #         x2_tmp = x1
            #         y2_tmp = y1

            x_hat = 15*np.sin(rad)
            y_hat = 15*np.cos(rad)
            # print('np.sin(angle)',np.sin(angle))
            # print('np.sin(angle)',np.sin(rad))
            # print(np.cos(rad))
            # print(x_hat)
            # print(y_hat)

            x1_rect = x1-x_hat
            y1_rect = y1-y_hat
            x2_rect = x1 + x_hat
            y2_rect = y1 + y_hat
            x3_rect = x2 - x_hat
            y3_rect = y2 - y_hat
            x4_rect = x2 + x_hat
            y4_rect = y2 + y_hat

            pts1 = []
            pts2 = []
            pts1.append((x1_rect, y1_rect))
            pts1.append((x1, y1))
            pts1.append((x2, y2))
            pts1.append((x3_rect, y3_rect))
            pts2.append((x2_rect, y2_rect))
            pts2.append((x4_rect, y4_rect))
            pts2.append((x2, y2))
            pts2.append((x1, y1))
            pts1_array = np.array(pts1, np.int32)
            pts2_array = np.array(pts2, np.int32)
            # cv2.polylines(image, [pts1_array],True,
            #                      (0,255,0),3)
            # cv2.polylines(image, [pts2_array], True,
            #                      (0, 0, 255), 3)
            # cv2.imshow('rect_img',image)
            # cv2.waitKey(0)
            # print(img_gray.shape)

            mask_zero = np.zeros([256,320])
            mask_rect1 = cv2.fillPoly(mask_zero.copy(), [pts1_array], (255, 255,255))
            mask_rect2 = cv2.fillPoly(mask_zero.copy(), [pts2_array], (255, 255,255))
            # print(mask_rect.shape)
            roi1 = cv2.bitwise_and(img_gray.copy(),img_gray.copy(), mask=mask_rect1.astype(np.uint8))
            roi2 = cv2.bitwise_and(img_gray.copy(),img_gray.copy(), mask=mask_rect2.astype(np.uint8))
            rect1_ratio = np.sum(roi1>0)/np.sum(mask_rect1>0)
            rect2_ratio = np.sum(roi2>0)/np.sum(mask_rect2>0)
            # print(rect_ratio)
            # cv2.imshow('rect_img_roi1',roi1)
            # cv2.imshow('rect_img_roi2',roi2)
            # cv2.waitKey(0)
            # print('coord',x1,x2,y1,y2)

            if condition1 and rect1_ratio > 0.7 and rect2_ratio > 0.7 and (angle>70 and angle<90):
                # print(grad[mid_pointy, mid_pointx-5])
                # print(grad[mid_pointy, mid_pointx+5])
                # print(grad[mid_pointy+5, mid_pointx])
                # print(grad[mid_pointy-5, mid_pointx])

                # condition3 = (img_gray[mid_pointy, mid_pointx-7] != 0)
                #          and (img_gray[mid_pointy, mid_pointx+7] != 0)
                #              and((img_gray[mid_pointy+7, mid_pointx]!= 0) and
                #              (img_gray[mid_pointy-7, mid_pointx] != 0))
                # print(grad[mid_pointy, mid_pointx-3] == 0)
                # print(grad[mid_pointy, mid_pointx-3] == 0)
                # # print()
                # condition4 = grad[mid_pointy, mid_pointx-3] == 0 and grad[mid_pointy, mid_pointx+3] == 0 \
                                # and grad[mid_pointy-3, mid_pointx] == 0 and grad[mid_pointy+3, mid_pointx] == 0
                # if condition3:
                cv2.line(image_copy2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                lines_return.append(line)

                # if condition3 and condition4:
                #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     cv2.imshow('lines',image)
                #     cv2.waitKey(0)
        # if show:
        #     # cv2.imshow('ori_lines', image_copy1)
    cv2.imshow('lines', image_copy2)
    cv2.waitKey(0)
    return lines_return

    #
    #     # lines_norm = cv2.HoughLinesP(edges_norm, 1, np.pi / 180, 30, minLineLength=15,maxLineGap=10)
    # # print ("lines=",lines)
    # # print ("========================================================")
    # i=1
    # x_min = -1
    # x_max = -1
    # y_min = -1
    # y_max = -1
    # if box is not None:
    #     # print(box)
    #     # print([0])
    #     x_min = box[1]
    #     y_min = box[0]
    #     x_max = box[3]
    #     y_max = box[2]
    # # 对通过霍夫变换得到的数据进行遍历
    # lines_return = []
    # # for line in lines:
    # #     # newlines1 = lines[:, 0, :]
    # #     print( "line["+str(i-1)+"]=",line)
    # #     x1,y1,x2,y2 = line[0]  #两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
    # #     # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    # #     # print(image.shape)
    # #     # print('xyxy',x1, y1, x2, y2)
    # #     # cv2.imshow('line', image)
    # #     # cv2.waitKey(0)
    # #     image = image.copy()
    # #     # 转换为浮点数，计算斜率
    # #     x1 = float(x1)
    # #     x2 = float(x2)
    # #     y1 = float(y1)
    # #     y2 = float(y2)
    # #     print ("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
    # #     if x2 - x1 == 0:
    # #       print( "直线是竖直的")
    # #       result=90
    # #     elif y2 - y1 == 0 :
    # #       print ("直线是水平的")
    # #       result=0
    # #     else:
    # #       # 计算斜率
    # #       k = -(y2 - y1) / (x2 - x1)
    # #       # 求反正切，再将得到的弧度转换为度
    # #       result = np.arctan(k) * 57.29577
    # #       print("直线倾斜角度为：" + str(result) + "度")
    # #     if np.abs(result) > 80 and np.abs(result) <= 90:
    # #         x1 = int(x1)
    # #         x2 = int(x2)
    # #         y1 = int(y1)
    # #         y2 = int(y2)
    # #         condition1 = (seg_img[y1, x1][0] != 0 and seg_img[y1, x1][1] != 0 and seg_img[y1, x1][2] != 0) or \
    # #                 (seg_img[y2, x2][0] != 0 and seg_img[y2, x2][1] != 0 and seg_img[y2, x2][2] != 0 )
    # #         # condition2 = np.abs(np.min([x1, x2])-y_min.cpu().detach().numpy())>5 and np.abs(np.max([x1, x2]) - y_max.cpu().detach().numpy()) > 5
    # #         #
    # #         # print('x_max', x_max)
    # #         # print('y1', y1, y2)
    # #         if condition1:
    # #             cv2.line(seg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    # #   显示最后的成果图
    # # cv2.imshow("line_detect",seg_img)
    # # normal_seg_copy = normal_seg.copy()
    # # normal_seg_copy = image.copy()
    # # normal_seg = image.copy()
    #
    # for line in lines:
    #     # newlines1 = lines[:, 0, :]
    #     # print("line[" + str(i - 1) + "]=", line)
    #     x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
    #     # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    #     # print(image.shape)
    #     # print('xyxy',x1, y1, x2, y2)
    #     # cv2.imshow('line', image)
    #     # cv2.waitKey(0)
    #     # normal_seg = normal_seg.copy()
    #     # 转换为浮点数，计算斜率
    #     x1 = float(x1)
    #     x2 = float(x2)
    #     y1 = float(y1)
    #     y2 = float(y2)
    #     # print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
    #     if x2 - x1 == 0:
    #         # print("直线是竖直的")
    #         result = 90
    #     elif y2 - y1 == 0:
    #         # print("直线是水平的")
    #         result = 0
    #     else:
    #         # 计算斜率
    #         k = -(y2 - y1) / (x2 - x1)
    #         # 求反正切，再将得到的弧度转换为度
    #         result = np.arctan(k) * 57.29577
    #         # print("直线倾斜角度为：" + str(result) + "度")
    #     # normal_seg = normal_seg.copy()
    #
    #     if np.abs(result) > 75 and np.abs(result) <= 90:
    #         # x1 = int(x1)
    #         # x2 = int(x2)
    #         # y1 = int(y1)
    #         # y2 = int(y2)
    #         x1 = int(x1)
    #         x2 = int(x2)
    #         y1 = int(y1)
    #         y2 = int(y2)
    #         # cv2.line(normal_seg, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         mid_pointx = int(0.5*(x1+x2))
    #         mid_pointy = int(0.5*(y1+y2))
    #         # print('x1x2',x1,x2)
    #         # print('x1x2_min',x_min,x_max)
    #         # if x_min > 0 and x_max > 0:
    #         #     condition3 = np.min([x1,x2]) > (x_min+5) and np.max([x1,x2]) < (x_max-5)
    #         # else:
    #         condition3 = np.min([x1,x2]) > np.max([0, x_min]) + 3 and np.max([x1,x2]) < np.min([x_max, 319])-3
    #
    #         write_flag = True
    #         if len(lines_return) > 0:
    #             for idx in range(len(lines_return)):
    #                 line_compare = lines_return[idx]
    #                 x1_comp, y1_comp, x2_comp, y2_comp = line_compare[0]
    #                 # print('x1-x1_comp',x1,x1_comp,x2,x2_comp)
    #                 if np.abs(x1_comp-x1) < 6 and np.abs(x2_comp-x2) < 6 :
    #                     write_flag = False
    #         # print(condition3,write_flag)
    #         # if not write_flag:
    #         #     continue
    #         # condition3 = (normal_seg[mid_pointy, mid_pointx-3][0] != 0 or normal_seg[mid_pointy, mid_pointx-3][1] != 0 or normal_seg[mid_pointy, mid_pointx-3][2] != 0) \
    #         #              and (normal_seg[mid_pointy, mid_pointx+3][0] != 0 or normal_seg[mid_pointy, mid_pointx+3][1] != 0 or normal_seg[mid_pointy, mid_pointx+3][2] != 0)
    #
    #         # condition2 = np.abs(np.min([x1, x2])-y_min.cpu().detach().numpy())>5 and np.abs(np.max([x1, x2]) - y_max.cpu().detach().numpy()) > 5
    #         #
    #         # print('x_max', x_max)
    #         # print('y1', y1, y2)
    #         # print('condition1', condition1)
    #         # print('condition3', condition3)
    #
    #         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    #
    #         #
    #         # if condition3 and write_flag:
    #         #     cv2.line(normal_seg_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    #         #     lines_return.append(line)
    #
    #
    #         if condition3 and write_flag:
    #         #     if (normal_seg[mid_pointy, mid_pointx-10][0] != 0 or normal_seg[mid_pointy, mid_pointx-10][1] != 0 or normal_seg[mid_pointy, mid_pointx-10][2] != 0) \
    #         #             and (normal_seg[mid_pointy, mid_pointx+10][0] != 0 or normal_seg[mid_pointy, mid_pointx+10][1] != 0 or normal_seg[mid_pointy, mid_pointx+10][2] != 0):
    #             condition4 = np.max(
    #                 (image[mid_pointy, mid_pointx - 3] - image[mid_pointy, mid_pointx + 3]) ** 2) < 100
    #             # condition1 = (normal_seg[y1, x1 - 3][0] != 0 and normal_seg[y1, x1 - 3][1] != 0 and normal_seg[y1, x1 - 3][
    #             #     2] != 0) and \
    #             #              (normal_seg[y1, x1 + 3][0] != 0 and normal_seg[y1, x1 + 3][1] != 0 and normal_seg[y1, x1 + 3][
    #             #                  2] != 0)
    #             # condition1 = np.min(normal_seg[y1, x1 - 5])!=0 and np.min(normal_seg[y1, x1 + 5])!=0 \
    #             #              and np.min(normal_seg[y2, x2 - 5])!=0 and np.min(normal_seg[y2, x2 + 5])!=0
    #             # print('boudary pixel', normal_seg[y1, x1 - 5], normal_seg[y1, x1 + 5],normal_seg[y2, x2 + 5],normal_seg[y2, x2 + 5], )
    #             # print('condition4', image[mid_pointy, mid_pointx - 3] - image[mid_pointy, mid_pointx + 3])
    #             # condition2 =
    #             # if condition1:
    #             # print('condition1', condition1)
    #             cv2.line(normal_seg_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    #             lines_return.append(line)
    #                 # cv2.imshow('one draw',normal_seg_copy)
    #                 # cv2.waitKey(0)
    #                 # lines_return.append(line)
    #
    #     #   显示最后的成果图
    # # print(normal_seg.shape)
    # # if len(normal_seg.shape) == 2:
    # #
    # # cv2.imshow("seg_line_detect", normal_seg)
    # # cv2.imshow("seg_line_detect_after", normal_seg_copy)
    # # cv2.waitKey(0)
    # fianl_return = lines_return
    # #
    # # if len(lines_return) > 1:
    # #     d_max = 0
    # #     for line_idx in range(len(lines_return)):
    # #         curr_line = lines_return[line_idx]
    # #         x1,y1,x2,y2 = lines_return[line_idx][0]
    # #         print(x1, x2, y1,y2)
    # #         line_dis = (x1-x2)**2 + (y1-y2)**2
    # #         if line_dis > d_max:
    # #             d_max = line_dis
    # #             fianl_return = [curr_line]
    # # print('len_fianl', len(fianl_return))
    # # fianl_img = normal_seg.copy()
    # # if len(fianl_return) > 0:
    # #     # print(fianl_return[0][0])
    # #     x1, y1, x2, y2 = fianl_return[0][0]
    # #     cv2.line(fianl_img,(x1,y1),(x2, y2), (0, 0, 255), 2)
    # # cv2.imshow('final',fianl_img)
    # return result, seg_img, lines_return, normal_seg_copy




