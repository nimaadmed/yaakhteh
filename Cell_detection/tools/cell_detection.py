#  Nimaad leading in modern medical diagnosis
#  Demo of cell localization and detection
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../tools')
import os
from torchvision import transforms
import torch.nn.functional as F
import  tkinter as tk
from tkinter import *
from tkinter import ttk
import time
import cv2
from PIL import Image, ImageTk, ImageDraw
from io import BytesIO
import torch
import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import im_detect
from model.nms_wrapper import nms
from model.test import im_detect
from model.nms_wrapper import nms
import numpy as np
import tensorflow as tf
from nets.vgg16 import vgg16

class Cell_App:
    def __init__(self):
        self.root = Tk()
        self.widthpixels = 1500
        self.heightpixels = 800
        self.root.wm_title("Cell Counter Application")
        self.C=[0,0,0,0,0]
        self.prev_C=[0,0,0,0,0]
        self.root.geometry('{}x{}'.format(self.widthpixels, self.heightpixels))
        self.button_frame=Frame(self.root)
        self.button_frame.pack(side=LEFT,fill=Y)
        self.img_frame=Frame(self.root)
        self.img_frame.pack(side=RIGHT)

        labelframe1 = LabelFrame(self.button_frame,text='start panel')
        labelframe1.pack(fill="both", expand="yes")
        self.start_button=Button(labelframe1,text='start', height = 2, width = 10, command=self.start_win)
        self.start_button.grid(row=0,column=0,pady=10)
        self.zoom_button=Button(labelframe1,text='Focus', height = 2, width = 10,command=lambda index=7 : self.auto_zoom(index))
        self.zoom_button.grid(row=0,column=1)
        self.zoom_button=Button(labelframe1,text='Fast Focus', height = 2, width = 10,command=lambda index=3 : self.auto_zoom(index))
        self.zoom_button.grid(row=0,column=2)

        labelframe2= LabelFrame(self.button_frame,text='move panel')
        labelframe2.pack(fill="both", expand="yes")
        step_label=Label(labelframe2,text='step size', height=1, width=10)
        step_label.grid(row=0, column=0,pady=10)
        self.step_text=Text(labelframe2, height=1, width=5)
        self.step_text.grid(row=0, column=1,pady=10)
        dum = Label(labelframe2, text='  ', height=3, width=10)
        dum.grid(row=1, column=0,pady=10)
        dum= Label(labelframe2, text='  ', height=3, width=3)
        dum.grid(row=1, column=1)
        img_up = Image.open("../pics/up.jpeg")
        photo_up = ImageTk.PhotoImage(img_up)
        self.up_button = Button(labelframe2, image=photo_up,height=50, width=50,command=self.move_up)
        self.up_button.grid(row=1, column=2)
        dum= Label(labelframe2, text='  ', height=3, width=3)
        dum.grid(row=1, column=3)

        dum = Label(labelframe2, text='  ', height=3, width=10)
        dum.grid(row=3, column=0)
        dum= Label(labelframe2, text='  ', height=3, width=3)
        dum.grid(row=3, column=1)
        img_down = Image.open("../pics/down.jpeg")
        photo_down = ImageTk.PhotoImage(img_down)
        self.down_button = Button(labelframe2, image=photo_down,height=50, width=50,command=self.move_down)
        self.down_button.grid(row=3, column=2)
        dum= Label(labelframe2, text='  ', height=3, width=3)
        dum.grid(row=3, column=3)

        dum = Label(labelframe2, text='  ', height=3, width=10)
        dum.grid(row=2, column=0)
        img_left = Image.open("../pics/left.jpeg")
        photo_left = ImageTk.PhotoImage(img_left)
        self.left_button = Button(labelframe2,image=photo_left,height=50, width=50,command=self.move_left)
        self.left_button.grid(row=2, column=1)
        dum = Label(labelframe2, text='  ', height=3, width=3)
        dum.grid(row=2, column=2)
        img_right = Image.open("../pics/right.jpeg")
        photo_right = ImageTk.PhotoImage(img_right)
        self.right_button = Button(labelframe2, image=photo_right, height=50, width=50,command=self.move_right)
        self.right_button.grid(row=2, column=3)

        labelframe3 = LabelFrame(self.button_frame,text='count panel')
        labelframe3.pack(fill="both", expand="yes")
        self.count_button=Button(labelframe3,text='Count', height = 2, width = 10,command=self.count)
        self.count_button.grid(row=0,column=0,pady=10)
        self.undo_count_button=Button(labelframe3,text='Undo Count', height = 2, width = 10,command=self.undo_count)
        self.undo_count_button.grid(row=0,column=1)
        self.edit_count_button=Button(labelframe3,text='Edit manually', height = 2, width = 10,command=self.edit_manually)
        self.edit_count_button.grid(row=0,column=2)


        temp=ttk.Separator(labelframe3,orient=HORIZONTAL)
        temp.grid(row=1,column=0,pady=15,sticky="ew")
        temp = tk.Label(labelframe3, text='Results', height=3, width=10)
        temp.grid(row=1,column=1,pady=10)
        temp=ttk.Separator(labelframe3,orient=HORIZONTAL)
        temp.grid(row=1,column=2,pady=10,sticky="ew")
        self.C_label=[]
        i=0
        self.count1_label=tk.Button(labelframe3,text='Neutrophil', height = 2, width = 10,command=lambda index=i : self.single_classifier(index))
        self.count1_label.grid(row=2, column=0)
        self.C_label.append(Entry(labelframe3,width=5,font="Helvetica 20 bold",highlightcolor='white'))
        self.C_label[0].insert(0,'0')
        self.C_label[0].grid(row=2, column=1)
        self.C_label[0].config(state='disabled')

        i=1
        self.count2_label=tk.Button(labelframe3,text='Lymph', height = 2, width = 10,command=lambda index=i : self.single_classifier(index))
        self.count2_label.grid(row=3, column=0)
        self.C_label.append(Entry(labelframe3,width=5,font="Helvetica 20 bold",highlightcolor='white'))
        self.C_label[1].insert(0, '0')
        self.C_label[1].grid(row=3, column=1)
        self.C_label[1].config(state='disabled')

        i=2
        self.count3_label=tk.Button(labelframe3,text='Monocyte', height = 2, width = 10,command=lambda index=i : self.single_classifier(index))
        self.count3_label.grid(row=4, column=0)
        self.C_label.append(Entry(labelframe3,width=5,font="Helvetica 20 bold",highlightcolor='white'))
        self.C_label[2].insert(0, '0')
        self.C_label[2].grid(row=4, column=1)
        self.C_label[2].config(state='disabled')

        i=3
        self.count4_label=tk.Button(labelframe3,text='Eosinophil', height = 2, width = 10,command=lambda index=i : self.single_classifier(index))
        self.count4_label.grid(row=5, column=0)
        self.C_label.append(Entry(labelframe3,width=5,font="Helvetica 20 bold",highlightcolor='white'))
        self.C_label[3].insert(0, '0')
        self.C_label[3].grid(row=5, column=1)
        self.C_label[3].config(state='disabled')

        count5_label=tk.Button(labelframe3,text='Basophil', height = 2, width = 10)
        count5_label.grid(row=6, column=0)
        self.C_label.append(Entry(labelframe3,width=5,font="Helvetica 20 bold",highlightcolor='white'))
        self.C_label[4].insert(0, '0')
        self.C_label[4].grid(row=6, column=1)
        self.C_label[4].config(state='disabled')

        self.apply=Button(labelframe3,text='Apply Changes', height = 2, width = 10,command=self.apply_changes)
        self.apply.grid(row=7, column=1)
        self.apply.config(state='disabled')
        tf.reset_default_graph()
        self.rects=[]
        self.other_imgs=[]
        CLASSES = ('__background__',
                   '1')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.tfmodel = "../models/vgg16_faster_rcnn_iter_50000.ckpt"
        if not os.path.isfile(self.tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(self.tfmodel + '.meta'))
        self.cfg_file = '../experiments/cfgs/vgg16.yml'
        cfg_from_file(self.cfg_file)
        self.tfconfig = tf.ConfigProto(allow_soft_placement=True)
        self.tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.tfconfig)
        self.net = vgg16()
        self.net.create_architecture("TEST", 2, tag='default', anchor_scales=[2], anchor_ratios=[1])
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.tfmodel)
        self.CONF_THRESH = 0.9
        self.NMS_THRESH = 0.3
        print('Loaded network {:s}'.format(self.tfmodel))
        name = '../models/global_model.pt'
        name2 = '../models/global_model_parameters.pt'
        self.model = torch.load(name, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(torch.load(name2, map_location=lambda storage, loc: storage))
        self.preprocess = transforms.Compose([
            # transformer.Pad_resize_conditional(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #self.root.mainloop()

    def start_win(self):

        for child in self.img_frame.winfo_children():
            child.destroy()
        self.rects = []
        self.other_imgs = []
        self.Cell_img = Image.open('../test_image/test1.jpg')
        img=cv2.imread('../test_image/test1.jpg')
        self.main_img = img
        self.seg_img = img
        self.set_img()


    def set_img(self):
        size = [1000, 1000]
        self.Cell_img.thumbnail((size[0], size[1]), Image.ANTIALIAS)
        self.render = ImageTk.PhotoImage(self.Cell_img)
        self.img_label=tk.Label(self.img_frame,image=self.render)
        self.img_label.image = self.render
        self.img_label.pack(side=LEFT)

    def auto_zoom(self,nn):
        print('Hi')

    def move_left(self):
        print('Hi')

    def move_right(self):
        print('Hi')

    def move_up(self):
        print('Hi')

    def move_down(self):
        print('Hi')

    def count(self):
        self.rects ,counter= self.Cell_detection(self.main_img, self.Cell_img,self.seg_img)
        self.prev_C=self.C.copy()
        for i in range(0,4):
            self.C[i]=self.C[i]+counter[i]
            self.C_label[i].config(state='normal')
            self.C_label[i].delete(0, 'end')
            self.C_label[i].insert(0,str(self.C[i]))
            self.C_label[i].config(state='disabled')

    def single_classifier(self, classifier_number):
        if len(self.rects)>0:
            render2 = ImageTk.PhotoImage(self.rects[classifier_number])
            self.img_label.configure(image=render2)
            self.img_label.image = render2
    def edit_manually(self):
        self.apply.config(state='normal')
        for i in range(0,5):
            self.C_label[i].config(state='normal')
    def apply_changes(self):
        self.prev_C=self.C.copy()
        for i in range(0,5):
            self.C[i]=int(self.C_label[i].get())
            print(self.C[i])
        self.apply.config(state='disabled')
        for i in range(0,5):
            self.C_label[i].config(state='disabled')
    def undo_count(self):
        self.C=self.prev_C.copy()
        for i in range(0,5):
            self.C_label[i].config(state='normal')
            self.C_label[i].delete(0, 'end')
            self.C_label[i].insert(0,str(self.C[i]))
            self.C_label[i].config(state='disabled')
    def Cell_detection(self,org_img1,resized_img,seg_img):
        nr = np.size(org_img1,0)
        nc = np.size(org_img1,1)
        nr_resized = resized_img.size[0]
        nc_resized = resized_img.size[1]
        CONF_THRESH = 0.9
        NMS_THRESH = 0.3
        bbox, score = self.localize_cells(self.sess, self.net, seg_img, CONF_THRESH, NMS_THRESH)
        rects = []
        counter = [0, 0, 0, 0]
        draw = []
        for ii in range(0, 4):
            rects.append(resized_img.copy())
            draw.append(ImageDraw.Draw(rects[ii]))
        l = len(bbox)
        for j in range(0, l):
            bb = [int(bbox[j][0]), int(bbox[j][1]), int(bbox[j][2]), int(bbox[j][3])]
            img2 = seg_img[bb[1]:bb[3], bb[0]:bb[2], :]

            img2 = cv2.resize(img2, (500, 500), fx=0, fy=0)
            res=self.test_model(img2)
            minr=bb[0]
            minc = bb[1]
            maxr=bb[2]
            maxc = bb[3]
            minr_org = (minr )
            minc_org = (minc)
            maxr_org = (maxr )
            maxc_org = (maxc )
            minr_resized =  nc_resized-(minr_org  /nr) * nr_resized
            minc_resized = (minc_org / nc) * nc_resized
            maxr_resized =nc_resized-(maxr_org / nr) * nr_resized
            maxc_resized =   (maxc_org / nc) * nc_resized
            draw[res].rectangle([int(minc_resized), int(minr_resized), int(maxc_resized), int(maxr_resized)])
            counter[res] += 1



        return rects,counter

    def localize_cells(self,sess, net, img, CONF_THRESH=0.5, NMS_THRESH=0.3):
        cls_ind = 1
        scores, boxes = im_detect(sess, net, img)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        bboxs = []
        scores = []
        for i in inds:
            bboxs.append(dets[i, :4])
            scores.append(dets[i, -1])
        return bboxs, scores

    def test_model(self, image, use_gpu=False):

        if use_gpu:
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                print("Classification is run on GPU")
        img_tensor = self.preprocess(image)

        if use_gpu:
            dtype = torch.cuda.FloatTensor
            tensor_var = img_tensor.unsqueeze_(0).cuda()
        else:
            dtype = torch.FloatTensor
            tensor_var = img_tensor.unsqueeze_(0)

        self.model.type(dtype)
        from torch.autograd import Variable as VV
        img_variable = VV(tensor_var)
        outputs = self.model(img_variable)
        _, preds = torch.max(outputs.data, 1)
        result = preds.item()
        return result


app=Cell_App()
app.root.mainloop()
