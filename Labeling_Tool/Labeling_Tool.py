from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import glob
import cv2
import numpy

# colors for the bboxes
COLORS = ['red', 'blue', 'purple', 'yellow', 'green', 'black']

def printf(format, *args):
    sys.stdout.write(format % args)

def fprintf(fp, format, *args):
    fp.write(format % args)


# get an XML element with specified name
def getElement(parent, name):
    nodeList = []
    if parent.childNodes:
        for node in parent.childNodes:
            if node.nodeType == node.ELEMENT_NODE:
                if node.tagName == name:
                    nodeList.append(node)
    return nodeList[0]


# get value of an XML element with specified name
def getElementValue(parent, name):
    if parent.childNodes:
        for node in parent.childNodes:
            if node.nodeType == node.ELEMENT_NODE:
                if node.tagName == name:
                    if node.hasChildNodes:
                        child = node.firstChild
                        if child:
                            return child.nodeValue

    return None

# set value of an XML element with specified name
def setElementValue(parent, name, value):
    if parent.childNodes:
        for node in parent.childNodes:
            if node.nodeType == node.ELEMENT_NODE:
                if node.tagName == name:
                    if node.hasChildNodes:
                        child = node.firstChild
                        if child:
                            child.nodeValue = value
    return None


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.scale = 1  # Added
        self.parent = master
        self.parent.title("Labeling Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList = []
        self.xmlList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None
        self.currentLabelclass = ''
        self.currentLabelpose = ''
        self.currentLabeltype = ''
        self.cla_can_temp = []
        # self.classcandidate_filename = 'class.txt'

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        #reference to polygon
        self.polygonList = []

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # # zoom
        # self.zoomcycle = 0
        # self.zimg_id = None


        # ----------------- GUI stuff ---------------------
        # dir entry & load
        # input image dir button
        self.srcDirBtn = Button(self.frame, text="Image input folder", command=self.selectSrcDir)
        self.srcDirBtn.grid(row=0, column=0, sticky = W+E)

        # input image dir entry
        self.svSourcePath = StringVar()
        self.entrySrc = Entry(self.frame, textvariable=self.svSourcePath)
        self.entrySrc.grid(row=0, column=1, sticky=W + E)
        # self.svSourcePath.set('/home')

        # load button
        self.ldBtn = Button(self.frame, text="Load Dir", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, rowspan=2, columnspan=2, padx=2, pady=2, ipadx=5, ipady=5, sticky= W)

        # label file save dir button
        self.desDirBtn = Button(self.frame, text="Label output folder", command=self.selectDesDir)
        self.desDirBtn.grid(row=1, column=0, sticky = W+E)

        # label file save dir entry
        self.svDestinationPath = StringVar()
        self.entryDes = Entry(self.frame, textvariable=self.svDestinationPath)
        self.entryDes.grid(row=1, column=1, sticky=W + E)
        # self.svDestinationPath.set('/home')

        #cropped image open button
        # self.cropped_window = Button(self.frame, text="Crop & Zoom Image", command=self.crop_plate)
        # self.cropped_window.grid(row=0, column=3, rowspan=2, columnspan=2, padx=2, pady=2, ipadx=5, ipady=5, sticky = E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouse_left_Click)
        self.mainPanel.bind("<Button-3>", self.mouse_right_Click)
        self.mainPanel.bind("<Button-2>", self.mouse_wheel_Click)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        # self.mainPanel.bind("<Button-4>", self.zoomin)
        # self.mainPanel.bind("<Button-5>", self.zoomout)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Esc> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("p", self.prevImage)  # press 'p' to go backforward
        self.parent.bind("n", self.nextImage)  # press 'n' to go forward
        self.mainPanel.grid(row=2, column=1, rowspan=4, sticky=W + N)


        # choose class
        self.classname = StringVar()
        self.classcandidate = ttk.Combobox(self.frame, state='readonly', textvariable=self.classname)
        self.classcandidate.grid(row=2, column=2, sticky=W + N)
        # if os.path.exists(self.classcandidate_filename):
        #     with open(self.classcandidate_filename) as cf:
        #         for line in cf.readlines():
        #             self.cla_can_temp.append(line.strip('\n'))
        self.classcandidate['values'] = ('Vehicle', 'Window', 'Plate')
        self.classcandidate.current(0)
        self.currentLabelclass = self.classcandidate.get()
        self.btnclass = Button(self.frame, text='ComfirmClass', command=self.setClass)
        self.btnclass.grid(row=2, column=3, sticky=W + E + N)

        # choose pose
        self.posename = StringVar()
        self.posecandidate = ttk.Combobox(self.frame, state='readonly', textvariable=self.posename)
        self.posecandidate.grid(row=3, column=2, sticky=W + N)
        self.posecandidate['values'] = ('Front', 'Back')
        self.posecandidate.current(0)
        self.currentLabelpose = self.posecandidate.get()
        self.btnpose = Button(self.frame, text='ComfirmPose', command=self.setPose)
        self.btnpose.grid(row=3, column=3, sticky=W + E + N)

        # choose type
        self.typename = StringVar()
        self.typecandidate = ttk.Combobox(self.frame, state='readonly', textvariable=self.typename)
        self.typecandidate.grid(row=4, column=2, sticky=W + N)
        self.typecandidate['values'] = ('Car', 'Bus', 'Truck')
        self.typecandidate.current(0)
        self.currentLabeltype = self.typecandidate.get()
        self.btntype = Button(self.frame, text='Comfirmtype', command=self.setType)
        self.btntype.grid(row=4, column=3, sticky=W + E + N)

        # showing bbox info & delete bbox
        # self.lb1 = Label(self.frame, text='Bounding boxes:')
        # self.lb1.grid(row=5, column=2, sticky=W+S)
        self.listbox = Listbox(self.frame, width=40, height=40)
        self.listbox.grid(row=5, column=2, sticky=N + S)
        self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=5, column=3, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=5, column=3, sticky=W + E + S)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=6, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def selectSrcDir(self):
        path = filedialog.askdirectory(title="Select image source folder", initialdir='/home/daniel/다운로드/서동현TEST')
        self.svSourcePath.set(path)
        return

    def selectDesDir(self):
        path = filedialog.askdirectory(title="Select label output folder", initialdir='/home/daniel/다운로드/서동현TEST')
        self.svDestinationPath.set(path)
        return

    def loadDir(self):
        self.parent.focus()
        # get image list
        # self.imageDir = os.path.join(r'./Images', '%03d' %(self.category))
        self.imageDir = self.svSourcePath.get()
        if not os.path.isdir(self.imageDir):
            messagebox.showerror("Error!", message="The specified dir doesn't exist!")
            return

        extlist = ["*.JPEG", "*.jpeg", "*JPG", "*.jpg", "*.PNG", "*.png", "*.BMP", "*.bmp"]
        for e in extlist:
            filelist = glob.glob(os.path.join(self.imageDir, e))
            xmllist = glob.glob(os.path.join(self.imageDir, '*.xml'))
            self.imageList.extend(filelist)
            self.xmlList.extend(xmllist)
        # self.imageList = glob.glob(os.path.join(self.imageDir, '*.JPEG'))
        if len(self.imageList) == 0:
            print('No .JPEG images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # set up output dir
        # self.outDir = os.path.join(r'./Labels', '%03d' %(self.category))
        self.outDir = self.svDestinationPath.get()
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)

        self.loadImage()
        print('%d images loaded from %s' % (self.total, self.imageDir))

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        size = self.img.size
        self.factor = max(size[0] / 1000, size[1] / 1000., 1.)
        self.img = self.img.resize((int(size[0] / self.factor), int(size[1] / self.factor)))
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clearBBox()
        # self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        fullfilename = os.path.basename(imagepath)
        self.imagename, _ = os.path.splitext(fullfilename)
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue
                    # tmp = [int(t.strip()) for t in line.split()]
                    tmp = line.split()
                    tmp[0] = int(int(tmp[0]) / self.factor)
                    tmp[1] = int(int(tmp[1]) / self.factor)
                    tmp[2] = int(int(tmp[2]) / self.factor)
                    tmp[3] = int(int(tmp[3]) / self.factor)
                    self.bboxList.append(tuple(tmp))
                    color_index = (len(self.bboxList) - 1) % len(COLORS)
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1],
                                                            tmp[2], tmp[3],
                                                            width=2,
                                                            outline=COLORS[color_index])
                    # outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' % (tmp[4], tmp[0], tmp[1], tmp[2], tmp[3]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[color_index])

    def saveImage(self):
        if self.labelfilename == '':
            return
        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            # f.write(self.currentLabelpose + '\n')
            # f.write(self.currentLabeltype + '\n')
            for bbox in self.bboxList:
                f.write("{} {} {} {} {}\n".format(int(int(bbox[0]) * self.factor),
                                                  int(int(bbox[1]) * self.factor),
                                                  int(int(bbox[2]) * self.factor),
                                                  int(int(bbox[3]) * self.factor),
                                                  bbox[4]))

                                                  # , bbox[5], bbox[6],
                                                  # int(int(bbox[7]) * self.factor),
                                                  # int(int(bbox[8]) * self.factor),
                                                  # int(int(bbox[9]) * self.factor),
                                                  # int(int(bbox[10]) * self.factor),
                                                  # bbox[11],
                                                  # int(int(bbox[12]) * self.factor),
                                                  # int(int(bbox[13]) * self.factor),
                                                  # int(int(bbox[14]) * self.factor),
                                                  # int(int(bbox[15]) * self.factor),
                                                  # bbox[16]
                                                  # ))
                # f.write(' '.join(map(str, bbox)) + '\n')
        print('Image No. %d saved' % (self.cur))


    def mouse_left_Click(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass))
                                  # , self.currentLabelpose, self.currentLabeltype))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' % (self.currentLabelclass, x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouse_wheel_Click(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass))
                                  # , self.currentLabelpose, self.currentLabeltype))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' % (self.currentLabelclass, x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']
        self.new_window = Toplevel()
        self.new_window.title("Cropped Image")
        self.canvas = Canvas(self.new_window)
        self.canvas.pack()
        self.cropped_image = self.img.crop((x1, x2, y1, y2))
        self.pil_img = PhotoImage(self.cropped_image)
        self.cv_img = cv2.cvtColor(numpy.array(self.pil_img), cv2.COLOR_BGR2RGB)




    def mouse_right_Click(self, event):
        self.STATE['x'], self.STATE['y'] = event.x, event.y
        self.bboxId = self.mainPanel.create_oval(self.STATE['x'], self.STATE['y'],
                                                 self.STATE['x'], self.STATE['y'], fill="red", width = 8)
        self.bboxList.append((self.STATE['x'], self.STATE['y'], event.x, event.y, self.currentLabelclass))
        self.bboxIdList.append(self.bboxId)
        self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' % (self.currentLabelclass, self.STATE['x'],
                                                                self.STATE['y'], event.x, event.y))

    def mouseMove(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            COLOR_INDEX = len(self.bboxIdList) % len(COLORS)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                          event.x, event.y,
                                                          width=2,
                                                          outline=COLORS[len(self.bboxList) % len(COLORS)])



    def zoomin(self, event):
        self.mainPanel.scale("all", self.mainPanel.canvasx(event.x), self.mainPanel.canvasy(event.y), 1.1, 1.1)
        # self.mainPanel.configure(scrollregion=self.mainPanel.("all"))


    def zoomout(self, event):
        self.mainPanel.scale("all", event.x, event.y, 0.9, 0.9)
        # self.mainPanel.configure(scrollregion=self.mainPanel.bbox("all"))

    # def zoomin(self, event):
    #     """Detect the zoom action by the mouse. Zoom on the mouse focus"""
    #     true_x = self.mainPanel.canvasx(event.x)
    #     true_y = self.mainPanel.canvasy(event.y)
    #     self.mainPanel.scale("all", true_x, true_y, 1.2, 1.2)
    #     self.scale *= 1.2  # **Added
    #     self.mainPanel.configure(scrollregion=self.mainPanel.bbox("all"))
    #
    # def zoomout(self, event):
    #     true_x = self.mainPanel.canvasx(event.x)
    #     true_y = self.mainPanel.canvasy(event.y)
    #     self.mainPanel.scale("all", true_x, true_y, 0.8, 0.8)
    #     self.scale *= 0.8  # **Added
    #     self.mainPanel.configure(scrollregion=self.mainPanel.bbox("all"))

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

    def setClass(self):
        self.currentLabelclass = self.classcandidate.get()
        print('set label class to : %s' % self.currentLabelclass)

    def setPose(self):
        self.currentLabelpose = self.posecandidate.get()
        print('set label pose to : %s' % self.currentLabelpose)

    def setType(self):
        self.currentLabeltype = self.typecandidate.get()
        print('set label type to : %s' % self.currentLabeltype)

    def crop_plate(self):
        pass
        # self.new_window = Toplevel()
        # self.new_window.title("Cropped Image")
        # self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # self.height, self.width, no_channels = self.cv_img.shape
        # self.new_canvas = Canvas(self.new_window)
        # self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img)
        # self.new_canvas.create_image(0, 0, image=self.photo, anchor=NW)





if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()