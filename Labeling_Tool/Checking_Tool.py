import xml.dom.minidom
import tkinter as tk
import sys
from PIL import Image, ImageTk
import tkinter.font
import os
from tkinter import filedialog
import cv2


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


class Global(tk.Frame):
    frameNumber = 0

class Application(Global):

    def __init__(self, parent, frames):
        # initialize frame
        tk.Frame.__init__(self, parent)

        # set root as parent
        self.parent = parent
        self.image_path = image_path
        self.xml_path = xml_path
        self.frames = frames

        # self.Index = None
        # read and parse XML document
        DOMTree = xml.dom.minidom.parse(xml_path)

        # create attribute for XML document
        self.xmlDocument = DOMTree.documentElement

        # get value of "folder" element
        self.folder = tk.StringVar()
        self.folder.set(getElementValue(self.xmlDocument, "folder"))

        # get value of "filename" element
        self.filename = tk.StringVar()
        self.filename.set(getElementValue(self.xmlDocument, "filename"))

        # create attribute for "size" element
        self.xmlSize = getElement(self.xmlDocument, "size")

        # get value of "width" element
        self.width = tk.StringVar()
        self.width.set(getElementValue(self.xmlSize, "width"))

        # get value of "height" element
        self.height = tk.StringVar()
        self.height.set(getElementValue(self.xmlSize, "height"))

        # get value of "depth" element
        self.depth = tk.StringVar()
        self.depth.set(getElementValue(self.xmlSize, "depth"))

        # create attribute for "object" element
        self.xmlObject = getElement(self.xmlDocument, "object")

        # get value of "name" element
        self.name = tk.StringVar()
        self.name.set(getElementValue(self.xmlObject, "name"))

        # get value of "result" element
        self.result = tk.StringVar()
        self.result.set(getElementValue(self.xmlObject, "result"))

        # get value of "groundtruth" element
        self.groundtruth = tk.StringVar()
        self.groundtruth.set(getElementValue(self.xmlObject, "groundtruth"))

        # initialize UI
        self.initUI()

    def initUI(self):
        # set frame title
        self.parent.title("Labeling Tool")

        # pack frame
        self.pack(fill=tk.BOTH, expand=1, side='right')

        # configure grid columns
        self.columnconfigure(0, pad=3)
        self.columnconfigure(1, pad=3)
        self.columnconfigure(2, pad=3)
        self.columnconfigure(3, pad=3)


        # configure grid rows
        self.rowconfigure(0, pad=3)
        self.rowconfigure(1, pad=3)
        self.rowconfigure(2, pad=3)
        self.rowconfigure(3, pad=3)
        self.rowconfigure(4, pad=3)
        self.rowconfigure(6, pad=3)
        self.rowconfigure(7, pad=3)
        self.rowconfigure(8, pad=3)
        self.rowconfigure(9, pad=3)
        self.rowconfigure(10, pad=3)
        self.rowconfigure(11, pad=3)

        # font setting
        font = tk.font.Font(size=11)

        # image
        self.loaded_img = cv2.imread(self.image_path)
        # Image.open(self.image_path)

        # Resize Image
        self.resized_image = cv2.resize(self.loaded_img, dsize=(1200, 1000))
        # loaded_img.resize((1200, 1000))
        self.resized_image  = cv2.cvtColor(self.resized_image , cv2.COLOR_BGR2RGBA)
        self.resized_image  = Image.fromarray(self.resized_image )

        render = ImageTk.PhotoImage(image=self.resized_image )
        img = tk.Label(self, image=render)
        img.image = render
        img.grid(row=0, column=0, columnspan=1, rowspan=20, padx=10, pady=20)

        # folder
        label1 = tk.Label(self, text="folder: ", font=font)
        label1.grid(row=1, column=2, sticky=tk.W, columnspan=30)

        entry1 = tk.Entry(self, width=45, textvariable=self.folder, font=font)
        entry1.grid(row=1, column=3)

        # filename
        label2 = tk.Label(self, text="filename : ", font=font)
        label2.grid(row=2, column=2, sticky=tk.W, columnspan=20)

        entry2 = tk.Entry(self, width=45, textvariable=self.filename, font=font)
        entry2.grid(row=2, column=3)

        # width
        label3 = tk.Label(self, text="width : ", font=font)
        label3.grid(row=3, column=2, sticky=tk.W)

        entry3 = tk.Entry(self, width=45, textvariable=self.width, font=font)
        entry3.grid(row=3, column=3)

        # height
        label4 = tk.Label(self, text="height : ", font=font)
        label4.grid(row=4, column=2, sticky=tk.W)

        entry4 = tk.Entry(self, width=45, textvariable=self.height, font=font)
        entry4.grid(row=4, column=3)

        # depth
        label5 = tk.Label(self, text="depth : ", font=font)
        label5.grid(row=5, column=2, sticky=tk.W)

        entry5 = tk.Entry(self, width=45, textvariable=self.depth, font=font)
        entry5.grid(row=5, column=3)

        # name
        label6 = tk.Label(self, text="name :", font=font)
        label6.grid(row=6, column=2, sticky=tk.W)

        entry6 = tk.Entry(self, width=45, textvariable=self.name, font=font)
        entry6.grid(row=6, column=3)

        # result
        label7 = tk.Label(self, text="result :", font=font)
        label7.grid(row=7, column=2, sticky=tk.W)

        entry7 = tk.Entry(self, width=45, textvariable=self.result, font =font)
        entry7.grid(row=7, column=3)

        # groundtruth
        label8 = tk.Label(self, text="groundtruth :", font=font)
        label8.grid(row=8, column=2, sticky=tk.W)

        entry8 = tk.Entry(self, width=45, textvariable=self.groundtruth, font=font)
        entry8.grid(row=8, column=3)

        #grid formating
        col_count, row_count = self.grid_size()

        for col in range(col_count):
            self.grid_columnconfigure(col, minsize=50)


        # for row in range(row_count):
        #     self.grid_rowconfigure(row, minsize=10)

        # create OK button
        button1 = tk.Button(self, text="SAVE", command=self.onOK, font=font)
        button1.config(height=8, width=21)
        button1.grid(row=12, column=2)
        button1.bind("<Return>", self.onOK)

        # create EXIT button
        button2 = tk.Button(self, text="EXIT", command=self.onCancel, font= font)
        button2.config(height=8, width=21)
        button2.grid(row=12, column=3)
        button2.bind("<Return>", self.onCancel)

        # create Previous button
        button3 = tk.Button(self, text="PREV", command=self.previous, font=font)
        button3.config(height=8, width=21)
        button3.grid(row=13, column=2)
        button3.bind("<Return>", self.previous)

        # create Next button
        button4 = tk.Button(self, text="NEXT", command=self.next, font=font)
        button4.config(height=8, width=21)
        button4.grid(row=13, column=3)
        button4.bind("<Return>", self.next)

    def onOK(self, event=None):
        # set values in xml document
        setElementValue(self.xmlDocument, "folder", self.folder.get())
        setElementValue(self.xmlDocument, "filename", self.filename.get())
        setElementValue(self.xmlSize, "width", self.width.get())
        setElementValue(self.xmlSize, "height", self.height.get())
        setElementValue(self.xmlSize, "depth", self.depth.get())
        setElementValue(self.xmlObject, "name", self.name.get())
        setElementValue(self.xmlObject, "result", self.result.get())
        setElementValue(self.xmlObject, "groundtruth", self.groundtruth.get())

        # open XML file
        f = open(self.xml_path, "w")

        # set xml header
        fprintf(f, '<?xml version="1.0" encoding="utf-8"?>\n')

        # write XML document to XML file
        self.xmlDocument.writexml(f)

        # close XML file
        f.close()

        # # exit program
        # self.quit()

    def onCancel(self, event=None):
        # exit program
        self.quit()

    def next(self, event=None):
        try:
            # print("previous :", Global.frameNumber)
            # previous_frame =  self.frames[Global.frameNumber]
            # print("previous frame :", previous_frame)
            # previous_frame.pack_forget()
            # previous_frame.pack()
            #
            # Global.frameNumber += 1
            # new_frame = self.frames[Global.frameNumber]
            # print("new:", Global.frameNumber)
            # print("new frame :", new_frame)
            # # new_frame.pack()
            # new_frame.tkraise()
            #
            # print("-----------------------")


            if Global.frameNumber == len(self.frames)-1:
                print("previous :", Global.frameNumber)
                previous_frame = self.frames[Global.frameNumber]
                print("previous frame:", previous_frame)
                previous_frame.pack_forget()
                previous_frame.pack()
                Global.frameNumber = 0
                new_frame = self.frames[Global.frameNumber]
                print("new frame :", new_frame)
                print("new:", Global.frameNumber)
                new_frame.pack()
                new_frame.tkraise()
                print("--------------")
            else:
                print("previous :", Global.frameNumber)
                previous_frame = self.frames[Global.frameNumber]
                print("previous frame:", previous_frame)
                previous_frame.pack_forget()
                previous_frame.pack()
                Global.frameNumber += 1
                new_frame = self.frames[Global.frameNumber]
                print("new frame :", new_frame)
                print("new:", Global.frameNumber)
                new_frame.pack()
                new_frame.tkraise()

                print("-----------------")
        except:
            print("This is the Last Page")


    def previous(self, event=None):
        try:
        #     print("previous :", Global.frameNumber)
        #     previous_frame = self.frames[Global.frameNumber]
        #     print("previous frame:", previous_frame)
        #     previous_frame.pack_forget()
        #     previous_frame.pack()
        #
        #     Global.frameNumber -= 1
        #     new_frame = self.frames[Global.frameNumber]
        #     print("new:", Global.frameNumber)
        #     print("new frame :", new_frame)
        #     new_frame.pack()
        #     new_frame.tkraise()
        #
        #     print("------------------------")


            if Global.frameNumber == 0:
                print("previous :", Global.frameNumber)
                previous_frame = self.frames[Global.frameNumber]
                print("previous frame:", previous_frame)
                previous_frame.pack_forget()
                previous_frame.pack()
                Global.frameNumber = len(self.frames) - 1
                new_frame = self.frames[Global.frameNumber]
                print("new frame :", new_frame)
                print("new:", Global.frameNumber)
                new_frame.pack()
                new_frame.tkraise()
                print("-----------------")
            else:
                print("previous :", Global.frameNumber)
                previous_frame = self.frames[Global.frameNumber]
                print("previous frame:", previous_frame)
                previous_frame.pack_forget()
                previous_frame.pack()
                Global.frameNumber -= 1
                new_frame = self.frames[Global.frameNumber]
                print("new frame :", new_frame)
                print("new:", Global.frameNumber)
                new_frame.pack()
                new_frame.tkraise()
                print("--------------------")

        except:
            print("This is the First Page")


    # def bounding_box(self):
    #     fromCenter = False
    #     showCrosshair = False
    #     r = cv2.selectROI("Image", self.resized_image, fromCenter, showCrosshair)
    #     top_left = int(r[0])
    #     _right = int(r[3])
def main():

    # initialize root object
    root = tk.Tk()

    # set size of frame
    # root.geometry("1200x800")
    root.resizable(True, True)

    # select open folder
    folder_path = filedialog.askdirectory(initialdir='/home/daniel/다운로드/서동현TEST')

    # List Extraction
    List = [os.path.normcase(f) for f in os.listdir(folder_path)]
    imgList = [os.path.join(folder_path, f) for f in List if os.path.splitext(f)[1] == '.jpg']
    xmlList = [os.path.join(folder_path, f) for f in List if os.path.splitext(f)[1] == '.xml']
    imgList.sort()
    xmlList.sort()

    # set key and value for a dictionary
    frameNumber = 0
    frames = []

    # append each frame to dictionary
    for image_path, xml_path in zip(imgList, xmlList):
        image_file_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_file_name = os.path.splitext(os.path.basename(xml_path))[0]
        if image_file_name == xml_file_name:
#            app = Application(root, image_path, xml_path, frames)
            frames.append((image_path, xml_path))
            # frames[frameNumber] = app
            # frameNumber += 1

    app = Application(root, frames)

    # # call object
    # app = Application(root)

    # enter main loop
    root.mainloop()


# call main() function
if __name__ == '__main__':
    main()
















