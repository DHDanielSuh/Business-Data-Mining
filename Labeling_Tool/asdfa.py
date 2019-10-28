from tkinter import *
from PIL import Image, ImageTk


class MyApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # self.main = ScrolledCanvas(self)
        # self.main.grid(row=0, column=0, sticky='nsew')
        # self.c = self.main.canv
        # self.geometry("1200x800")
        self.canv = Canvas(self, bd=0, highlightthickness=0)
        self.canv.grid(row=0, column=0, sticky='nsew')

        self.resizable(True, True)
        self.currentImage = {}
        self.load_imgfile('/home/daniel/다운로드/서동현TEST/0001.jpg')

        self.canv.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canv.bind('<B1-Motion>', self.on_mouse_drag)
        self.canv.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canv.bind('<Button-3>', self.on_right_click)



    def load_imgfile(self, filename):
        img = Image.open(filename)
        img = img.convert('L')
        self.currentImage['data'] = img

        photo = ImageTk.PhotoImage(img)
        self.canv.xview_moveto(0)
        self.canv.yview_moveto(0)
        self.canv.create_image(0, 0, image=photo, anchor='nw', tags='img')
        self.canv.config(scrollregion=self.canv.bbox('all'))
        self.currentImage['photo'] = photo

    def on_mouse_down(self, event):
        self.anchor = (event.widget.canvasx(event.x),
                       event.widget.canvasy(event.y))
        self.item = None

    def on_mouse_drag(self, event):
        bbox = self.anchor + (event.widget.canvasx(event.x),
                              event.widget.canvasy(event.y))
        if self.item is None:
            self.item = event.widget.create_rectangle(bbox, outline="yellow")
        else:
            event.widget.coords(self.item, *bbox)

    def on_mouse_up(self, event):
        if self.item:
            self.on_mouse_drag(event)
            box = tuple((int(round(v)) for v in event.widget.coords(self.item)))

            roi = self.currentImage['data'].crop(box) # region of interest
            values = roi.getdata() # <----------------------- pixel values
            print(roi.size, len(values))
            #print list(values)

    def on_right_click(self, event):
        found = event.widget.find_all()
        for iid in found:
            if event.widget.type(iid) == 'rectangle':
                event.widget.delete(iid)



app =  MyApp()
app.resizable(True, True)
app.mainloop()