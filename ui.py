from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image

from face import *

root = Tk()
root.geometry("900x900")
root.resizable(width=True, height=True)


def openfn():
    filename = filedialog.askopenfilename(title='Выбрать фото')
    return filename


def detect():
    find_face("dataset/original/1.jpg")


def open_img():
    x = openfn()
    img = Image.open(x)
    shutil.copyfile(x, "dataset/original/1.jpg")
    width, height = img.size
    if (width > 500) & (height > 500):
        img = img.resize((width // 2, height // 2))
    else:
        img = img.resize((width * 2, height * 2))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()


btn = Button(root, text='Выбрать фото', command=open_img).pack()
butn = Button(root, text='Проверить наличие маски', bd='5', command=detect).pack()
root.mainloop()
