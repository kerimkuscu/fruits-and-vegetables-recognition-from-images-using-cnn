import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('model.h5')

#dictionary to label all traffic signs class.
classes = { 1: 'Apple Braeburn',
            2: 'Apple Crimson Snow',
            3: 'Apple Golden 1',
            4: 'Apple Golden 2',
            5: 'Apple Golden 3',
            6: 'Apple Granny Smith',
            7: 'Apple Pink Lady',
            8: 'Apple Red 1',
            9: 'Apple Red 2',
            10: 'Apple Red 3',
            11: 'Apple Red Delicious',
            12: 'Apple Red Yellow 1',
            13: 'Apple Red Yellow 2',
            14: 'Apricot',
            15: 'Avocado',
            16: 'Avocado ripe',
            17: 'Banana',
            18: 'Banana Lady Finger',
            19: 'Banana Red',
            20: 'Beetroot',
            21: 'Blueberry',
            22: 'Cactus fruit',
            23: 'Cantaloupe 1',
            24: 'Cantaloupe 2',
            25: 'Carambula',
            26: 'Cauliflower',
            27: 'Cherry 1',
            28: 'Cherry 2',
            29: 'Cherry Rainier',
            30: 'Cherry Wax Black',
            31: 'Cherry Wax Red',
            32: 'Cherry Wax Yellow',
            33: 'Chestnut',
            34: 'Clementine',
            35: 'Cocos',
            36: 'Corn',
            37: 'Corn Husk',
            38: 'Cucumber Ripe',
            39: 'Cucumber Ripe 2',
            40: 'Dates',
            41: 'Eggplant',
            42: 'Fig',
            43: 'Ginger Root',
            44: 'Granadilla',
            45: 'Grape Blue',
            46: 'Grape Pink',
            47: 'Grape White',
            48: 'Grape White 2',
            49: 'Grape White 3',
            50: 'Grape White 4',
            51: 'Grapefruit Pink',
            52: 'Grapefruit White',
            53: 'Guava',
            54: 'Hazelnut',
            55: 'Huckleberry',
            56: 'Kaki',
            57: 'Kiwi',
            58: 'Kohlrabi',
            59: 'Kumquats',
            60: 'Lemon',
            61: 'Lemon Meyer',
            62: 'Limes',
            63: 'Lychee',
            64: 'Mandarine',
            65: 'Mango',
            66: 'Mango Red',
            67: 'Mangostan',
            68: 'Maracuja',
            69: 'Melon Piel de Sapo',
            70: 'Mulberry',
            71: 'Nectarine',
            72: 'Nectarine Flat',
            73: 'Nut Forest',
            74: 'Nut Pecan',
            75: 'Onion Red',
            76: 'Onion Red Peeled',
            77: 'Onion White',
            78: 'Orange',
            79: 'Papaya',
            80: 'Passion Fruit',
            81: 'Peach',
            82: 'Peach 2',
            83: 'Peach Flat',
            84: 'Pear',
            85: 'Pear 2',
            86: 'Pear Abate',
            87: 'Pear Forelle',
            88: 'Pear Kaiser',
            89: 'Pear Monster',
            90: 'Pear Red',
            91: 'Pear Stone',
            92: 'Pear Williams',
            93: 'Pepino',
            94: 'Pepper Green',
            95: 'Pepper Orange',
            96: 'Pepper Red',
            97: 'Pepper Yellow',
            98: 'Physalis',
            99: 'Physalis with Husk',
            100: 'Pineapple',
            101: 'Pineapple Mini',
            102: 'Pitahaya Red',
            103: 'Plum',
            104: 'Plum 2',
            105: 'Plum 3',
            106: 'Pomegranate',
            107: 'Pomelo Sweetie',
            108: 'Potato Red',
            109: 'Potato Red Washed',
            110: 'Potato Sweet',
            111: 'Potato White',
            112: 'Quince',
            113: 'Rambutan',
            114: 'Raspberry',
            115: 'Redcurrant',
            116: 'Salak',
            117: 'Strawberry',
            118: 'Strawberry Wedge',
            119: 'Tamarillo',
            120: 'Tangelo',
            121: 'Tomato 1',
            122: 'Tomato 2',
            123: 'Tomato 3',
            124: 'Tomato 4',
            125: 'Tomato Cherry Red',
            126: 'Tomato Heart',
            127: 'Tomato Maroon',
            128: 'Tomato not Ripened',
            129: 'Tomato Yellow',
            130: 'Walnut',
            131: 'Watermelon', }
                 
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Fruit Recognition From Images')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Fruit Recognition From Images",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
