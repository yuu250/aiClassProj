# -*- coding = utf-8 -*-
# @Time : 2023/5/23 22:06
# @Author : KAI
# @File : ScanByAI_App.py
# @Software : PyCharm
import pathlib
import tkinter.filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

def pred_poke(img, model):
    """杰哥的代码"""
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    import tensorflow as tf
    import numpy as np

    class_names = ['Abra', 'Aerodactyl', 'Alakazam', 'Alolan Sandslash', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill',
                   'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard',
                   'Charmander', 'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett',
                   'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee',
                   'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon',
                   'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler',
                   'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno',
                   'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan',
                   'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp',
                   'Magmar', 'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo',
                   'Moltres', 'MrMime', 'Muk', 'Nidoking', 'Nidoqueen', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish',
                   'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey',
                   'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck',
                   'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash',
                   'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow',
                   'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel', 'Vaporeon',
                   'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 'Vulpix', 'Wartortle',
                   'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat']
    rescale_layer = Rescaling(scale=1. / 255)  # 将像素值除以255
    img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = rescale_layer(img_array)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)

    result = class_names[np.argmax(predictions)]
    print(result)

    return result

class GUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)

        # 载入模型
        self.model = load_model('dense121_final.h5', compile=False)

        # 图片输入路径
        _path = pathlib.Path().absolute().as_posix()
        self.path_var = ttk.StringVar(value=_path)

        # header labelframe
        option_text = '导入图片并预测'
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15, style=PRIMARY)
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        # preview frame
        preview_text = '预览图:'
        self.preview_f = ttk.Frame(self, padding=5)
        self.preview_f.pack(fill=X, expand=YES, padx=5, pady=5)
        img_label = ttk.Label(self.preview_f, text=preview_text)
        img_label.pack(side=LEFT, anchor=N)

        # result frame
        result_text = '预测结果:'
        self.result_f = ttk.Frame(self, padding=5)
        self.result_f.pack(fill=X, padx=5, pady=5)
        result_label = ttk.Label(self.result_f, text=result_text)
        result_label.pack(side=LEFT, anchor=N)

        # text pad
        self.text_pad = ttk.Text(self.result_f, height=30, width=30)
        self.text_pad.pack(fill=X, side=LEFT)

        self.creat_path_row()
        self.creat_btn_row()


    def on_browse(self):
        path = tkinter.filedialog.askopenfilename(title='选择一张图片')
        if path:
            self.path_var.set(path)
        self.creat_image_row()

    def on_predict(self, model):
        result = pred_poke(self.path_var.get(), model)
        self.text_pad.delete(1.0, 'end')
        self.text_pad.insert('insert', result)

    def creat_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text='路径', width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ety = ttk.Entry(path_row, textvariable=self.path_var)
        path_ety.pack(side=LEFT, fill=X, expand=YES, padx=5)
        browse_btn = ttk.Button(master=path_row, text='浏览', command=self.on_browse, width=8)
        browse_btn.pack(side=LEFT, padx=5)

    def creat_btn_row(self):
        btn_row = ttk.Frame(self.option_lf)
        btn_row.pack(fill=X, expand=YES, pady=15)
        predict_btn = ttk.Button(master=btn_row, text='开始预测', command=lambda: self.on_predict(self.model), width=8, style=SECONDARY)
        predict_btn.pack(padx=5)

    def creat_image_row(self):
        image_row = ttk.Frame(self.preview_f)
        image_row.pack(side=LEFT, padx=5, pady=5)
        global img
        img_open = Image.open(self.path_var.get())
        img = img_open.resize((350, 350))
        img = ImageTk.PhotoImage(img)
        label_show_image = ttk.Label(image_row, image=img)
        label_show_image.config(image=img)
        label_show_image.pack(side=BOTTOM, fill=BOTH, expand=YES, anchor=N)


if __name__ == '__main__':
    app = ttk.Window('宝可梦识别', themename='minty',size=(800, 700))
    app.attributes('-topmost', 1) # 让窗口位置其它窗口之上
    GUI(app)
    app.mainloop()

