
import PySimpleGUI as sg
import os
from PIL import Image
import io
import importlib
import sys
import shutil
from PySimpleGUI.PySimpleGUI import Column, HSeparator
from numpy.lib.shape_base import expand_dims
import time

from ui_test import *

start_time = time.time()
system_model = ComplementarySystem()
print("--- BUILDING MODEL : %s seconds ---" % (time.time() - start_time))

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

file_list_column = [
    [sg.Text("Anchor Image Folder", font = ("Raleway",10),size = (20,1))],
    [
        sg.In(size = (100,1), enable_events=True, key = "-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True,size = (120,5),
            key = "-FILE LIST-"
        )
    ],
    [sg.Text(size = (20,1),visible=False)],
    [sg.Text("Candidate Outfits Folder",font = ("Raleway",10),size = (20,1))],
    [
        sg.In(size = (100,1), enable_events=True, key = "-FOLDER_CANDIDATE-"),
        sg.FolderBrowse(),
    ],
    [sg.Text("Choose an anchor image and complementary outfit folder",font = ("Raleway",10),size = (50,1))],
    [sg.Text(size = (20,1),visible=False)],
    [
        sg.Text("Choose how many recommendations you want to get (max. 4)",size = (50,1),font = ("Raleway",10)),
        sg.Combo(["1","2","3","4"], default_value="1", enable_events=True, key = "REC_COMBO")
    ],
    [sg.Button("Find Complementary")],
    [
        sg.Column([
            [sg.Text(size = (20,1), key = "-TOUT-")],
            [sg.Image(key = "-IMAGE-",size = (12,12))],
        ], key = "COL"),
        sg.Column([
            [sg.Text(key = "-TOUT_COMP1-", font = ("Raleway",10),size = (20,1))],
            [sg.Image(key = "-IMAGE_COMP1-",size = (12,12))],
        ], key = "COL1"),
        sg.Column([
            [sg.Text(key = "-TOUT_COMP2-", font = ("Raleway",10),size = (20,1))],
            [sg.Image(key = "-IMAGE_COMP2-",size = (12,12))],
        ],visible=True,key = "COL2"),
        sg.Column([
            [sg.Text(key = "-TOUT_COMP3-", font = ("Raleway",10),size = (20,1))],
            [sg.Image(key = "-IMAGE_COMP3-",size = (12,12))],
        ],visible=True,key = "COL3"),
        sg.Column([
            [sg.Text(key = "-TOUT_COMP4-", font = ("Raleway",10),size = (20,1))],
            [sg.Image(key = "-IMAGE_COMP4-",size = (12,12))],
        ],visible=True,key = "COL4"),
    ],
    [   
        sg.Column([
            [sg.Text("Please choose the outfit you liked the most : ", font = ("Raleway",10),size = (40,1))],
            [sg.Radio(str(text+1),1, key = f"RCHOICE{text}") for text in range(4)]
        ])
        
    ],
    [
        sg.Button("Save",size = (25,1),pad = (5,5)),
        sg.Button("Clear",size = (25,1), pad = (5,5))
    ],
    

]

listbox_column = [
    [
        sg.Listbox(
            values=[], enable_events=True,size = (40,5),
            key = "-FILE LIST-"
        )
    ],
    [sg.Text(size = (20,1),visible=False)],

]

"""
image_viewer1_column = [
    
    [sg.Text(size = (40,1), key = "-TOUT-")],
    [sg.Image(key = "-IMAGE-",size = (8,8))],
    
]

image_viewer2_column = [
    [sg.Text(size = (40,1), key = "-TOUT_COMP-")],
    [sg.Image(key = "-IMAGE_COMP-",size = (8,8))],
    
]"""
sg.ChangeLookAndFeel("Purple")
#DarkBlue9
layout = [
    [
        sg.Column(file_list_column,pad = (10,10),size = (1250,800)),
        #sg.VSeparator(),
        #sg.Column(listbox_column, pad = (10,10)),
    ]
]

window = sg.Window("Complementary Outfit Recommendation System",layout, size = (1250,800))
#window = sg.Window(title = "Tamamlayıcı Kıyafet Öneri Sistemi", layout)

candidate_dict = {}

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder,f))
            and f.lower().endswith((".png",".jpg"))

        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":
        try:
            filename_anchor = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            image = Image.open(filename_anchor)
            image.thumbnail((200,200))
            bio = io.BytesIO()
            image.save(bio, format = "PNG")
            window["-TOUT-"].update(filename_anchor)
            window["-IMAGE-"].update(data = bio.getvalue())
        except:
            pass
    elif event == "-FOLDER_CANDIDATE-":
        folder_comp = values["-FOLDER_CANDIDATE-"]
    elif event == "Find Complementary":
        if folder_comp and filename_anchor:
            num_comp = int(values["REC_COMBO"])
            window.Element("COL").Update(visible = True)
            for ind in range(1,num_comp+1):
                window.Element(f"COL{ind}").Update(visible = True)
                time.sleep(1)
            
            start_time = time.time()
            print(filename_anchor)
            print(folder_comp)
            result_files = system_model.startModel(filename_anchor,folder_comp,True)
            print("--- PREDICTING ITEM : %s seconds ---" % (time.time() - start_time))
            for i in range(min(num_comp,len(result_files))):
                res = os.path.join(folder_comp,result_files[i])
                res_image = Image.open(res)
                res_image.thumbnail((200,200))
                bio = io.BytesIO()
                res_image.save(bio, format = "PNG")
                window[f"-TOUT_COMP{i+1}-"].update(res)
                candidate_dict[f"-TOUT_COMP{i+1}-"] = res               
                window[f"-IMAGE_COMP{i+1}-"].update(data = bio.getvalue())
    elif event == "Save":
        try:
            if not os.path.isdir("complementary_results"):
                os.makedirs("complementary_results")
            print([int(values[f"RCHOICE{j}"]) for j in range(4)])
            file_ind = np.argmax([int(values[f"RCHOICE{j}"]) for j in range(4)])
            print(file_ind)
            print(values)
            file_to_copy = candidate_dict[f'-TOUT_COMP{file_ind+1}-']
            file_names = os.listdir("complementary_results/")
            print("FOLDER NAMES = ", file_names)
            newfolder = max([int(f_name) for f_name in file_names])+1 if len(file_names) != 0 else 1
            os.makedirs("complementary_results/" + str(newfolder))
            newfolder = "complementary_results/" + str(newfolder)
            print(newfolder)
            print(file_to_copy)
            print(filename_anchor)
            shutil.copy(file_to_copy,newfolder)
            shutil.copy(filename_anchor,newfolder)
        except Exception as e:
            print(e)
            sg.Popup('Please do not try to save outfits without getting two images..')

    elif event == "Clear":
        window["-TOUT-"].update('')
        window["-TOUT_COMP1-"].update('')
        window["-TOUT_COMP2-"].update('')
        window["-TOUT_COMP3-"].update('')
        window["-TOUT_COMP4-"].update('')
        window["-FILE LIST-"].update('')
        window["-IMAGE-"].update('')
        window["-IMAGE_COMP1-"].update('')
        window["-IMAGE_COMP2-"].update('')
        window["-IMAGE_COMP3-"].update('')
        window["-IMAGE_COMP4-"].update('')
        window["-FOLDER-"].update('')
        window["-FOLDER_CANDIDATE-"].update('')
    


window.close()
