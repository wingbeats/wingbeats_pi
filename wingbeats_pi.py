
from __future__ import division
from __future__ import print_function
print()
print('Loading libs...')

import os, sys, glob, time, errno
from shutil import copyfile
import numpy as np

if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import *
else:
    import tkinter as tk
    from tkinter import *

import soundfile as sf

import tensorflow as tf


def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


def center(toplevel):
    toplevel.update_idletasks()

    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    
    size = (500, 500)

    x = w / 2 - size[0] / 2
    y = h / 2 - size[1] / 2
    
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))


MONITOR_STATE = True


def on_closing():
    global MONITOR_STATE

    MONITOR_STATE = False
    
    root.destroy()


model_name = 'basic_cnn_1d.pb'

SR = 8000
input_shape = (5000, 1)

test_list_file = 'test_list'

path_to_watch = "/bluetooth"

target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 
'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

print()
print('Loading model...') 

with tf.gfile.FastGFile(model_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name = '')


test_list = []

with open(test_list_file) as f:
    test_list = f.read().splitlines()

class_test_list = []
filename_test_list = []

for i in range(len(test_list)):
    class_test_list.append(test_list[i].split('/')[0])
    filename_test_list.append(test_list[i].split('/')[-1])

print()
print('STOPPED! Press start at the window, send samples and watch the results on the screen...')

with tf.Session() as sess:
    x = sess.graph.get_tensor_by_name('input_1:0')
    softmax_tensor = sess.graph.get_tensor_by_name('output_node:0')

    def start_monitor():
        global MONITOR_STATE

        button1.config(state="disabled")
        button2.config(state="normal")

        root.update_idletasks()

        make_sure_path_exists('saved')

        df = glob.glob(path_to_watch + '/*.wav')

        for i in range(len(df)):
            if df[i].split(path_to_watch + '/')[1] in filename_test_list:
                os.system('sudo rm ' + df[i])

        before = dict ([(f, None) for f in os.listdir (path_to_watch)])

        MONITOR_STATE = True

        print()
        print('STARTED! Monitoring...')

        while MONITOR_STATE:
            time.sleep(0.001)

            root.update()

            after = dict ([(f, None) for f in os.listdir (path_to_watch)])

            added = [f for f in after if not f in before]

            if added:
                for i in range(len(added)):
                    if '.wav' in added[i]:
                        data, rate = sf.read(path_to_watch + '/' + added[i])

                        if len(data) == 5000:
                            true_class_name = 'Unknown class'
                            
                            for find_name in range(len(filename_test_list)):
                                if added[i] == filename_test_list[find_name]:
                                    true_class_name = class_test_list[find_name]
    
                            text1.insert(tk.END, '\n' + "Classifying: " + str(true_class_name) + ' - ' + str(added[i]))
                            text1.see(tk.END)
                            text1.update_idletasks()
                            
                            s_t = time.time()
                            
                            data = data / max(data)
                        
                            data = np.expand_dims(data, axis = -1)
                            data = np.expand_dims(data, axis = 0)
                        
                            preds = sess.run(softmax_tensor, feed_dict = {x: data})
                            
                            sorted_indices = sorted(range(len(preds)), key = lambda k: preds[k])
                            sorted_acc = sorted(preds)
                            
                            text1.insert(tk.END, '\n')
                    
                            for j in range(len(target_names)):
                                final_preds = str(target_names[sorted_indices[5 - j]])
                                current_pred_and_prob = str(j + 1) + ') ' + str(("%.2f" % round(sorted_acc[5 - j], 2))) + ' ' + str(final_preds)
                                text1.insert(tk.END, '\n' + current_pred_and_prob)
                            
                            e_t = time.time()
                            
                            total_time_took = str("Time (s): ") + str(("%.3f" % round(e_t - s_t, 3)))
                            text1.insert(tk.END, '\n\n' + total_time_took + '\n')
                            text1.insert(tk.END, '\n')

                            text1.see(tk.END)
                            text1.update_idletasks()
                            
                            copyfile(path_to_watch + '/' + added[i], 'saved/' + added[i])
                            os.system('sudo rm ' + path_to_watch + '/' + added[i])
                
            before = after


    def stop_monitor():
        global MONITOR_STATE

        button2.config(state="disabled")
        button1.config(state="normal")

        root.update_idletasks()
        
        MONITOR_STATE = False

        print()
        print('STOPPED! Press start at the window, send samples and watch the results on the screen...')


    root = tk.Tk()
    root.title("Wingbeats Pi")
    center(root)
    root.resizable(0, 0)
    frame = tk.Frame(root)

    text1 = Text(frame, width = 69, bg = "black", fg = "cyan")
    text1.pack(side = "left", fill = "both")
    scrollbar1 = Scrollbar(frame)
    scrollbar1.pack(side = "right", fill = "y")
    scrollbar1.config(command = text1.yview)
    text1.config(yscrollcommand = scrollbar1.set)

    frame.pack()
    root.update_idletasks()

    btn_text1 = tk.StringVar()
    button1 = Button(root, height = 4, width = 20, textvariable = btn_text1, command = start_monitor) 
    btn_text1.set("START")
    button1.pack()

    btn_text2 = tk.StringVar()
    button2 = Button(root, height = 4, width = 20, textvariable = btn_text2, command = stop_monitor) 
    btn_text2.set("STOP")
    button2.pack()
    button2.config(state = "disabled")

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()
