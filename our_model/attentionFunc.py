#might need to install these and update playsound in terminal
#pip3 install playsound
#pip3 install PyObjC
from enum import Enum
import subprocess
from playsound import playsound
#import tkinter as tk

#master = tk.Tk()

class Status(Enum):
  GREEN = 1
  YELLOW = 2
  RED = 3

class P:#class vars are persistent
  status = Status.GREEN
  count = 0
  play = True

def attention_status(cur_bodies, cur_faces, thresh = 0.79, cthresh = 4, safety = 0.5):
  with open('attentionOutput.txt', 'w') as f:
    info_message = f"{cur_bodies} people detected in frame."
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=2, column=1) 
    info_message = f"{cur_faces} people facing forward."
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=3, column=1)

    attn_score = cur_faces/cur_bodies
    info_message = f"Paying Attention score: {attn_score}"
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=4, column=1)

    if attn_score <= thresh:
      if P.count != cthresh:
        P.count = P.count +1
    else:
      if P.count > 0:
        P.count = P.count -1#perhaps -2

    if P.count < (cthresh * safety):
      P.status = Status.GREEN
      P.play = True
    elif P.count == cthresh:
      P.status = Status.RED
      if P.play:
        playsound('./break_x.wav')
        P.play = False#only plays once per red trigger
    else:
      P.status = Status.YELLOW

    info_message = f"Class status: {P.status}\n"
    print(info_message)
    f.write(info_message)
    ##tk.Label(master, text=info_message).grid(row=5, column=1)

  return 0

#main test:
attention_status(5, 3)
attention_status(5, 3)
attention_status(5, 4)
attention_status(5, 3)
attention_status(5, 3)
attention_status(5, 3)
attention_status(5, 5)
attention_status(5, 3)
attention_status(5, 5)
attention_status(5, 5)
attention_status(5, 5)
attention_status(5, 3)
attention_status(5, 3)
attention_status(5, 3)
attention_status(5, 3)