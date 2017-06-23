'''
Holds constants for the gui
'''
from Tkinter import Tk, Label, PhotoImage, Button

root = Tk()

window_width = 300
window_height = 300
window_position = 100
window_title = 'uatu_0.0.0'

loaded_model_text = Label(root, text='No model loaded')
loaded_test_text = Label(root, text='No test loaded')

tk_bb_image= PhotoImage(file='/home/gabi/PycharmProjects/uatu/images/bb.png')
bb_image_left = Label(root, image=tk_bb_image)
bb_image_right = Label(root, image=tk_bb_image)

run_button = Button(root, text='Run', state='disabled')
step_button = Button(root, text='Step', state='disabled')
stop_button = Button(root, text='Stop', state='disabled')

certainty_text = Label(root, text='1.00')
prediction_text = Label(root, text='MATCH')