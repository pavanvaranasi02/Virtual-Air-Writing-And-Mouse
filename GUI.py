import tkinter as tk
import time
import subprocess

def setBackgroundIMG(root, imgPath):
    # Set the background image
    bg_image = tk.PhotoImage(file="background.png")
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(relx=0.5, rely=0.5, anchor="center")

def stop_program(process):
    process.terminate()

def runVM():
    global is_open, process
    # Run the Python code
    path = "virtual_mouse_hands.py"
    if is_open and process != None:
        cur_status.config(text="Virtual Mouse is closing wait few seconds...")
        time.sleep(2)
        try:
            stop_program(process)
            is_open = False
        except:
            cur_status.config(text="Virtual Mouse is still running...")  
    process = subprocess.Popen(["python", path])
    if not is_open: 
        is_open = True
    if(process != None): cur_status.config(text="Virtual Mouse is running...")

def runFAW():
    global is_open, process
    # Run the Python code
    path = "finger_air_writing.py"
    if is_open and process != None:
        cur_status.config(text="Finger air writing is closing wait few seconds...")
        time.sleep(2)
        try:
            stop_program(process)
            is_open = False
        except:
            cur_status.config(text="Finger air writing is still running...")     
    process = subprocess.Popen(["python", path])
    if not is_open: 
        is_open = True
    if(process != None): 
        cur_status.config(text="Finger air writing is running...")


root = tk.Tk()

canvas = tk.Canvas(root, width=480, height=240)
# canvas.config(bg="#ADD8E6")     # Light Blue
# canvas.config(bg="#F0F0F0")     # Light Gray
canvas.config(bg="#333333")     # Dark gray
# canvas.config(bg="#4B0082")       # Dark purple
canvas.pack()

is_open = False
process = None

button1 = tk.Button(canvas, text="Virtual Mouse", command=runVM)
button1.place(relx=0.3, rely=0.5, anchor="center")
# button1.config(bg="#1E90FF", fg="white") # Dark Blue
# button1.config(bg="#808080", fg="white") # Dark Gray
button1.config(bg="#1E90FF", fg="white") # light blue
# button1.config(bg="#F0F0F0", fg="black") # light gray

button2 = tk.Button(canvas, text="Finger Air Writing", command=runFAW)
button2.place(relx=0.7, rely=0.5, anchor="center")
# button2.config(bg="#1E90FF", fg="white") # Dark Blue
# button2.config(bg="#808080", fg="white") # Dark Gray
button2.config(bg="#1E90FF", fg="white") # light blue
# button2.config(bg="#F0F0F0", fg="black") # light gray

cur_status = tk.Label(root, text="Waiting for user input.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
cur_status.pack(side=tk.BOTTOM, fill=tk.X)


root.mainloop()