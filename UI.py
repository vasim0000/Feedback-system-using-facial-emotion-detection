import subprocess
import threading
from tkinter import Tk, Canvas, Button, PhotoImage, Text, Scrollbar, Frame
from threading import Thread
from tkinter import Label

choice = None
process = None  # Global variable to hold the process

def button_clicked(button_number):
    global choice
    choice = button_number
    window.destroy()
    run_seg3(choice)

def run_seg3(choice):
    global process
    process = subprocess.Popen(["python", "segregation3.py", str(choice)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)

    # Create the output window
    output_window = Tk()
    output_window.title("Output")
    output_window.geometry("650x450")

    # Create a frame for the black section
    output_frame = Frame(output_window, bg="black")
    output_frame.pack(fill="both", expand=True)

    output_text_widget = Text(
        output_frame,
        wrap="word",
        bg="black",
        fg="white"
    )
    output_text_widget.pack(fill="both", expand=True)


    def update_output():
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_text_widget.insert("end", line)
            output_text_widget.see("end")  # Scroll to the end

    Thread(target=update_output, daemon=True).start()

    # Add a label below the output section
    lbl_instruction = Label(output_window, text="Press Esc key to stop analysis and view results", fg="white",bg="blue")
    lbl_instruction.pack()
    output_window.mainloop()

window = Tk()

window.geometry("776x467")
window.configure(bg="#2B2B2B")

canvas = Canvas(
    window,
    bg="#2B2B2B",
    height=467,
    width=776,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    776.0,
    35.0,
    fill="#505F75",
    outline=""
)

def button_1_clicked():
    button_clicked(1)

def button_2_clicked():
    button_clicked(2)

def button_3_clicked():
    button_clicked(3)

button_image_1 = PhotoImage(file="button_1.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=button_1_clicked,
    relief="flat",
    text="Feed Analysis",
    compound="center"
)
button_1.place(x=44.0, y=340.0, width=204.0, height=51.0)

button_image_2 = PhotoImage(file="button_2.png")
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=button_2_clicked,
    relief="flat",
    text="Video Analysis",
    compound="center"
)
button_2.place(x=44.0, y=229.0, width=204.0, height=51.0)

button_image_3 = PhotoImage(file="button_3.png")
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=button_3_clicked,
    relief="flat",
    text="Image Analysis",
    compound="center"
)
button_3.place(x=44.0, y=118.0, width=204.0, height=51.0)

canvas.create_text(
    13.0,
    8.0,
    anchor="nw",
    text="FEED BACK SYSTEM",
    fill="#FFFFFF",
    font=("Inter", 16 * -1)
)

image_image_1 = PhotoImage(file="image_1.png")
image_1 = canvas.create_image(
    523.0,
    254.0,
    image=image_image_1
)

window.mainloop()
