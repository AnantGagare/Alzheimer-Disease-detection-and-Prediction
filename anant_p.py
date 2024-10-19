import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import sqlite3

# Create main window
root = tk.Tk()
root.configure(background="seashell2")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Alzheimer Disease Detection & Prediction")

# Load and place background image
image2 = Image.open('anant-1.jpg')
image2 = image2.resize((750, 830), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image, bd=5)
background_label.image = background_image
background_label.place(x=0, y=0)

# Main Frame
frame_alpr = tk.LabelFrame(root, text="Welcome", width=675, height=830, bd=5, font=('times', 14, ' bold '), bg="#271983", fg="white")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=670, y=0)

# Header
lbl = tk.Label(root, text="Alzheimer Disease Detection & Prediction", font=('Elephant', 30, ' bold '), bg="White", fg="Black")
lbl.place(x=710, y=30)

# Messages
message_list = [
    ("Thought those with ", 190, 100),
    ("Alzheimer's might forget", 240, 140),
    ("us, We as a society", 160, 180),
    ("must remember them", 210, 220),
]
for text, x, y in message_list:
    lbl = tk.Label(frame_alpr, text=text, font=('Lucida Calligraphy', 15, ' bold '), bg="#271983", fg="white")
    lbl.place(x=x, y=y)

# Create SQLite database and table for users
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT, password TEXT)''')
conn.commit()

# Functions for registration and login windows
def register_user():
    username = entry_username.get()
    password = entry_password.get()

    if username and password:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        messagebox.showinfo("Success", "Registration Successful!")
        reg_window.destroy()
    else:
        messagebox.showerror("Error", "Please enter both fields.")

def login_user():
    username = entry_username.get()
    password = entry_password.get()

    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()

    if result:
        messagebox.showinfo("Success", "Login Successful!")
        login_window.destroy()
    else:
        messagebox.showerror("Error", "Invalid credentials. Please try again.")

def register():
    global reg_window, entry_username, entry_password
    reg_window = tk.Toplevel(root)
    reg_window.title("Sign Up")
    reg_window.geometry("300x250")

    label_username = tk.Label(reg_window, text="Username")
    label_username.pack(pady=10)
    entry_username = tk.Entry(reg_window)
    entry_username.pack(pady=5)

    label_password = tk.Label(reg_window, text="Password")
    label_password.pack(pady=10)
    entry_password = tk.Entry(reg_window, show='*')
    entry_password.pack(pady=5)

    register_button = tk.Button(reg_window, text="Register", command=register_user)
    register_button.pack(pady=20)

def login():
    global login_window, entry_username, entry_password
    login_window = tk.Toplevel(root)
    login_window.title("Login")
    login_window.geometry("300x250")

    label_username = tk.Label(login_window, text="Username")
    label_username.pack(pady=10)
    entry_username = tk.Entry(login_window)
    entry_username.pack(pady=5)

    label_password = tk.Label(login_window, text="Password")
    label_password.pack(pady=10)
    entry_password = tk.Entry(login_window, show='*')
    entry_password.pack(pady=5)

    login_button = tk.Button(login_window, text="Login", command=login_user)
    login_button.pack(pady=20)

# Buttons
button1 = tk.Button(frame_alpr, text="SIGN UP", command=register, width=15, height=1, font=('times', 15, ' bold '), bg="#3BB9FF", fg="white")
button1.place(x=250, y=350)

button2 = tk.Button(frame_alpr, text="LOGIN", command=login, width=15, height=1, font=('times', 15, ' bold '), bg="#3BB9FF", fg="white")
button2.place(x=250, y=450)

# Run the main window loop
root.mainloop()
