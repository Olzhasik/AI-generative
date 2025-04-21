import tkinter as tk
from tkinter import messagebox, ttk
from diffusers import StableDiffusionPipeline
from PIL import ImageTk, Image


def callback_fn(step, timestep, latents):
    percent = int((step / num_inference_steps) * 100)
    progress_var.set(percent)
    progress_label.config(text=f"Жүктелуде: {percent}%")
    root.update()

num_inference_steps = 50

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")


def generate_image():
    prompt = entry.get()
    if not prompt:
        messagebox.showwarning("Ескерту", "Сипаттама енгізіңіз!")
        return
    try:
        progress_label.pack()
        progress_bar.pack()
        progress_var.set(0)

        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            callback=callback_fn,
            callback_steps=1
        ).images[0]

        image.save("generated_image.png")
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

        progress_label.pack_forget()
        progress_bar.pack_forget()

        messagebox.showinfo("Жаңарту", "Сурет сәтті генерацияланды!")
    except Exception as e:
        messagebox.showerror("Қате", f"Сурет генерациялау кезінде қате болды: {e}")


root = tk.Tk()
root.title("Сурет генерациялау")
root.geometry("400x500")

label_prompt = tk.Label(root, text="Сурет сипаттамасын енгізіңіз:")
label_prompt.pack(pady=10)

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

progress_var = tk.DoubleVar()
progress_label = tk.Label(root, text="Жүктелуде: 0%")
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=200)

progress_label.pack_forget()
progress_bar.pack_forget()

label = tk.Label(root)
label.pack(pady=20)

generate_button = tk.Button(root, text="Сурет генерациялау", command=generate_image)
generate_button.pack(pady=10)

root.mainloop()
