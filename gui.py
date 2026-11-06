import tkinter as tk                      # Python's built-in window toolkit
import cv2                                # OpenCV: webcam + drawing
from PIL import Image, ImageTk            # turns an OpenCV frame into a picture Tkinter can show
from ultralytics import YOLO

cam = cv2.VideoCapture(0)                 # 0 = the laptop's built-in webcam
model = YOLO("yolov8n.pt")

root = tk.Tk()                            # the window
root.title("Live object detection")

video = tk.Label(root)                    # a Label can hold text or a picture; the webcam goes here
video.pack()                              # pack() places it in the window, top to bottom
cutoff = tk.Scale(root, from_=0.05, to=0.95, resolution=0.05,
                  orient="horizontal", length=300, label="Confidence cutoff")
cutoff.set(0.55)
cutoff.pack()

def update():                             # shows ONE frame, then schedules itself again
    ok, frame = cam.read()
    if ok:
        result = model(frame, conf=cutoff.get(), verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.4f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # OpenCV stores colours as BGR; Tkinter wants RGB
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))    # frame -> a picture Tkinter can display
        video.configure(image=photo)                        # put the picture in the Label
        video.image = photo                                 # keep a reference, or Python deletes the picture
    root.after(15, update)                # run update again in 15 ms

def close():                              # runs when you click the window's X
    cam.release()                         # free the webcam
    root.destroy()                        # close the window

root.protocol("WM_DELETE_WINDOW", close)  # "when the X is clicked, call close"
update()                                  # show the first frame
root.mainloop()                           # Tkinter's own loop: keeps the window alive and responsive
