import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO
from cnn.inference import classify_crop

from src.rl_agent import ThresholdRLAgent
from src.rl_environment import compute_reward
from src.nlp_module import generate_report

# Loading Model and Agent
model = YOLO("models/yolo_ppe_final.pt")
agent = ThresholdRLAgent()

# NLP Analysis and Report Generation
def format_nlp_report(counts, total):
    no_helmet = counts.get("no_helmet", 0) + counts.get("head", 0)
    helmet = counts.get("helmet", 0)
    
    if total == 0:
        return ["STATUS: STANDBY", "No personnel detected", "Monitoring area..."]
    
    if no_helmet > 0:
        return ["STATUS: VIOLATION", f"Unsafe: {no_helmet} person(s) without helmet", "Issue PPE immediately!"]
    
    return ["STATUS: SECURE", f"Safe: {helmet} person(s)", "All requirements met."]

# Sidebar UI
def apply_sidebar(frame, title, stats, report_lines):
    h, w, _ = frame.shape
    sidebar_w = 260 
    sidebar = np.full((h, sidebar_w, 3), (30, 30, 30), dtype=np.uint8) 
    
    cv2.putText(sidebar, title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(sidebar, (15, 50), (sidebar_w-15, 50), (70, 70, 70), 1)
    
    cv2.putText(sidebar, "SYSTEM TELEMETRY", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(sidebar, stats, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    y_offset = 140
    cv2.putText(sidebar, "IMAGE ANALYSIS", (15, y_offset-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    status_text = report_lines[0]
    status_color = (75, 255, 75) if "SECURE" in status_text else (75, 75, 255)
    
    for i, line in enumerate(report_lines):
        color = status_color if i == 0 else (220, 220, 220)
        font_scale = 0.5 if i == 0 else 0.45
        thickness = 1
        
        max_txt_width = sidebar_w - 30
        words = line.split(' ')
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            (line_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            if line_w > max_txt_width:
                cv2.putText(sidebar, current_line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                y_offset += 25
                current_line = word + " "
            else:
                current_line = test_line
        
        cv2.putText(sidebar, current_line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 35 

    cv2.putText(sidebar, "ESC: Return to Menu", (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    
    return np.hstack((sidebar, frame))

# Image Mode with NLP Reporting
def run_image_mode(image_path=None):
    if not image_path:
        image_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
    if not image_path: return

    frame = cv2.imread(image_path)
    if frame is None: return

    threshold = agent.get_best_threshold()
    results = model(frame, conf=threshold)[0]
    counts = {}

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label = classify_crop(crop)
        counts[label] = counts.get(label, 0) + 1
        
        display_label = "NO HELMET" if label in ["head", "no_helmet"] else "HELMET"
        color = (75, 255, 75) if display_label == "HELMET" else (75, 75, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    report_lines = format_nlp_report(counts, sum(counts.values()))
    combined = apply_sidebar(frame, "IMAGE ANALYSIS", f"RL-Threshold: {threshold:.2f}", report_lines)
    
    cv2.imshow("SafetyScan", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    main_menu()

# Webcam Mode with Real-Time RL Thresholding and NLP Reporting
def run_webcam_mode():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        threshold = agent.choose_action()
        results = model(frame, conf=threshold)[0]

        counts = {}
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label = classify_crop(crop)
            counts[label] = counts.get(label, 0) + 1

            detections.append({"confidence": 1.0})
            
            display_label = "NO HELMET" if label in ["head", "no_helmet"] else "HELMET"
            color = (75, 255, 75) if display_label == "HELMET" else (75, 75, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        reward = compute_reward(detections)
        agent.update(threshold, reward)
        agent.decay_epsilon()

        report_lines = format_nlp_report(counts, sum(counts.values()))
        combined = apply_sidebar(frame, "LIVE MONITOR", f"RL-Thresh: {threshold:.2f}", report_lines)
        
        cv2.imshow("SafetyScan", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # RL Analysis and Visualization
    agent.plot_learning()
    print("Best threshold learned:", agent.get_best_threshold())

    main_menu()

# GUI Menu
def main_menu():
    root = tk.Tk()
    root.title("SafetyScan")
    root.geometry("400x380")
    root.configure(bg="#1e1e1e")

    tk.Label(root, text="PPE HELMET DETECTION", font=("Helvetica", 18, "bold"),
             fg="#ffffff", bg="#1e1e1e", pady=25).pack()

    def btn_cmd(cmd):
        root.destroy()
        cmd()

    btn_style = {
        "width": 25,
        "font": ("Helvetica", 11),
        "bg": "#2b2b2b",
        "fg": "white",
        "relief": "flat",
        "pady": 12,
        "cursor": "hand2"
    }
    
    tk.Button(root, text="Launch Webcam",
              command=lambda: btn_cmd(run_webcam_mode), **btn_style).pack(pady=10)

    tk.Button(root, text="Upload Image File",
              command=lambda: btn_cmd(run_image_mode), **btn_style).pack(pady=10)

    tk.Button(root, text="Exit System",
              command=root.quit, fg="#ff5555",
              bg="#1e1e1e", relief="flat").pack(side="bottom", pady=20)

    root.mainloop()

if __name__ == "__main__":
    main_menu()