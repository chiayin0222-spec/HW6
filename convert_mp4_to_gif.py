import cv2
from PIL import Image

def convert_mp4_to_gif(mp4_path, gif_path, target_width=640, fps=10):
    print(f"[INFO] Loading video: {mp4_path}")
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return
    
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not original_fps or original_fps <= 0:
        original_fps = 25
        
    frame_step = max(1, int(original_fps / fps))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            target_height = int(h * (target_width / w))
            img = Image.fromarray(frame_rgb)
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            frames.append(img)
            
        count += 1
        
    cap.release()
    
    if frames:
        print(f"[INFO] Saving animated GIF to: {gif_path} (Total frames: {len(frames)})")
        duration = int(1000 / fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration,
            loop=0
        )
        print(f"[SUCCESS] GIF generated successfully: {gif_path}")
    else:
        print("[ERROR] No frames extracted from video.")

if __name__ == "__main__":
    convert_mp4_to_gif("tutorial_slideshow.mp4", "tutorial_slideshow.gif")
