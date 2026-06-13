import os
import sys
import time
import threading
import http.server
import socketserver
import shutil

PORT = 8009  # Port to serve the slideshow locally

def start_server():
    """Starts a simple HTTP server to serve the slideshow."""
    class QuietSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress console logging to keep terminal output clean
            pass

    Handler = QuietSimpleHTTPRequestHandler
    # Allow address reuse to prevent port-in-use errors
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"[INFO] Local server started on port {PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[ERROR] Failed to start local server: {e}")

# Start HTTP server in a background daemon thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(1)  # Give the server a moment to spin up

# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("[ERROR] 'playwright' is not installed. Please run: pip install playwright && playwright install")
    sys.exit(1)

def record():
    print("[INFO] Launching headless browser for recording...")
    with sync_playwright() as p:
        # Launch headless Chromium
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            print(f"[ERROR] Failed to launch Chromium: {e}")
            print("[INFO] Trying to install Chromium browser dependencies...")
            os.system("playwright install chromium")
            browser = p.chromium.launch(headless=True)

        # Create a new browser context with video recording enabled
        # We target 1280x720 (720p HD) for a high-quality widescreen video
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=".",
            record_video_size={"width": 1280, "height": 720}
        )

        page = context.new_page()
        slideshow_url = f"http://localhost:{PORT}/tutorial_slideshow.html"
        print(f"[INFO] Loading slideshow from: {slideshow_url}")
        page.goto(slideshow_url)

        # Wait for resources and animations to load
        time.sleep(2)

        # Click the start button on Slide 1
        print("[INFO] Starting slideshow...")
        start_btn = page.query_selector(".btn-start")
        if start_btn:
            start_btn.click()
            time.sleep(1.5)  # Wait for transition animation

        # Total slides: 6
        # We will cycle through all slides, simulating clicks and interactions where relevant
        for slide_idx in range(1, 7):
            print(f"[INFO] Recording Slide {slide_idx} / 6...")

            if slide_idx == 4:
                # Slide 4: Feature selection schemes. Click different scheme cards to show dynamic details.
                schemes = ["pearson", "rfe", "lasso", "manual"]
                for sch in schemes:
                    print(f"       Interacting: Displaying feature selection scheme '{sch}'")
                    page.evaluate(f"showScheme('{sch}')")
                    time.sleep(1.8)  # Stay on the detailed tab for a bit
            elif slide_idx == 6:
                # Slide 6: Profit Predictor. Simulating slider adjustments to show dynamic ML profit updates.
                print("       Interacting: Simulating slider values updates (Profit Prediction)...")
                # Move RD slider up
                page.evaluate("document.getElementById('sliderRD').value = 180000; calculateProfit();")
                time.sleep(2.0)
                # Move Marketing slider up
                page.evaluate("document.getElementById('sliderMkt').value = 400000; calculateProfit();")
                time.sleep(2.0)
                # Reset RD slider down
                page.evaluate("document.getElementById('sliderRD').value = 60000; calculateProfit();")
                time.sleep(2.0)
            else:
                # Standard slides: Hold static view for 5 seconds to allow reading
                time.sleep(5.0)

            # Move to next slide if not on the last slide
            if slide_idx < 6:
                page.keyboard.press("ArrowRight")
                time.sleep(1.2)  # Wait for page transition slide animation to complete

        # Hold on final slide view for 1 second
        time.sleep(1.0)

        # Retrieve recorded video path before closing context
        raw_video_path = page.video.path()
        context.close()
        browser.close()

        print(f"[INFO] Browser recording complete.")
        print(f"[INFO] Raw video saved at: {raw_video_path}")

        # Rename the output video file to a user-friendly name in the root folder
        target_video_name = "tutorial_slideshow.webm"
        target_video_path = os.path.join(os.getcwd(), target_video_name)

        if os.path.exists(raw_video_path):
            if os.path.exists(target_video_path):
                os.remove(target_video_path)
            shutil.move(raw_video_path, target_video_path)
            print(f"[SUCCESS] High-quality slideshow video saved successfully to: {target_video_path}")
            
            # Create an additional copy as .mp4 for compatibility (retaining WebM container format)
            # Many modern media players can play WebM files even if renamed to .mp4.
            mp4_copy_path = os.path.join(os.getcwd(), "tutorial_slideshow.mp4")
            if os.path.exists(mp4_copy_path):
                os.remove(mp4_copy_path)
            shutil.copy2(target_video_path, mp4_copy_path)
            print(f"[SUCCESS] Compatibility copy created: {mp4_copy_path}")
        else:
            print("[ERROR] Playwright video output file could not be found.")

if __name__ == "__main__":
    record()
