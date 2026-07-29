import os
import cv2
import numpy as np

def create_demo_videos():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_videos")
    os.makedirs(out_dir, exist_ok=True)
    
    path1 = os.path.join(out_dir, "cctv_fl1.mp4")
    path2 = os.path.join(out_dir, "cctv_fl2.mp4")

    w, h = 1280, 720
    fps = 25
    num_frames = 250  # 10 seconds simulation video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw1 = cv2.VideoWriter(path1, fourcc, fps, (w, h))
    vw2 = cv2.VideoWriter(path2, fourcc, fps, (w, h))

    print(f"[create_cctv_demo] Generating realistic synthetic CCTV videos in {out_dir}...")

    # People simulation tracks
    # Person A walks from Entrance (200, 200) to Kasir (500, 300) then to Stairs (1050, 580) on FL1
    # Person A then emerges on FL2 at Stairs (200, 200) and walks to Table (800, 400)!

    for i in range(num_frames):
        t = i / fps  # Current time in seconds

        # --- CAMERA 1 (FL1: Kasir & Tangga) ---
        f1 = np.ones((h, w, 3), dtype=np.uint8) * 45  # Dark floor tile
        
        # Grid lines for floor tile texture
        for y in range(0, h, 80):
            cv2.line(f1, (0, y), (w, y), (55, 55, 55), 1)
        for x in range(0, w, 80):
            cv2.line(f1, (x, 0), (x, h), (55, 55, 55), 1)

        # Kasir Counter Area
        cv2.rectangle(f1, (400, 100), (700, 250), (90, 70, 50), -1)
        cv2.putText(f1, "KASIR / ORDER COUNTER", (430, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # Seating Area
        cv2.rectangle(f1, (100, 400), (350, 650), (70, 70, 70), -1)
        cv2.putText(f1, "SEATING AREA 1", (140, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Stairs Transition Zone Polygon (TZ_STAIRS_FL1: x 70%-95%, y 60%-95%)
        tz1 = np.array([[int(w*0.70), int(h*0.60)], [int(w*0.95), int(h*0.60)], [int(w*0.95), int(h*0.95)], [int(w*0.70), int(h*0.95)]], np.int32)
        cv2.fillPoly(f1, [tz1], (30, 80, 100))
        cv2.polylines(f1, [tz1], True, (0, 200, 255), 2)
        cv2.putText(f1, "TRANSITION ZONE (AREA TANGGA FL1)", (int(w*0.71), int(h*0.78)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Person A (Customer - Red shirt) on FL1
        # Frames 0-100: Walks to Kasir, Frames 100-150: Order, Frames 150-200: Walks to Stairs
        if i <= 150:
            ratio = min(1.0, i / 100.0)
            pa_x = int(150 + ratio * 350)
            pa_y = int(500 - ratio * 220)
        else:
            ratio = min(1.0, (i - 150) / 50.0)
            pa_x = int(500 + ratio * 550)
            pa_y = int(280 + ratio * 280)

        if i <= 200:
            # Draw Person A top-down view (Head + Shoulders)
            cv2.circle(f1, (pa_x, pa_y), 32, (0, 0, 220), -1)
            cv2.circle(f1, (pa_x, pa_y), 18, (30, 30, 30), -1)  # Head
            # Draw human bounding box representation for detector
            cv2.rectangle(f1, (pa_x - 35, pa_y - 70), (pa_x + 35, pa_y + 70), (0, 140, 255), 3)

        # Person B (Standing customer near Table)
        pb_x, pb_y = 220, 480
        cv2.circle(f1, (pb_x, pb_y), 30, (200, 100, 0), -1)
        cv2.circle(f1, (pb_x, pb_y), 16, (30, 30, 30), -1)
        cv2.rectangle(f1, (pb_x - 30, pb_y - 60), (pb_x + 30, pb_y + 60), (0, 140, 255), 3)


        # --- CAMERA 2 (FL2: Seating & Tangga) ---
        f2 = np.ones((h, w, 3), dtype=np.uint8) * 40  # Dark wooden floor tile

        # Grid lines
        for y in range(0, h, 80):
            cv2.line(f2, (0, y), (w, y), (50, 50, 50), 1)
        for x in range(0, w, 80):
            cv2.line(f2, (x, 0), (x, h), (50, 50, 50), 1)

        # Stairs Exit Zone Polygon (TZ_STAIRS_FL2: x 5%-30%, y 5%-40%)
        tz2 = np.array([[int(w*0.05), int(h*0.05)], [int(w*0.30), int(h*0.05)], [int(w*0.30), int(h*0.40)], [int(w*0.05), int(h*0.40)]], np.int32)
        cv2.fillPoly(f2, [tz2], (30, 80, 100))
        cv2.polylines(f2, [tz2], True, (0, 200, 255), 2)
        cv2.putText(f2, "TRANSITION ZONE (AREA TANGGA FL2)", (int(w*0.06), int(h*0.22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Seating Area FL2
        cv2.rectangle(f2, (500, 200), (1050, 600), (70, 70, 70), -1)
        cv2.putText(f2, "SOFA SEATING AREA FL2", (650, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Person A appears on FL2 after leaving FL1 transition zone (Frame 180 onwards)
        if i >= 180:
            ratio = min(1.0, (i - 180) / 60.0)
            pa2_x = int(180 + ratio * 600)
            pa2_y = int(180 + ratio * 220)

            # Draw Person A (Same Red Shirt visual features for ReID matching!)
            cv2.circle(f2, (pa2_x, pa2_y), 32, (0, 0, 220), -1)
            cv2.circle(f2, (pa2_x, pa2_y), 18, (30, 30, 30), -1)
            cv2.rectangle(f2, (pa2_x - 35, pa2_y - 70), (pa2_x + 35, pa2_y + 70), (0, 140, 255), 3)

        # Person C (Seated on FL2)
        pc_x, pc_y = 850, 350
        cv2.circle(f2, (pc_x, pc_y), 30, (0, 180, 100), -1)
        cv2.circle(f2, (pc_x, pc_y), 16, (30, 30, 30), -1)
        cv2.rectangle(f2, (pc_x - 30, pc_y - 60), (pc_x + 30, pc_y + 60), (0, 140, 255), 3)

        vw1.write(f1)
        vw2.write(f2)

    vw1.release()
    vw2.release()
    print(f"[create_cctv_demo] CCTV demo videos generated successfully:")
    print(f"  - Cam 1 (FL1): {path1}")
    print(f"  - Cam 2 (FL2): {path2}")

if __name__ == "__main__":
    create_demo_videos()
