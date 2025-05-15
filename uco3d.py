# import necessary libraries
import os
import cv2

# renaming the rgb video subfolders
root_dir = './data/rgb_furniture'

for furniture_type in os.listdir(root_dir):
    furniture_path = os.path.join(root_dir, furniture_type)
    if not os.path.isdir(furniture_path):
        continue

    subfolders = [f for f in os.listdir(furniture_path) if os.path.isdir(os.path.join(furniture_path, f))]
    subfolders.sort()  # optional: to make ordering consistent

    for i, folder in enumerate(subfolders, 1):
        old_path = os.path.join(furniture_path, folder)
        new_name = f"{i:02d}"
        new_path = os.path.join(furniture_path, new_name)

        # Avoid overwriting if new folder name already exists
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# renaming the masked video subfolders
root_dir = './data/masked_furniture'

for furniture_type in os.listdir(root_dir):
    furniture_path = os.path.join(root_dir, furniture_type)
    if not os.path.isdir(furniture_path):
        continue

    subfolders = [f for f in os.listdir(furniture_path) if os.path.isdir(os.path.join(furniture_path, f))]
    subfolders.sort()  # optional: to make ordering consistent

    for i, folder in enumerate(subfolders, 1):
        old_path = os.path.join(furniture_path, folder)
        new_name = f"{i:02d}"
        new_path = os.path.join(furniture_path, new_name)

        # Avoid overwriting if new folder name already exists
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
# check the duration of each data
def get_duration_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

rgb_path = "./data/rgb_furniture/coffee_table/01/rgb_video.mp4"
print(f"RGB Video: {get_duration_opencv(rgb_path)} seconds")
mask_path = "./data/masked_furniture/coffee_table/01/mask_video.mkv"
print(f"Masked Video: {get_duration_opencv(mask_path)} seconds")

# extract 10 rgb frames for each furniture
input_root = './data/rgb_furniture'
output_root = './data/rgb_furniture_frames'

for category in os.listdir(input_root):
    category_path = os.path.join(input_root, category)
    if not os.path.isdir(category_path):
        continue

    for video_folder in os.listdir(category_path):
        folder_path = os.path.join(category_path, video_folder)
        if not os.path.isdir(folder_path):
            continue

        video_path = os.path.join(folder_path, 'rgb_video.mp4')
        if not os.path.isfile(video_path):
            continue

        # Create output directory
        output_folder = os.path.join(output_root, category, video_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // 10, 1)

        for i in range(10):
            frame_idx = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(output_folder, f"frame_{i+1:02d}.jpg")
                cv2.imwrite(out_path, frame)
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to read frame {frame_idx} in {video_path}")

        cap.release()

# extract 10 masked frames for each furniture
input_root = './data/masked_furniture'
output_root = './data/masked_furniture_frames'

for category in os.listdir(input_root):
    category_path = os.path.join(input_root, category)
    if not os.path.isdir(category_path):
        continue

    for video_folder in os.listdir(category_path):
        folder_path = os.path.join(category_path, video_folder)
        if not os.path.isdir(folder_path):
            continue

        video_path = os.path.join(folder_path, 'mask_video.mkv')
        if not os.path.isfile(video_path):
            continue

        # Create output directory
        output_folder = os.path.join(output_root, category, video_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // 10, 1)

        for i in range(10):
            frame_idx = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(output_folder, f"frame_{i+1:02d}.jpg")
                cv2.imwrite(out_path, frame)
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to read frame {frame_idx} in {video_path}")

        cap.release()
