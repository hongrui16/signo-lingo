import subprocess
import av

def fix_video(input_path, output_path):
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-codec', 'copy',
        '-movflags', 'faststart',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print("Failed to fix video:", e)
        return None

def open_video(vid_path):
    try:
        container = av.open(vid_path)
        return container
    except av.error.InvalidDataError as e:
        print("Invalid data found when processing input:", e)
        # 修复视频
        output_path = vid_path.replace(".mp4", "_fixed.mp4")
        fixed_path = fix_video(vid_path, output_path)
        if fixed_path:
            try:
                container = av.open(fixed_path)
                return container
            except av.error.InvalidDataError as e:
                print("Still failed to open video after fixing:", e)
                return None
        else:
            return None

# 使用函数
vid_path = '/scratch/rhong5/dataset/signLanguage/AUTSL/train/signer3_sample152_color.mp4'
container = open_video(vid_path)
if container:
    stream = container.streams.video[0]
    n_frames = stream.frames
    # 进行后续处理
else:
    print("Unable to open video file.")
