# datasets/video_reader.py
#
# WHAT THIS FILE DOES:
#   Opens a video file and gives you one frame at a time with a timestamp.
#
# WHY IT EXISTS:
#   opencv's VideoCapture works fine but has a verbose API.
#   This wraps it so you can just write:
#       for frame, t in VideoReader("clip.mp4"):
#           ...
#   instead of writing the while/read/release loop everywhere.

import cv2


class VideoReader:
    """
    Opens a video file and lets you loop over its frames.

    Each loop step gives you:
        frame        : numpy array (H x W x 3) — a BGR image
        timestamp    : float — seconds since the start of the video

    Usage:
        reader = VideoReader("data/raw_clips/my_clip.mp4")
        for frame, t in reader:
            print(f"Frame at {t:.2f} seconds, shape: {frame.shape}")
    """

    def __init__(self, video_path):
        self.video_path = video_path

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Read frame rate from the file header (e.g. 25.0 = 25 frames per second)
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index  = 0

        print(f"Opened: {video_path}")
        print(f"  FPS: {self.fps}  |  Frames: {self.total_frames}  |  Duration: {self.total_frames / self.fps:.1f}s")

    def __iter__(self):
        # Python calls this to start a for loop — just return self
        return self

    def __next__(self):
        # Python calls this each time the for loop needs the next item
        ok, frame = self.cap.read()
        if not ok:
            # No more frames — stop the for loop
            self.cap.release()
            raise StopIteration

        timestamp = self.frame_index / self.fps
        self.frame_index += 1
        return frame, timestamp

    def release(self):
        """Close the video file early if needed."""
        self.cap.release()
