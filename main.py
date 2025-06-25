# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: main.py
# Description: This script demonstrates a basic example of how to use the GazeProcessor class
#              from the LaserGaze project. It sets up the gaze detection system with
#              optional visualization settings and an asynchronous callback for processing
#              gaze vectors. The example provided here can be modified or extended by
#              contributors to fit specific needs or to experiment with different settings
#              and functionalities. It serves as a starting point for developers looking to
#              integrate and build upon the gaze tracking capabilities provided by the
#              GazeProcessor in their own applications.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

from GazeProcessor import GazeProcessor
from VisualizationOptions import VisualizationOptions
import asyncio
import json
from datetime import datetime
import argparse
import os
import subprocess

# Process every nth frame after calibration (1 = process every frame)
PROCESS_EVERY_N_FRAMES = 1

def convert_webm_to_mp4(input_path):
    """Convert WebM to MP4 using ffmpeg."""
    output_path = input_path.replace('.webm', '.mp4')
    if not os.path.exists(output_path):
        print(f"Converting {input_path} to MP4...")
        try:
            # Using subprocess instead of os.system for better error handling
            result = subprocess.run(
                ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-c:a', 'aac', output_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Error during conversion: {result.stderr}")
                return None
            print("Conversion complete.")
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            return None
    return output_path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video file for gaze tracking')
    parser.add_argument('--file_path', type=str, help='Path to the input video file')
    parser.add_argument('--video_folder', type=str, help='Path to folder containing videos to process')
    return parser.parse_args()

def get_output_filename(video_path):
    # Get the base name of the video file without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"gaze_vectors_{video_name}.json"

def get_video_files(folder_path):
    """Get all video files from the specified folder."""
    video_extensions = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
    video_files = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return video_files
        
    for file in os.listdir(folder_path):
        if file.lower().endswith(video_extensions):
            video_files.append(os.path.join(folder_path, file))
    
    return sorted(video_files)

async def process_video(video_file):
    """Process a single video file."""
    # Convert WebM to MP4 if needed
    if video_file.lower().endswith('.webm'):
        converted_file = convert_webm_to_mp4(video_file)
        if converted_file is None:
            print(f"Failed to convert {video_file}, skipping...")
            return
        video_file = converted_file
    
    output_file = get_output_filename(video_file)
    
    # Clear the output file if it exists
    open(output_file, 'w').close()
    
    vo = VisualizationOptions()
    # Initialize GazeProcessor with video file and frame skipping
    gp = GazeProcessor(
        video_file=video_file,
        visualization_options=vo,
        callback=gaze_vectors_collected,
        process_every_n_frames=PROCESS_EVERY_N_FRAMES
    )
    await gp.start()

async def gaze_vectors_collected(left, right, video_file, frame_number):
    """
    Callback function to collect and save gaze vectors
    """
    data = {
        "frame": frame_number,
        "left_vector": left.tolist() if left is not None else None,
        "right_vector": right.tolist() if right is not None else None
    }
    
    output_file = get_output_filename(video_file)
    # Append to file
    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

async def main():
    args = parse_arguments()
    
    if args.video_folder:
        video_files = get_video_files(args.video_folder)
        if not video_files:
            print(f"No video files found in {args.video_folder}")
            return
            
        print(f"Found {len(video_files)} video files to process")
        for i, video_file in enumerate(video_files, 1):
            print(f"\nProcessing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
            await process_video(video_file)
            
    elif args.file_path:
        video_file = args.file_path
        await process_video(video_file)
    else:
        video_file = "tst_screen.MOV"
        await process_video(video_file)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()