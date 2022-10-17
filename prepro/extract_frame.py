import os
from os.path import join
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import numpy as np

def select_frames(frame_dir):
    dir_name = os.path.basename(frame_dir)
    frame_name = os.listdir(frame_dir)
    length = len(frame_name)
    frame_name = sorted(frame_name)
    frame_path = np.array([os.path.join(frame_dir, i) for i in frame_name])
    idx = np.arange(32) * (length // 32)
    frame_path_new = frame_path[idx].tolist()
    frame_path = frame_path.tolist()
    tar_dir = "/home/qinyixin/data/tgifqa" + dir_name
    os.makedirs(tar_dir,exist_ok=True)
    for j in frame_path:
        if j in frame_path_new:
            name = os.path.basename(j)
            tar_path = tar_dir + '/' + name
            os.system("cp {} {}".format(j, tar_path))

def extract_frame_from_video(video_path, save_frame_path, fps=1, num_frames=-1,
                             start_ts=-1, end_ts=-1,
                             suppress_msg=False, other_args="", overwrite=True):
    
    extra_args = " -hide_banner -loglevel panic " if suppress_msg else ""
    extra_args += " -y " if overwrite else ""
    
    if num_frames <= 0 :
        split_cmd_template = "ffmpeg {extra} -i {video_path} -vf fps={fps} {output_frame_path}%06d.jpg"
        
        cur_split_cmd = split_cmd_template.format(
            extra=extra_args, video_path=video_path, fps=fps, output_frame_path=save_frame_path)
    
    try:
        _ = subprocess.run(cur_split_cmd.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error returned by ffmpeg cmd {e}")
        

def extract_frame(video_file_path, save_dir, fps, debug=False, corrupt_files=[]):
    filename = os.path.basename(video_file_path)
    vid, _ = os.path.splitext(filename)
    frame_name = f"{vid}"
    save_dir = save_dir + '/' + frame_name
    frame_save_path = join(save_dir, frame_name)

    if (video_file_path not in corrupt_files and len(corrupt_files)):
        # print(f"skipping {video_file_path}")
        return
    if len(corrupt_files):
        print(f"exracting frames for {video_file_path}")
    launch_extract = True
    if launch_extract:
        os.makedirs(save_dir, exist_ok=True)
        # scale=width:height
        extract_frame_from_video(video_file_path, frame_save_path, fps=fps,
                                 suppress_msg=not debug, other_args="")

def extract_all_frames(video_root_dir, save_dir, fps, num_workers, debug=False):


    videoFiles = [os.path.join(video_root_dir,i) for i in os.listdir(video_root_dir)]

    if num_workers > 0:
        from functools import partial
        extract_frame_partial = partial(
            extract_frame, fps=fps,
            save_dir=save_dir, debug=debug)

        with mp.Pool(num_workers) as pool, tqdm(total=len(videoFiles)) as pbar:
            for idx, _ in enumerate(
                    pool.imap_unordered(
                        extract_frame_partial, videoFiles, chunksize=8)):
                pbar.update(1)
    else:
        for idx, d in tqdm(enumerate(videoFiles),
                           total=len(videoFiles), desc="extracting frames from video"):
            extract_frame(d, save_dir, fps=fps, debug=debug)
            if debug and idx >= 10:
                break

if __name__ == '__main__':
    video_dir = '/mnt/nfs/datasets/CMG/TGIF/'
    save_dir = '/home/qinyixin/data/tgifqa/video_frames'
    fps = 30

    extract_all_frames(video_dir, save_dir, fps, 10, False)
    #frame_dir = os.listdir('/mnt/hdd1/qinyixin/MSRVTT-QA/video_frames')
    #for i in tqdm(frame_dir):
        #path = os.path.join(save_dir, i)
        #select_frames(path)
    