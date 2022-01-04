import os
import cv2
import pysrt
import argparse
import face_recognition
from tqdm.auto import tqdm

files = []

def dump_frames(vid_path, frame_output_path):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    pbar = tqdm()
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            cv2.imwrite(f"{frame_output_path}/{timestamps[-1]}.jpg", curr_frame)
            pbar.update(1)
        else:
            break
    cap.release()

chamath = face_recognition.load_image_file('chamath.jpg')
sacks = face_recognition.load_image_file('sacks.jpg')
friedberg = face_recognition.load_image_file('friedberg.jpg')
jason = face_recognition.load_image_file('jason.jpg')

chamath_encoding = face_recognition.face_encodings(chamath)[0]
sacks_encoding = face_recognition.face_encodings(sacks)[0]
friedberg_encoding = face_recognition.face_encodings(friedberg)[0]
jason_encoding = face_recognition.face_encodings(jason)[0]

def who_is_speaking(image_path):
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces([chamath_encoding, sacks_encoding, friedberg_encoding, jason_encoding], unknown_encoding)
    mp = {
        0: "chamath",
        1: "sacks",
        2: "friedberg",
        3: "jason"
    }
    for i, r in enumerate(results):
        if r:
            return mp[i]
    return "none"

def get_in_milliseconds(subriptime):
    return subriptime.hours * 3600000  + subriptime.minutes * 60000 + subriptime.seconds * 1000 + subriptime.milliseconds

def get_files(start, end):
    ret = []
    for file in files:
        if float(start) <= float(file[4:-4])  and float(file[4:-4]) <= float(end):
            ret.append(file)
    return ret

def dump_transcript(sub_path):
    subs = pysrt.open(sub_path)
    for sub in tqdm(subs):
        startms = get_in_milliseconds(sub.start)
        endms = get_in_milliseconds(sub.end)
        time_files = get_files(startms, endms)
        majority_speaker = {}
        for file in time_files:
            try:
                speaker = who_is_speaking(file)
                majority_speaker[speaker] = majority_speaker.get(speaker, 0) + 1
            except:
                pass
        if not len(majority_speaker):
            major_speaker = "None"
        else:
            major_speaker = max(majority_speaker.items(), key=lambda m: m[1])[0]
        sub.__dict__['speaker'] = major_speaker
        print(major_speaker.title() + ": " + sub.text)


def process(args):
    global files
    os.mkdir(args.frames_path)
    dump_frames(args.video_path, args.frames_path)
    files = os.listdir(args.frames_path)
    files = sorted(files, key=lambda m: float(m[:-4]))
    files = [os.path.join(args.frames_path, i) for i in files]
    dump_transcript(args.subtitle_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump transcript", prefix_chars="-")
    parser.add_argument("-video_path", required=True, help="video path", type=str)
    parser.add_argument("-subtitle_path", required=True, help="subtitle path", type=str)
    parser.add_argument("-frames_path", required=True, help="frames path", type=str)
    parser.add_argument("-output_path", required=True, help="output path", type=str)
    args = parser.parse_args()
    process(args)
