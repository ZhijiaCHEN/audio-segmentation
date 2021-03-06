import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import sys
if len(sys.argv) != 2:
    print('Illegal input. Useage: \n\tpython video-segmentation.py target')
    exit()
file = sys.argv[1]
clip = mp.VideoFileClip(file)
audio = clip.audio
fps = 100
sampleRate = audio.fps//fps
audioArray = audio.to_soundarray()[::sampleRate]
absSignal = [abs(x[0]) + abs(x[1]) for x in audioArray]

def average_filter(signal, winSize):
    """Filter the input signal using running window average

    Args:
        signal ([list]): [input signal]
        winSize ([int]): [window size]

    Returns:
        [list]: [output signal]
    """
    return np.convolve(signal, np.ones(winSize), 'same') / winSize

def max_filter(signal, winSize):
    """Filter the input signal using running window maximum

    Args:
        signal ([list]): [input signal]
        winSize ([int]): [window size]

    Returns:
        [list]: [output signal]
    """
    ret = [0] * len(signal)
    for i in range(len(signal)):
        ret[i] = max(signal[i:i+winSize])
    return ret

winSize = int(0.5 * fps) # set average window size to 0.5s
absSignal = average_filter(absSignal, winSize)

# absSignal = max_filter(absSignal, winSize)
time = np.array(range(1, len(absSignal) + 1)) * (1/fps/60)

# detect silence
sth = 0.001# silence threshold
silence = [i for i, x in enumerate(absSignal) if x <= sth]

# merge silence intervals, and cut the video at the medium position of each merged silent interval
cut = []
sIdx = 0
for eIdx in range(1, len(silence)):
    if (silence[eIdx] - silence[eIdx - 1])/fps > 10: # merge silent intervals that are closer than 10s 
        cut.append(sum(silence[sIdx:eIdx])/(eIdx - sIdx))
        sIdx = eIdx
cut.append(sum(silence[sIdx:eIdx])/(eIdx - sIdx))

cnt = 0
for sIdx, eIdx in zip(cut[:-1], cut[1:]):
    ts, te = sIdx/fps, eIdx/fps
    cnt += 1
    ffmpeg_extract_subclip(file, ts, te, targetname="{}-{}.mp4".format('.'.join(file.split('.')[:-1]), cnt))


