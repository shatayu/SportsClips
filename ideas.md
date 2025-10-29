* seems like main bottleneck is OCR
* idea for this v1: input should be youtube video and in-game time (eg q2 6:19) and output should be 10 second clip beginning at Q2 6:19
* once the video is downloaded binary search for Q2 6:19; take the middle frame, OCR it, then based on its output binary search until you find the 6:19 frame
* then when you find that frame figure out whatever info you need to cut it and the next 10 seconds from the original video or whatever