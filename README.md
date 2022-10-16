# LiveSpeechToText

on running the main.py file user encounters "listening to your voice" and the code will translate every speech heard through the microphone untill the execution is terminated.

IMPLEMENTATION DETAILS.

implemented by threading two proccess...one process is responsible to collect any speech audio frame in a queue and the other proccess gives frame buffer from the input queue to wave2vec inference class whihc translates frames to speech and adds the text,confidance,inference time  to output queue.

contains an option to use a custom model trained by me on a medical audio dataset...to use the custom model download the following files and place it in the same folder and set custom model to True.(1)https://drive.google.com/file/d/1iUstBQxqmCtefrqXTuWmM_OD7IEOttDB/view?usp=sharing and (2)https://drive.google.com/drive/folders/15dtFgem6MbMzEXnDflPWXdLYPZ_cdNmW?usp=sharing.
