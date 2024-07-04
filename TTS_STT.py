from gtts import gTTS
import os
# import speech_recognition as sr

def SpeakText(command, language = 'en', saveOnly = False, savefile = "audio_clip"):
    myobj = gTTS(text=command, lang=language, slow=False)
    myobj.save(savefile+".mp3")
    if not saveOnly: os.system("start "+savefile+".mp3")
    return

# def ListenText(pause_threshold=2, verbose = True):	
#     r = sr.Recognizer() 
#     r.pause_threshold = pause_threshold
#     while(1): 
#         try:
#             with sr.Microphone() as source2:
#                 r.adjust_for_ambient_noise(source2, duration=0.2)
#                 audio2 = r.listen(source2)
#                 MyText = r.recognize_google(audio2)
#                 MyText = MyText.lower()
#                 if verbose: print("YOU SAID:", MyText)
#                 return MyText                
#         except sr.RequestError as e:
#             print("Could not request results; {0}".format(e))
#         except sr.UnknownValueError:
#             print("Please Speak")

# print("Speak Now:")
# x = ListenText()        # pause_threshold == How long to wait before stopping listening
# SpeakText(x)