import os
import random

import pyaudio
import wave
import cv2
from playsound import playsound
from google.cloud import dialogflow


class DialogueFlow():
    def __init__(self, project_id='polyu-wkwf', session_id='test', language_code='zh-CN',
                 credential_path='polyu-wkwf-99b942e126ff.json'):
        # initiate google credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        # session information
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(project_id, session_id)
        print("Session path: {}\n".format(self.session))

        # audio config
        # set up the audio configuration for recording the user's voice
        # define the audio format, frequency, chunk, channel and output name
        # the audio format is used for encoding and decoding because the datatype we sent to Google DialogFlow is Byte.
        # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
        _audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
        self.sample_rate_hertz = 16000
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.audio_file_name = "input.wav"
        self.audio_config = dialogflow.InputAudioConfig(
            audio_encoding=_audio_encoding,
            language_code=language_code,
            sample_rate_hertz=self.sample_rate_hertz,
        )

        # turn the fulfillment text into audio and play
        self.output_audio_config = dialogflow.OutputAudioConfig(
            audio_encoding=dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16
        )
        self.input_audio = -1

    def record_voice(self, seconds=3):
        # use the pyaudio and wave library to record the voice
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=self.sample_format,
            channels=self.channels,
            input_device_index=1,
            rate=self.sample_rate_hertz,
            frames_per_buffer=self.chunk,
            input=True
        )

        frames = []  # Initialize array to store frames

        print('Start recording')

        # set the recording time by the seconds variable
        # Store data in chunks for 3 seconds
        for i in range(0, int(self.sample_rate_hertz / self.chunk * seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        pa.terminate()

        print('Finished recording')
        # use the wave library to turn the frames into a WAV file
        # Save the recorded data as a WAV file
        wf = wave.open(self.audio_file_name, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pa.get_sample_size(self.sample_format))
        wf.setframerate(self.sample_rate_hertz)
        wf.writeframes(b''.join(frames))
        wf.close()
        # Open the WAV file and save it to self.input_audio variable
        with open(self.audio_file_name, "rb") as audio_file:
            self.input_audio = audio_file.read()

    def dialogflow_request(self):
        if self.input_audio != -1:
            query_input = dialogflow.QueryInput(audio_config=self.audio_config)

            # add the output_audio_config to the request so that the server will return you a response with audio data
            request = dialogflow.DetectIntentRequest(
                session=self.session, query_input=query_input, input_audio=self.input_audio,
                output_audio_config=self.output_audio_config
            )
            # get response from dialogflow
            response = self.session_client.detect_intent(request=request)

            # print the result
            print("=" * 20)
            print("Query text: {}".format(response.query_result.query_text))
            print(
                "Detected intent: {} (confidence: {})\n".format(
                    response.query_result.intent.display_name,
                    response.query_result.intent_detection_confidence,
                )
            )
            print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))

            # Save the response audio file into a wav file.
            # The response's audio_content is binary.
            with open("output.wav", "wb") as out:
                out.write(response.output_audio)
                print('Audio content written to file "output.wav"')
            return response.query_result.query_text, response.query_result.fulfillment_text

    def play_audio(self):
        # Play the wave file with the pyAudio

        # Init audio stream 
        wf = wave.open("output.wav", 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        # Play entire file
        data = wf.readframes(self.chunk)
        while data != b'':
            stream.write(data)
            data = wf.readframes(self.chunk)
        # Graceful shutdown 
        stream.close()
        p.terminate()

    # use the function in UI button
    def _init_dialogue_flow(self, event=None):
        df = DialogueFlow()
        df.record_voice()
        user_txt, dialogflow_txt = df.dialogflow_request()
        self.change_txt(user_txt, dialogflow_txt)
        df.play_audio()

        if dialogflow_txt == '好的，为你播放北风吹':
            print('now playing: 北风吹')
            playsound('music/beifengchui.mp3')
        elif dialogflow_txt == '好的，为你播放南泥湾':
            print('now playing: 南泥湾')
            playsound('music/nanniwan.mp3')
        elif dialogflow_txt == '好的，这就展示图片':
            print('now showing pictures')
            num = random.randint(1, 5)
            img = cv2.imread('pic/{}.png'.format(num))
            cv2.imshow('风景', img)
            cv2.waitKey(5000)


if __name__ == '__main__':
    os.system('@echo off')
    # Since we are using traditional Chinese in google Dialogflow,
    # the cmd may not display the content well.
    # So we have to change the system encoding method to 936
    os.system('chcp 936 >nul')

    df = DialogueFlow()
    df.record_voice()
    user_txt, dialogflow_txt = df.dialogflow_request()
    df.play_audio()

    if dialogflow_txt == '好的，为你播放北风吹':
        print('now playing: 北风吹')
        playsound('music/beifengchui.mp3')
    elif dialogflow_txt == '好的，为你播放南泥湾':
        print('now playing: 南泥湾')
        playsound('music/nanniwan.mp3')
    elif dialogflow_txt == '好的，这就展示图片':
        print('now showing pictures')
        num = random.randint(1, 5)
        img = cv2.imread('pic/{}.png'.format(num))
        cv2.imshow('风景', img)
        cv2.waitKey(5000)
