from google.cloud import texttospeech
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'polyu-wkwf-99b942e126ff.json'
# For hungry, use ttsH(); for tired, use ttsT()

def ttsH():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="我现在有点饿了，可以给我喂食吗？")

    voice = texttospeech.VoiceSelectionParams(
        # language_code='en-gb',
        # name='en-GB-Standard-A',
        language_code='zh-CN',
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1)
    #  valid speaking_rate is between 0.25 and 4.0
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config)
    with open('SpeechCache.wav', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)

def ttsT():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="我现在很疲惫，可以让我休息一下吗？")

    voice = texttospeech.VoiceSelectionParams(
        # language_code='en-gb',
        # name='en-GB-Standard-A',
        language_code='zh-CN',
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1)
    #  valid speaking_rate is between 0.25 and 4.0
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config)
    with open('SpeechCache.wav', 'wb') as out:
        out.write(response.audio_content)
        print('Audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)

# ttsH()
# ttsT()
