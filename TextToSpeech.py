from google.cloud import texttospeech
from playsound import playsound
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'polyu-wkwf-99b942e126ff.json'
# For medicine remind, use take_medicine_notice_audio()
# For hungry, use hungry_audio()
# For tired, use tiring_audio()
# For dialog starting, use dialog_starting_audio()
# For dialog closing, use dialog_closing_audio()


def take_medicine_notice_audio():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="你好，现在该吃药了！")

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
        print('Hungry audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)


def hungry_audio():
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
    with open('./SpeechCache.wav', 'wb') as out:
        out.write(response.audio_content)
        print('Hungry audio content written to file "SpeechCache.wav"')
    file = './SpeechCache.wav'
    os.system(file)


def tiring_audio():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="我现在很疲惫，想休息一下了")

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
        print('Tiring audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)


def dialog_starting_audio():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="语音对话已经启动！一起聊天吧！")

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
        print('dialog starting audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)


def dialog_closing_audio():
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(
        text="语音对话未启动！请等待鱼面对摄像头后请重试。")

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
        print('dialog closing audio content written to file "SpeechCache.wav"')
    file = 'SpeechCache.wav'
    os.system(file)
