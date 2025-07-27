"""
>>> from sound.audio import list_audio_devices
>>> list_audio_devices()


"""
import pyaudio


def list_audio_devices():
    mics = []
    indices = []
    infos = []

    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        try:
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                mics.append(p.get_device_info_by_index(i).get('name'))
                indices.append(i)
                infos.append(p.get_device_info_by_index(i))
        except OSError:
            pass

    return mics, indices, infos, p


def default_audio_device():
    pass





