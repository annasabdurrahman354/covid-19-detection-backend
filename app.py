from flask import Flask, request, jsonify
import librosa
import io
import soundfile as sf
import numpy as np
import pickle
from os.path import join
from flask_cors import CORS
import scipy.io.wavfile as wav
import base64

app = Flask(__name__)
CORS(app)

with open(join('trained_model.pkl'), 'rb') as f:
	model = pickle.load(f)

def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    cough_mask = np.array([False]*len(x))

    # define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    # segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            # counting and adding cough samples
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            # cough end
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            # reset counter for number of sample tolerance
            else:
                below_th_counter = 0
        else:
            # start cough
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask

def extract_feature(x, sr, clinical_data):
    normalized_signal = librosa.util.normalize(x)

    mfcc = librosa.feature.mfcc(y=normalized_signal, sr=sr, n_mfcc=39)
    mel = librosa.feature.melspectrogram(y=normalized_signal, sr=sr)
    mel = librosa.amplitude_to_db(mel, ref=np.max)
    stft = np.abs(librosa.stft(normalized_signal))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(normalized_signal)
    rms = librosa.feature.rms(y=normalized_signal)

    mfcc_features = np.mean(mfcc, axis=1)
    mel_features = np.mean(mel, axis=1)
    chroma_features = np.mean(chroma, axis=1)
    zcr_feature = np.mean(zcr)
    rms_feature = np.mean(rms)

    features = np.concatenate((chroma_features, mfcc_features, mel_features, [zcr_feature, rms_feature], clinical_data))

    return features

@app.route('/predict_covid', methods=['POST'])
def predict_covid():
    try:
        # Check if the request contains the required keys
        if 'berkas_audio' not in request.files or \
           any(key not in request.form for key in ['kesulitan_bernapas', 'pilek', 'batuk', 'demam', 'anosmia', 'nyeri_otot', 'sakit_tenggorokan', 'diare', 'kelelahan']):
            return jsonify({"error": "Request tidak valid! Pastikan berkas suara batuk dan 9 data gejala telah diinputkan!"}), 400

        # Get the WAV file and boolean values from the request
        audio_file = request.files.get('berkas_audio')
        boolean_values = [request.form[key] == 'true' for key in ['kesulitan_bernapas', 'pilek', 'batuk', 'demam', 'anosmia', 'nyeri_otot', 'sakit_tenggorokan', 'diare', 'kelelahan']]

        if audio_file:
            tmp = io.BytesIO(audio_file.read())
            x, sr = sf.read(tmp)
            x = librosa.resample(x, orig_sr=sr, target_sr=48000)
            sr = 48000
            cough_segments, cough_mask = segment_cough(x, sr, 0.2, 0.2)

            dataset = []
            segments_base64 = []

            if len(cough_segments) == 0:
                return jsonify({"error": "Tidak ada segmen batuk terdeteksi!"}), 400

            for i, segment in enumerate(cough_segments):
                feature = extract_feature(segment, sr, boolean_values)
                dataset.append(feature)
                
                newsegment = []
                
                # Ensure the audio is PCM 16-bit
                if segment.dtype != np.int16:
                    newsegment = (segment * 32767).astype(np.int16)


                wav_filename = "temp_audio.wav"
                wav.write(wav_filename, sr, newsegment)

                with open(wav_filename, 'rb') as wav_file:
                    wav_data = wav_file.read()

                base64_wav_data = base64.b64encode(wav_data).decode('utf-8')
                segments_base64.append(base64_wav_data)

            test_pred = model.predict(dataset)
            y_pred = np.where(test_pred > 0.5, 1, 0)

            return jsonify({
                "wav_base64": segments_base64,
                "prediction": y_pred.tolist(),
                "probability": test_pred.tolist(),
                }), 200
        else:
            return jsonify({"error": "Berkas audio tidak valid!"}), 400

    except Exception as e:
        return jsonify({"error": f"Terjadi error: {str(e)}"}), 500
