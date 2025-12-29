host = '0.0.0.0'
port = 5092
threads = 4
CHUNK_MINITE=10

import sys
sys.stdout = sys.stderr

import os,sys,json,math,re,threading
import shutil
import uuid
import subprocess
import datetime
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
from pathlib import Path
ROOT_DIR=Path(os.getcwd()).as_posix()
os.environ['HF_HOME'] = ROOT_DIR + "/models"
os.environ['HF_HUB_CACHE'] = ROOT_DIR + "/models"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
if sys.platform == 'win32':
    os.environ['PATH'] = ROOT_DIR + f';{ROOT_DIR}/ffmpeg;' + os.environ['PATH']


try:
    print("\nLoading Parakeet TDT 0.6B V3 ONNX model with INT8 quantization...")
    import onnx_asr
    asr_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8").with_timestamps()
    print("Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()

print("="*50)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  

def get_audio_duration(file_path: str) -> float:
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Could not get duration of file '{file_path}': {e}")
        return 0.0

def format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    s = str(delta)
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3]
    else:
        integer_part = s
        fractional_part = "000"

    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['segment'].strip()
        
        if text:
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
            
    return "\n".join(srt_content)

def segments_to_vtt(segments: list) -> str:
    vtt_content = ["WEBVTT", ""]
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start']).replace(',', '.')
        end_time = format_srt_time(segment['end']).replace(',', '.')
        text = segment['segment'].strip()
        
        if text:
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    return "\n".join(vtt_content)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # OpenAI compatible parameters
    model_name = request.form.get('model', 'model') # Ignored but accepted
    response_format = request.form.get('response_format', 'json')
    
    # Legacy support
    if model_name == 'parakeet_srt_words':
         pass # Handled below optionally

    original_filename = secure_filename(file.filename)

    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
    
    temp_files_to_clean = []

    try:
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)
        
        print(f"[{unique_id}] Converting '{original_filename}' to standard WAV format...")
        ffmpeg_command = [
            'ffmpeg', '-nostdin', '-y', '-i', temp_original_path,
            '-ac', '1', '-ar', '16000', target_wav_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return jsonify({"error": "File conversion failed", "details": result.stderr}), 500
        temp_files_to_clean.append(target_wav_path)

        CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60  
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400
        
        num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
        chunk_paths = []
        print(f"[{unique_id}] Total duration: {total_duration:.2f}s. Splitting into {num_chunks} chunks.")
        
        if num_chunks>1:
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION_SECONDS
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                print(f"[{unique_id}] Creating chunk {i+1}/{num_chunks}...")
                chunk_command = [
                    'ffmpeg', '-nostdin', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(CHUNK_DURATION_SECONDS),
                    '-c', 'copy',
                    chunk_path
                ]
                subprocess.run(chunk_command, capture_output=True, text=True)
        else:
            chunk_paths.append(target_wav_path)
            
        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0

        def clean_text(text):
            """Clean up spacing artifacts from token joining"""
            if not text: return ""
            # Handle potential SentencePiece underline
            text = text.replace('\u2581', ' ')
            text = text.strip()
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text)
            # Standard cleaning
            text = text.replace(" '", "'")
            return text

        for i, chunk_path in enumerate(chunk_paths):
            print(f"[{unique_id}] Transcribing chunk {i+1}/{num_chunks}...")
            
            result = asr_model.recognize(chunk_path)
            
            if result and result.text:
                start_time = result.timestamps[0] if result.timestamps else 0
                end_time = result.timestamps[-1] if len(result.timestamps) > 1 else start_time + 0.1
                
                cleaned_text = clean_text(result.text)
                
                segment = {
                    'start': start_time + cumulative_time_offset,
                    'end': end_time + cumulative_time_offset,
                    'segment': cleaned_text
                }
                all_segments.append(segment)
                
                for j, (token, timestamp) in enumerate(zip(result.tokens, result.timestamps)):
                    if j < len(result.timestamps) - 1:
                        word_end = result.timestamps[j + 1]
                    else:
                        word_end = end_time
                    
                    # Clean tokens too
                    clean_token = token.replace('\u2581', ' ').strip()
                    word = {
                        'start': timestamp + cumulative_time_offset,
                        'end': word_end + cumulative_time_offset,
                        'word': clean_token
                    }
                    all_words.append(word)

            chunk_actual_duration = get_audio_duration(chunk_path)
            cumulative_time_offset += chunk_actual_duration

        print(f"[{unique_id}] All chunks transcribed, merging results.")

        if not all_segments:
            # Return empty structure if nothing found, consistent with failures or silence? 
            # OpenAI sometimes returns empty json text.
            pass

        # Formatting Output
        full_text = " ".join([seg['segment'] for seg in all_segments])

        if response_format == 'srt' or model_name == 'parakeet_srt_words':
            srt_output = segments_to_srt(all_segments)
            if model_name == 'parakeet_srt_words':
                 json_str_list = [{"start": it['start'], "end": it['end'], "word":it['word']} for it in all_words]
                 srt_output += "----..----" + json.dumps(json_str_list)
            return Response(srt_output, mimetype='text/plain')
        
        elif response_format == 'vtt':
            return Response(segments_to_vtt(all_segments), mimetype='text/plain')
        
        elif response_format == 'text':
            return Response(full_text, mimetype='text/plain')
            
        elif response_format == 'verbose_json':
             # Minimal verbose_json structure
             return jsonify({
                 "task": "transcribe",
                 "language": "english", # detection not implemented here, hardcoded or param?
                 "duration": total_duration,
                 "text": full_text,
                 "segments": [{
                     "id": idx,
                     "seek": 0,
                     "start": seg['start'],
                     "end": seg['end'],
                     "text": seg['segment'],
                     "tokens": [], # Populate if needed
                     "temperature": 0.0,
                     "avg_logprob": 0.0,
                     "compression_ratio": 0.0,
                     "no_speech_prob": 0.0
                 } for idx, seg in enumerate(all_segments)]
             })
             
        else:
            # Default JSON
            return jsonify({"text": full_text})

    except Exception as e:
        print(f"A serious error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        print(f"[{unique_id}] Cleaning up temporary files...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] Temporary files cleaned.")

def openweb():
    import webbrowser,time
    time.sleep(5)
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}')

if __name__ == '__main__':
    print(f"Starting server...")
    print(f"Web interface: http://127.0.0.1:{port}")
    print(f"API Endpoint: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"Running with {threads} threads.")
    print(f"Starting web browser thread...")
    threading.Thread(target=openweb).start()
    print(f"Starting waitress server...")
    serve(app, host=host, port=port, threads=threads)
    print(f"Server started!")
