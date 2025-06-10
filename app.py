import os
import uuid
import base64
import logging
import tempfile
import time
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import psutil

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure session cookies for production
if os.environ.get('RENDER'):
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Configure logging
logging.basicConfig(level=logging.INFO if os.environ.get('RENDER') else logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
sessions = {}
SESSION_FILE = '/tmp/sessions.json'

def load_sessions():
    """Load sessions from file."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r') as f:
                global sessions
                sessions = json.load(f)
                logger.info("Sessions loaded from file")
    except Exception as e:
        logger.error(f"Error loading sessions: {str(e)}")

def save_sessions():
    """Save sessions to file."""
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(sessions, f)
        logger.info("Sessions saved to file")
    except Exception as e:
        logger.error(f"Error saving sessions: {str(e)}")

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    cors_credentials=True,
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=120,
    ping_interval=30
)
logger.info("SocketIO initialized successfully")

# Initialize Gemini model
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    llm = None
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        convert_system_message_to_human=True,
        temperature=0.7
    )
    logger.info("Gemini model initialized successfully")

def process_frame(frame_data, scale_factor=0.2):
    """Process a base64-encoded frame, resize it, and return the base64 string."""
    try:
        # Skip processing if memory is low
        if psutil.virtual_memory().percent > 90:
            logger.warning("Memory usage high, skipping frame processing")
            return None

        if ',' in frame_data:
            img_data = base64.b64decode(frame_data.split(",")[1])
        else:
            img_data = base64.b64decode(frame_data)

        img = Image.open(BytesIO(img_data))
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        quality = 30 if os.environ.get('RENDER') else 50
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info(f"Frame processed successfully, size: {len(img_str)} bytes")
        return img_str
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return None

def invoke_llm_for_description(frame_base64, mode):
    """Invoke LLM to describe the frame and return the response."""
    try:
        system_prompt = SystemMessage(
            content="You are a helpful AI assistant analyzing images. Provide a brief description (1-2 sentences) of what you see in the image, then ask 'How can I help you?'."
        )
        user_prompt = HumanMessage(content=[
            {"type": "text", "text": f"Describe what you see in this {'screen' if mode == 'desktop' else 'camera'} image briefly."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame_base64}"}
        ])
        response = llm.invoke([system_prompt, user_prompt])
        response_text = response.content if hasattr(response, 'content') else str(response)
        return response_text
    except Exception as e:
        logger.error(f"LLM description error: {str(e)}")
        return f"Unable to describe the image. How can I help you?"

@app.route('/')
def index():
    """Render the index page."""
    load_sessions()
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    if session_id not in sessions:
        sessions[session_id] = {'mode': None, 'responses': [], 'frame_path': None, 'session_initialized': False}
        save_sessions()
    logger.info(f"Session {session_id} initialized")
    return render_template('index.html')

@app.route('/chat/<mode>')
def chat(mode):
    """Render the chat page for the specified mode."""
    load_sessions()
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found in /chat for session_id: {session_id}")
        return redirect(url_for('index'))
    return render_template('chat.html', mode=mode, responses=sessions[session_id]['responses'])

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start the stream for the specified mode."""
    load_sessions()
    session_id = session.get('session_id')
    logger.info(f"Starting stream for session_id: {session_id}")

    if not session_id or session_id not in sessions:
        logger.error(f"Session not found in /start_stream for session_id: {session_id}")
        return jsonify({'error': 'Session not found'}), 400

    data = request.get_json()
    mode = data.get('mode')
    if mode not in ['desktop', 'camera']:
        logger.error(f"Invalid mode: {mode}")
        return jsonify({'error': 'Invalid mode'}), 400

    sessions[session_id]['responses'] = []
    sessions[session_id]['mode'] = mode
    sessions[session_id]['frame_path'] = None
    sessions[session_id]['session_initialized'] = False
    save_sessions()
    logger.info(f"Stream started for mode: {mode}, session_id: {session_id}")
    return jsonify({'status': 'Stream started', 'redirect': url_for('chat', mode=mode)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the stream and clean up resources."""
    load_sessions()
    session_id = session.get('session_id')
    data = request.get_json()
    mode = data.get('mode')
    logger.info(f"Stop stream requested for mode: {mode}, session_id: {session_id}")

    if session_id and session_id in sessions and sessions[session_id]['mode'] == mode:
        if 'frame_path' in sessions[session_id] and sessions[session_id]['frame_path'] and os.path.exists(sessions[session_id]['frame_path']):
            os.remove(sessions[session_id]['frame_path'])
            logger.info(f"Deleted frame file: {sessions[session_id]['frame_path']}")
        sessions[session_id]['mode'] = None
        sessions[session_id]['responses'] = []
        sessions[session_id]['frame_path'] = None
        sessions[session_id]['session_initialized'] = False
        save_sessions()
        logger.info(f"Session {session_id} reset")
    else:
        logger.info(f"No active stream found for session_id: {session_id}, mode: {mode}")
    return jsonify({'redirect': url_for('index')})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    load_sessions()
    session_id = session.get('session_id')
    logger.info(f"Client connected with session_id: {session_id}")
    emit('connected', {'status': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    session_id = session.get('session_id')
    logger.info(f"Client disconnected with session_id: {session_id}")

@socketio.on('screen_frame')
def handle_screen_frame(frame_data):
    """Handle screen frame updates from the client and save to disk."""
    try:
        load_sessions()
        session_id = session.get('session_id')
        logger.info(f"Received screen_frame for session_id: {session_id}")

        if session_id and session_id in sessions and sessions[session_id]['mode'] == 'desktop':
            processed_frame = process_frame(frame_data)
            if processed_frame:
                # Delete old frame if exists
                if 'frame_path' in sessions[session_id] and sessions[session_id]['frame_path'] and os.path.exists(sessions[session_id]['frame_path']):
                    os.remove(sessions[session_id]['frame_path'])
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='/tmp') as tmp_file:
                    tmp_file.write(base64.b64decode(processed_frame))
                    sessions[session_id]['frame_path'] = tmp_file.name
                logger.info(f"Desktop screen frame saved to {tmp_file.name}")

                # Send initial description if session not initialized
                if not sessions[session_id]['session_initialized'] and llm:
                    frame_base64 = processed_frame  # Already decoded
                    response_text = invoke_llm_for_description(frame_base64, 'desktop')
                    sessions[session_id]['responses'].append({'prompt': '', 'response': response_text})
                    sessions[session_id]['session_initialized'] = True
                    save_sessions()
                    emit('initial_response', {'response': response_text})

                save_sessions()
                emit('frame_processed', {'status': 'success'})
            else:
                emit('frame_processed', {'status': 'error', 'message': 'Failed to process frame'})
        else:
            emit('frame_processed', {'status': 'error', 'message': 'Invalid session or mode'})
    except Exception as e:
        logger.error(f"Error in handle_screen_frame: {str(e)}")
        emit('frame_processed', {'status': 'error', 'message': str(e)})

@socketio.on('camera_frame')
def handle_camera_frame(frame_data):
    """Handle camera frame updates from the client and save to disk."""
    try:
        load_sessions()
        session_id = session.get('session_id')
        logger.info(f"Received camera_frame for session_id: {session_id}")

        if session_id and session_id in sessions and sessions[session_id]['mode'] == 'camera':
            processed_frame = process_frame(frame_data)
            if processed_frame:
                # Delete old frame if exists
                if 'frame_path' in sessions[session_id] and sessions[session_id]['frame_path'] and os.path.exists(sessions[session_id]['frame_path']):
                    os.remove(sessions[session_id]['frame_path'])
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='/tmp') as tmp_file:
                    tmp_file.write(base64.b64decode(processed_frame))
                    sessions[session_id]['frame_path'] = tmp_file.name
                logger.info(f"Camera frame saved to {tmp_file.name}")

                # Send initial description if session not initialized
                if not sessions[session_id]['session_initialized'] and llm:
                    frame_base64 = processed_frame  # Already decoded
                    response_text = invoke_llm_for_description(frame_base64, 'camera')
                    sessions[session_id]['responses'].append({'prompt': '', 'response': response_text})
                    sessions[session_id]['session_initialized'] = True
                    save_sessions()
                    emit('initial_response', {'response': response_text})

                save_sessions()
                emit('frame_processed', {'status': 'success'})
            else:
                emit('frame_processed', {'status': 'error', 'message': 'Failed to process frame'})
        else:
            emit('frame_processed', {'status': 'error', 'message': 'Invalid session or mode'})
    except Exception as e:
        logger.error(f"Error in handle_camera_frame: {str(e)}")
        emit('frame_processed', {'status': 'error', 'message': str(e)})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio input and generate a response using the frame from disk."""
    logger.info("Received /process_audio request")
    load_sessions()
    session_id = session.get('session_id')

    if not session_id or session_id not in sessions:
        logger.error(f"Session not found for session_id: {session_id}")
        return jsonify({'error': 'Session not found'}), 400

    data = request.get_json()
    prompt = data.get('prompt')
    mode = data.get('mode')

    if not prompt:
        logger.warning("No prompt provided in the request.")
        return jsonify({'error': 'No prompt provided'}), 400

    if not llm:
        logger.error("LLM not initialized")
        return jsonify({'error': 'AI service not available'}), 500

    if sessions[session_id]['mode'] != mode:
        logger.error(f"Stream not initialized for mode: {mode}")
        return jsonify({'error': 'Stream not initialized'}), 400

    # Wait for frame if not available
    frame_path = sessions[session_id].get('frame_path')
    retries = 0
    max_retries = 5
    while (not frame_path or not os.path.exists(frame_path)) and retries < max_retries:
        logger.warning(f"No frame available for mode: {mode}, retrying ({retries + 1}/{max_retries})")
        time.sleep(1)
        retries += 1
        frame_path = sessions[session_id].get('frame_path')

    if not frame_path or not os.path.exists(frame_path):
        logger.error(f"No frame available for mode: {mode} after retries")
        return jsonify({'error': 'No frame available. Please ensure screen/camera sharing is active.'}), 429

    # Check if session is initialized
    if not sessions[session_id]['session_initialized']:
        logger.info("Session not initialized, ignoring prompt")
        return jsonify({'response': ''})  # Silent response until initialized

    try:
        with open(frame_path, 'rb') as f:
            frame_data = f.read()
            frame_base64 = base64.b64encode(frame_data).decode()

        system_prompt = SystemMessage(
            content="You are a helpful AI assistant analyzing images and responding to user prompts. Keep responses concise and helpful."
        )
        user_prompt = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame_base64}"}
        ])

        logger.info(f"Invoking LLM with prompt: {prompt}")
        response = llm.invoke([system_prompt, user_prompt])
        response_text = response.content if hasattr(response, 'content') else str(response)

        sessions[session_id]['responses'].append({'prompt': prompt, 'response': response_text})
        save_sessions()
        logger.info(f"LLM response generated successfully")
        return jsonify({'response': response_text})

    except Exception as e:
        logger.error(f"Assistant error in process_audio: {str(e)}")
        return jsonify({'error': f"Failed to process request: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'socketio_available': True,
        'langchain_available': True,
        'llm_initialized': llm is not None,
        'google_api_key_set': bool(os.getenv("GOOGLE_API_KEY")),
        'memory_usage_percent': psutil.virtual_memory().percent
    }), 200

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    # Use port from environment (default to 10000 if not set)
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting application on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=not os.environ.get('RENDER'), allow_unsafe_werkzeug=True)
