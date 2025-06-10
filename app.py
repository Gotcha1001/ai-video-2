import os
import uuid
import base64
import logging
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session

try:
    from flask_socketio import SocketIO, emit

    SOCKETIO_AVAILABLE = True
except ImportError as e:
    print(f"SocketIO import error: {e}")
    SOCKETIO_AVAILABLE = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure logging for production
if os.environ.get('RENDER'):
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
sessions = {}

# Initialize SocketIO with production-ready settings
if SOCKETIO_AVAILABLE:
    try:
        if os.environ.get('RENDER'):
            # Production settings for Render
            socketio = SocketIO(
                app,
                cors_allowed_origins="*",
                cors_credentials=True,
                async_mode='threading',
                logger=False,
                engineio_logger=False,
                ping_timeout=60,
                ping_interval=25
            )
        else:
            # Development settings
            socketio = SocketIO(
                app,
                cors_allowed_origins="*",
                cors_credentials=True,
                async_mode='threading'
            )
        logger.info("SocketIO initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SocketIO: {e}")
        socketio = None
else:
    socketio = None

# Initialize Gemini model
if LANGCHAIN_AVAILABLE:
    try:
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
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        llm = None
else:
    llm = None


def process_frame(frame_data, scale_factor=0.3):
    """Process a base64-encoded frame, resize it more aggressively for production."""
    try:
        if ',' in frame_data:
            img_data = base64.b64decode(frame_data.split(",")[1])
        else:
            img_data = base64.b64decode(frame_data)

        img = Image.open(BytesIO(img_data))

        # More aggressive resizing for production to reduce memory usage
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Reduce quality further for production
        quality = 50 if os.environ.get('RENDER') else 70

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        logger.info(f"Frame processed successfully, size: {len(img_str)} bytes")
        return img_str
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return None


@app.route('/')
def index():
    """Render the index page."""
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    if session_id not in sessions:
        sessions[session_id] = {'mode': None, 'responses': [], 'frame': None}
    logger.info(f"Session {session_id} initialized")
    return render_template('index.html')


@app.route('/chat/<mode>')
def chat(mode):
    """Render the chat page for the specified mode."""
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found in /chat for session_id: {session_id}")
        return redirect(url_for('index'))
    return render_template('chat.html', mode=mode, responses=sessions[session_id]['responses'])


@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start the stream for the specified mode."""
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
    logger.info(f"Stream started for mode: {mode}, session_id: {session_id}")
    return jsonify({'status': 'Stream started', 'redirect': url_for('chat', mode=mode)})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the stream for the specified mode."""
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        logger.error(f"Session not found in /stop_stream for session_id: {session_id}")
        return jsonify({'error': 'Session not found'}), 400

    data = request.get_json()
    mode = data.get('mode')
    logger.info(f"Stop stream requested for mode: {mode}, session_id: {session_id}")

    if sessions[session_id]['mode'] == mode:
        sessions[session_id]['mode'] = None
        sessions[session_id]['responses'] = []
        sessions[session_id]['frame'] = None
        logger.info(f"Session {session_id} reset")
        return jsonify({'redirect': url_for('index')})
    else:
        logger.error(f"Invalid mode for stop_stream: {mode}")
        return jsonify({'error': 'Invalid mode'}), 400


if socketio:
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
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
        """Handle screen frame updates from the client."""
        try:
            session_id = session.get('session_id')
            logger.info(f"Received screen_frame for session_id: {session_id}")

            if session_id and session_id in sessions and sessions[session_id]['mode'] == 'desktop':
                processed_frame = process_frame(frame_data)
                if processed_frame:
                    sessions[session_id]['frame'] = processed_frame
                    logger.info(f"Desktop screen frame updated")
                    emit('frame_processed', {'status': 'success'})
                else:
                    emit('frame_processed', {'status': 'error', 'message': 'Failed to process frame'})
        except Exception as e:
            logger.error(f"Error in handle_screen_frame: {e}")
            emit('frame_processed', {'status': 'error', 'message': str(e)})


    @socketio.on('camera_frame')
    def handle_camera_frame(frame_data):
        """Handle camera frame updates from the client."""
        try:
            session_id = session.get('session_id')
            logger.info(f"Received camera_frame for session_id: {session_id}")

            if session_id and session_id in sessions and sessions[session_id]['mode'] == 'camera':
                processed_frame = process_frame(frame_data)
                if processed_frame:
                    sessions[session_id]['frame'] = processed_frame
                    logger.info(f"Camera frame updated")
                    emit('frame_processed', {'status': 'success'})
                else:
                    emit('frame_processed', {'status': 'error', 'message': 'Failed to process frame'})
        except Exception as e:
            logger.error(f"Error in handle_camera_frame: {e}")
            emit('frame_processed', {'status': 'error', 'message': str(e)})


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio input and generate a response."""
    logger.info("Received /process_audio request")
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

    try:
        frame = sessions[session_id]['frame']
        if not frame:
            logger.error(f"No frame available for mode: {mode}")
            return jsonify({'error': 'No frame available. Please ensure screen/camera sharing is active.'}), 500

        system_prompt = SystemMessage(
            content="You are a helpful AI assistant analyzing images and responding to user prompts. Keep responses concise and helpful."
        )
        user_prompt = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame}"}
        ])

        logger.info(f"Invoking LLM with prompt: {prompt}")
        response = llm.invoke([system_prompt, user_prompt])
        response_text = response.content if hasattr(response, 'content') else str(response)

        sessions[session_id]['responses'].append({'prompt': prompt, 'response': response_text})
        logger.info(f"LLM response generated successfully")
        return jsonify({'response': response_text})

    except Exception as e:
        logger.error(f"Assistant error in process_audio: {str(e)}", exc_info=True)
        return jsonify({'error': f"Failed to process request: {str(e)}"}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'socketio_available': SOCKETIO_AVAILABLE and socketio is not None,
        'langchain_available': LANGCHAIN_AVAILABLE,
        'llm_initialized': llm is not None,
        'google_api_key_set': bool(os.getenv("GOOGLE_API_KEY"))
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
    # Use port from environment (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting application on port {port}")

    # Check if required dependencies are available
    if not SOCKETIO_AVAILABLE:
        logger.error("SocketIO not available. Real-time features will not work.")
        app.run(debug=False, host='0.0.0.0', port=port)
    elif not llm:
        logger.error("LLM not initialized. AI features will not work.")
        if socketio:
            socketio.run(app, host='0.0.0.0', port=port, debug=False)
        else:
            app.run(debug=False, host='0.0.0.0', port=port)
    else:
        # Production mode
        if os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production':
            socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
        else:
            # Development mode
            socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
