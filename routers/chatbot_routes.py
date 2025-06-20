# routers/chatbot_routes.py
from flask import Blueprint, request, jsonify
from utils.causal_chatbot import get_chatbot_response # Import the core chatbot logic

chatbot_bp = Blueprint('chatbot_bp', __name__)

@chatbot_bp.route('/message', methods=['POST'])
def handle_chat_message():
    """
    API endpoint for the chatbot to receive user messages and provide responses.
    """
    data = request.json
    user_message = data.get('user_message')
    # Session context includes processed_data, causal_graph_adj, etc.
    session_context = data.get('session_context', {})

    if not user_message:
        return jsonify({"detail": "No user message provided."}), 400

    try:
        response_text = get_chatbot_response(user_message, session_context)
        return jsonify({"response": response_text}), 200
    except Exception as e:
        print(f"Error in chatbot route: {e}")
        return jsonify({"detail": f"An error occurred in the chatbot: {str(e)}"}), 500