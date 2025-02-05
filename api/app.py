from flask import Flask, jsonify, request
import gradio as gr
import threading
from models.huggingface_model import generate_service_description, service_types, classify_service_type

app = Flask(__name__)


# Flask API Endpoint
@app.route('/generate-description', methods=['POST'])
def generate_description():
    """Flask API endpoint for generating service descriptions."""
    data = request.get_json()
    service_name = data.get('service_name', '').strip()

    if not service_name:
        return jsonify({"error": "Service name is required"}), 400

    # Generate service description
    description = generate_service_description(service_name)

    # Define the response structure
    response = {
        "service_name": service_name,
        "description": {
            "description": description,  # Generated description
        }
    }

    return jsonify(response)


# Gradio Function
def gradio_generate_description(service_name):
    # Generate description using the Hugging Face model
    description = generate_service_description(service_name)

    # Assuming the service is a plumber, you can adjust service type dynamically.
    service_type = classify_service_type(service_name)  # Placeholder: adjust based on user input or model logic
    return service_type, description


# Start Flask in a separate thread
def run_flask():
    app.run( port=5001, debug=False, use_reloader=False)


# Start Flask Thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Launch Gradio Interface
gr.Interface(
    fn=gradio_generate_description,
    inputs=gr.Textbox(label="Enter Service Name (e.g., emergency pipe leak repair)", lines=1),
    outputs=[gr.Textbox(label="Service Type", lines=1), gr.Textbox(label="Generated Description", lines=10)],
    title="Service Description Generator",
    description="Enter a detailed service name (e.g., 'emergency pipe leak repair') to get a detailed description and service type.",
).launch( server_port=7860)


# # Combined Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# Service Description Generator")
#     gr.Markdown("Enter a detailed service name to get its type and description.")
#
#     with gr.Row():
#         input_text = gr.Textbox(label="Enter Service Name (e.g., emergency pipe leak repair)", lines=1)
#
#     with gr.Row():
#         output_type = gr.Textbox(label="Service Type", lines=1)
#         output_desc = gr.Textbox(label="Generated Description", lines=5)
#
#     submit_button = gr.Button("Generate")
#     submit_button.click(gradio_generate_description, inputs=input_text, outputs=[output_type, output_desc])
#
#     # Copyright Notice
#     gr.Markdown("**Copyright @faskath**")
#
# demo.launch(server_port=7860)  # Single Gradio instance running on port 7860