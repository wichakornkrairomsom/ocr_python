from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
from google.cloud import vision
from google.cloud.vision_v1 import types
import ast
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Construct the full path to the credentials file
credentials_filename = os.getenv('GOOGLE_CREDENTIALS_FILE')
project_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(project_dir, 'credentials', credentials_filename)

# Set up Google Cloud Vision client
client = vision.ImageAnnotatorClient.from_service_account_file(credentials_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
def crop_image(image, top_left, bottom_right):
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    return image[y1:y2, x1:x2]

def mark_positions(image, positions):
    marked_image = image.copy()
    for pos in positions:
        cv2.circle(marked_image, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)  # Red dot at position
    return marked_image

@app.route('/detect-position', methods=['POST'])
def detect_position():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    fields = ['car_regis', 'body_number', 'thai_ID', 'name', 'book_number','engine_number','car_type','brand']
    results = {}
    for field in fields:
        if field in request.form:
            coords_str = request.form[field]
            #print(coords_str)
            try:
                coords_list = ast.literal_eval(coords_str)
                
                if isinstance(coords_list[0], (int, float)):
                    coords_list = [coords_list]
                
                top_left_coords = [(float(coords_list[i][0]), float(coords_list[i][1])) for i in range(0, len(coords_list), 2)]
                bottom_right_coords = [(float(coords_list[i+1][0]), float(coords_list[i+1][1])) for i in range(0, len(coords_list), 2)]

                texts = []
                positions = []
                for i in range(len(top_left_coords)):
                    top_left = top_left_coords[i]
                    bottom_right = bottom_right_coords[i]
                    cropped_image = crop_image(image, top_left, bottom_right)
                    
                    if cropped_image.size == 0:
                        continue

                    success, encoded_image = cv2.imencode('.jpg', cropped_image)
                    if not success:
                        continue

                    content = encoded_image.tobytes()
                    img = types.Image(content=content)

                    # Perform text detection on the cropped image
                    response = client.text_detection(image=img)
                    texts_in_box = response.text_annotations
                    if texts_in_box:
                        texts.append(texts_in_box[0].description)
                        # Calculate position for marking
                        x_pos = (top_left[0] + bottom_right[0]) / 2
                        y_pos = (top_left[1] + bottom_right[1]) / 2
                        positions.append((x_pos, y_pos))

                if texts:
                    results[field] = texts[0]

                # Mark positions on the original image
                if positions:
                    image = mark_positions(image, positions)

            except (ValueError, SyntaxError) as e:
                return jsonify({"error": f"Invalid coordinate format for {field}: {str(e)}"}), 400

    # Save the marked image
    marked_image_filename = 'marked_image.jpg'
    marked_image_path = os.path.join(current_dir, marked_image_filename)
    cv2.imwrite(marked_image_path, image)

    # Return the path to the marked image and the detected texts
    return jsonify({"status": 200, "message": "detect success", "texts": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8088)
