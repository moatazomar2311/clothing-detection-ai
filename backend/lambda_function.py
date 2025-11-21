import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from ai_edge_litert.interpreter import Interpreter

# Global variables (initialized once, reused across invocations)
interpreter = None
class_names = [
    'dress', 'hat', 'longsleeve', 'outwear', 'pants',
    'shirt', 'shoes', 'shorts', 'skirt', 't-shirt'
]

def initialize_model():
    """Initialize the TFLite model (called once on cold start)"""
    global interpreter
    if interpreter is None:
        print("🔄 Initializing model...")
        interpreter = Interpreter(model_path='new_model.tflite')
        interpreter.allocate_tensors()
        print("✅ Model ready!")

def mobilenetv2_preprocess(img_array):
    """Preprocess image for MobileNetV2"""
    img_array = img_array.astype(np.float32)
    img_array = img_array / 127.5 - 1.0
    return img_array

def load_and_preprocess_image(image_data, target_size):
    """Load and preprocess image from bytes"""
    img = Image.open(BytesIO(image_data))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # pyrefly: ignore [missing-attribute]
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenetv2_preprocess(img_array)
    
    return img_array

def lambda_handler(event, context):
    """
    Lambda function handler for clothing classification
    CORS is handled by Lambda Function URL configuration
    """
    try:
        # Initialize model on first invocation
        initialize_model()
        
        # Get input details
        # pyrefly: ignore [missing-attribute]
        input_details = interpreter.get_input_details()
        # pyrefly: ignore [missing-attribute]
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
        target_size = (IMG_WIDTH, IMG_HEIGHT)
        
        # Parse the incoming image data
        image_base64 = None
        
        if 'body' in event and event['body']:
            body = event['body']
            
            # If body is a JSON string, parse it
            if isinstance(body, str):
                try:
                    body_json = json.loads(body)
                    image_base64 = body_json.get('image', body_json.get('data', ''))
                except json.JSONDecodeError:
                    image_base64 = body
            else:
                image_base64 = body.get('image', body.get('data', ''))
        
        if not image_base64:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'No image data provided',
                    'hint': 'Send POST request with JSON body: {"image": "base64_string"}'
                })
            }
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',', 1)[1]
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_base64)
            print(f"Successfully decoded image, size: {len(image_data)} bytes")
        except Exception as e:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': f'Invalid base64 image data: {str(e)}'})
            }
        
        # Preprocess image
        processed_image = load_and_preprocess_image(image_data, target_size)
        
        # Run inference
        # pyrefly: ignore [missing-attribute]
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        # pyrefly: ignore [missing-attribute]
        interpreter.invoke()
        # pyrefly: ignore [missing-attribute]
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predictions
        probabilities = output_data[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Prepare all probabilities
        all_predictions = {
            name: float(prob) 
            for name, prob in zip(class_names, probabilities)
        }
        
        # Sort by confidence
        sorted_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )
        
        print(f"✅ Prediction: {predicted_class} ({confidence:.2%})")
        
        # Return result (CORS handled by Function URL config)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': sorted_predictions,
                'model_input_size': f"{IMG_HEIGHT}x{IMG_WIDTH}"
            })
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during inference',
                'type': type(e).__name__
            })
        }
