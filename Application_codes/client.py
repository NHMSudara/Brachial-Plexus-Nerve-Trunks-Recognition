import socket
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pynq_dpu import DpuOverlay
import time

def preprocess_image(frame, target_size=(96, 96)):
    # Resize the image to the target size
    image = cv2.resize(frame, target_size)

    # Convert the image from BGR to RGB (if needed)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension to the image
    image = np.expand_dims(image, axis=0)

    return image

def load_and_execute_xmodel(dpu_overlay, xmodel_file, input_data):
    try:
        # Load the xmodel
        dpu_overlay.load_model(xmodel_file)
        print("xmodel loaded")
        
        # Get the DPU runner
        dpu = dpu_overlay.runner

        # Get input and output tensors
        input_tensors = dpu.get_input_tensors()
        output_tensors = dpu.get_output_tensors()

        # Prepare input data
        input_tensor = input_tensors[0]
        output_tensor = output_tensors[0]
        input_shape = tuple(input_tensor.dims)
        input_data = np.resize(input_data, input_shape).astype(np.float32)

        # Create buffer for output data
        output_shape = tuple(output_tensor.dims)
        output_data = np.empty(output_shape, dtype=np.float32)
        start_time = time.time()
        # Execute the model
        job_id = dpu.execute_async([input_data], [output_data])
        dpu.wait(job_id)
        end_time = time.time()
        print("DPU execution finished")
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return output_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_frame(frame, dpu_overlay, xmodel_file):
    input_data = preprocess_image(frame, target_size=(96, 96))
    output_data = load_and_execute_xmodel(dpu_overlay, xmodel_file, input_data)
    
    if output_data is not None:
        output_data = output_data.squeeze()
        output_data = (output_data > 0.002)
        output_data = output_data.astype(np.uint8) * 255
        output_data = np.expand_dims(output_data, axis=0)
        output_data = np.reshape(output_data, (96, 96, 1))

        cont, _ = cv2.findContours(output_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(input_data[0].shape, dtype=np.uint8)
        cv2.drawContours(mask, cont, -2, (255, 255, 0), 2)
        output_data = mask + input_data[0]

        return output_data
    else:
        return frame

def receive_frame(client_socket):
    data_len = int.from_bytes(client_socket.recv(4), 'big')
    data = b''
    while len(data) < data_len:
        packet = client_socket.recv(data_len - len(data))
        if not packet:
            return None
        data += packet
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame

def send_frame(client_socket, frame):
    data = cv2.imencode('.jpg', frame)[1].tobytes()
    client_socket.sendall(len(data).to_bytes(4, 'big') + data)

def main():
    host = '10.42.0.1'
    port = 12345

    xmodel_file = "compiled_model/kr260.xmodel"
    overlay = DpuOverlay("dpu.bit")
    print("DPU overlay loaded")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Connected to {host}:{port}")

    while True:
        frame = receive_frame(client_socket)
        if frame is None:
            break

        processed_frame = process_frame(frame, overlay, xmodel_file)        
     
        processed_frame=cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        send_frame(client_socket, processed_frame)

    client_socket.close()

if __name__ == "__main__":
    main()

