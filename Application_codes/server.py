import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt

def send_frame(client_socket, frame):
    data = cv2.imencode('.jpg', frame)[1].tobytes()
    client_socket.sendall(len(data).to_bytes(4, 'big') + data)

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

def main():
    host = '10.42.0.1'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Listening on {host}:{port}")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    video_path = "input_video.mp4"  # Path to your video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        send_frame(client_socket, frame)
        processed_frame = receive_frame(client_socket)

        if processed_frame is not None:
            plt.subplot(1, 2, 1)
            plt.title("Original Frame")
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
            
            # Overlay the processed frame on the original frame
            overlay = cv2.addWeighted(frame, 0.5, processed_frame, 0.8, 0)            

            plt.subplot(1, 2, 2)
            plt.title("Processed Frame")
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            plt.pause(0.001)
        else:
            print("Client disconnected.")
            break

    cap.release()
    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    main()
