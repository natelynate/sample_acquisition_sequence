import os

def main(base):
    try:
        face_frame_dir = os.path.join(base, 'data', 'face_image') 
        face_lm_dir = os.path.join(base, 'data', 'landmark_coordinates') 
        face_lm_img_dir = os.path.join(base, 'data', 'face_binary')
        lefteye_frame_dir = os.path.join(base, 'data', 'eye_image_left') 
        righteye_frame_dir = os.path.join(base, 'data', 'eye_image_right') 
        gazepoint_dir = os.path.join(base, 'data', 'gazepoint')
        
        # Confirm directory existence, else create new 
        os.makedirs(face_frame_dir, exist_ok=True)
        os.makedirs(face_lm_img_dir, exist_ok=True)
        os.makedirs(face_lm_dir, exist_ok=True)
        os.makedirs(lefteye_frame_dir, exist_ok=True)
        os.makedirs(righteye_frame_dir, exist_ok=True)
        os.makedirs(gazepoint_dir, exist_ok=True)

        print(f"Data directories are created in {base}")
    except:
        print("Error while creating save directories. Confirm the project directory")
        
if __name__ == "__main__":
    base = os.getcwd()
    main(base)
