from cadence.preprocess._cli import make_modality_cli
from cadence.preprocess.face.pipeline import preprocess_face_session

if __name__ == "__main__":
    make_modality_cli(preprocess_face_session, modality_name="face")
