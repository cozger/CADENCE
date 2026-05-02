from cadence.preprocess._cli import make_modality_cli
from cadence.preprocess.ecg.pipeline import preprocess_ecg_session

if __name__ == "__main__":
    make_modality_cli(preprocess_ecg_session, modality_name="ecg")
