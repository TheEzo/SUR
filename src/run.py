from image_classifier import ImageClassifier
from dataset import DatasetLoader
import numpy as np
import subprocess
import os
import argparse

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_all", dest="run_all", action="store_true", help="Runs all models and saves their predictions to selected files.")
    arg_parser.add_argument("--train", dest="train", action="store_true", help="Train the model before evaluation, otherwise just load previously trained snapshot.")
    arg_parser.add_argument("--image_classifier_snapshot", dest="image_classifier_snapshot", default="snapshots/image_classifier.h5", help="Choose image classifier snapshot.")
    arg_parser.add_argument("--image_classifier", dest="image_classifier", action="store_true", help="Runs image classifier only and saves predictions to selected file.")
    arg_parser.add_argument("--image_classifier_path", dest="image_classifier_file", default="../image_classifier.txt")
    arg_parser.add_argument("--train_path", dest="train_path", default="datasets/train", help="Select folder with train data.")    
    arg_parser.add_argument("--val_path", dest="val_path", default="datasets/dev", help="Select folder val data.")
    arg_parser.add_argument("--test_path", dest="test_path", default="datasets/eval", help="Select folder with test data.")
    return arg_parser.parse_args()

def setup_gpu():
    # Find out whether CUDA-capable GPU is available and if it is, allow Tensorflow to use it
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    if len(freeGpu) == 0:
        print("No free GPU available, running in CPU-only mode!")
    else:
        print("Found GPU: " + str(freeGpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()

if __name__ == "__main__":
    args = parse_args()
    setup_gpu()
    image_dataset_loader = DatasetLoader(dataset_type="images", train_path=args.train_path, val_path=args.val_path, test_path=args.test_path)

    if args.run_all or args.image_classifier:
        model = ImageClassifier(dataset=image_dataset_loader)
        model.build_model()
        if args.train:
            model.train()
        else:
            model.load_weights(path=args.image_classifier_snapshot)
        
        eval = model.evaluate()
        # TODO: Write to file - np.savetxt()


