import argparse
import os
import subprocess

from dataset import DatasetLoader
from image_classifier import ImageClassifier
from voice_classifier import VoiceClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_all", dest="run_all", action="store_true",
                            help="Runs all models and saves their predictions to selected files.")
    arg_parser.add_argument("--load_model", dest="load_model", action="store_true",
                            help="Load the model before training.")
    arg_parser.add_argument("--train", dest="train", action="store_true",
                            help="Train the model, otherwise just load previously trained snapshot.")
    arg_parser.add_argument("--image_classifier_snapshot", dest="image_classifier_snapshot",
                            default="snapshots/image_classifier.h5", help="Choose image classifier snapshot.")
    arg_parser.add_argument("--image_classifier", dest="image_classifier", action="store_true",
                            help="Runs image classifier only and saves predictions to selected file.")
    arg_parser.add_argument("--image_classifier_file", dest="image_classifier_file", default="../image_classifier.txt")
    arg_parser.add_argument("--train_path", dest="train_path", default="datasets/train",
                            help="Select folder with train data.")
    arg_parser.add_argument("--val_path", dest="val_path", default="datasets/dev", help="Select folder val data.")
    arg_parser.add_argument("--test_path", dest="test_path", default="datasets/eval",
                            help="Select folder with test data.")
    arg_parser.add_argument('--voice_classifier', action='store_true', help='Runs voice classifier.')
    arg_parser.add_argument("--voice_classifier_file", dest="voice_classifier_file", default="../voice_classifier.txt")
    arg_parser.add_argument("--voice_classifier_snapshot", dest="voice_classifier_snapshot",
                            default="snapshots/voice_classifier.h5", help="Choose voice classifier snapshot.")
    return arg_parser.parse_args()


def setup_gpu():
    # Find out whether CUDA-capable GPU is available and if it is, allow Tensorflow to use it
    freeGpu = subprocess.check_output(
        'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"',
        shell=True)
    if len(freeGpu) == 0:
        print("No free GPU available, running in CPU-only mode!")
    else:
        print("Found GPU: " + str(freeGpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()


def write_eval(file_path, eval_list):
    with open(file_path, "w+") as f:
        for row in eval_list:
            for col in row:
                f.write(f"{col} ")
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    setup_gpu()

    if args.run_all or args.image_classifier:
        image_dataset_loader = DatasetLoader(dataset_type="images", train_path=args.train_path, val_path=args.val_path,
                                             test_path=args.test_path)
        model = ImageClassifier(dataset=image_dataset_loader)
        model.build_model()

        if args.train:
            if args.load_model:
                model.load_weights(path=args.image_classifier_snapshot)
            model.train(snapshot_path=args.image_classifier_snapshot)
        else:
            model.load_weights(path=args.image_classifier_snapshot)
            eval_list = model.evaluate()
            write_eval(args.image_classifier_file, eval_list)

    if args.run_all or args.voice_classifier:
        voice_dataset_loader = DatasetLoader(args.train_path, args.val_path, args.test_path, 'voice')

        model = VoiceClassifier(dataset=voice_dataset_loader)
        model.build_model()

        if args.train:
            model.train()
        else:
            eval_list = model.evaluate(args.voice_classifier_snapshot)
            write_eval(args.voice_classifier_file, eval_list)
