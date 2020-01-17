import argparse
from gtzan import AppManager

# Constants
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

# @RUN: Main function to call the appmanager
def main(args):
    if args.type not in ["dl", "ml"]:
        raise ValueError("Invalid type for the application. You should use dl or ml.")

    app = AppManager(args, genres)
    app.run()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    # Required arguments
    parser.add_argument('-t', '--type', help='dl or ml for Deep Learning or Classical ML approaches, respectively.', type=str, required=True)

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-m', '--model', help='Path to trained model', type=str, required=True)
    parser.add_argument('-s', '--song', help='Path to song to classify', type=str, required=True)
    args = parser.parse_args()

    # Call the main function
    main(args)