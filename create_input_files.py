from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       json_path='dataset_coco.json',
                       image_folder='./',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='data',
                       max_len=50)
