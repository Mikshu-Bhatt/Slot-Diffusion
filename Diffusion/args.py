import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--csv_path')
    # parser.add_argument('--part', type=int)
    # parser.add_argument('--output_dir')
    parser.add_argument('--vae_ckpt')
    parser.add_argument('--batch_size',type = int)
    args = parser.parse_args()

    return args

args = parse_args()

