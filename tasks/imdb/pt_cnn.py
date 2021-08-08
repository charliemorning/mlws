import torch
from sklearn.datasets import load_files


def main():
    imdb = load_files(r"C:\Users\Charlie\Developer\aclImdb")
    imdb.data

if __name__ == '__main__':
    main()