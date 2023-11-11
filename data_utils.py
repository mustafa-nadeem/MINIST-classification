import gzip
import struct

def load_images(path: str) :
    with gzip.open(path, "rb") as file:
        header = struct.unpack('>4i', file.read(16))
        magic, size, width, height = header
        print(header)


def main(file):
    load_images(file)

if __name__ == "__main__":
    main("data/test-images.gz")
