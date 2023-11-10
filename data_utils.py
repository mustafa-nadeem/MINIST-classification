import gzip

def open_gzip(path: str) :
    with gzip.open(path) as f:
        file_content = f.read()
        print(file_content.decode("utf-8", "replace"))


def main(file):
    open_gzip(file)

if __name__ == "__main__":
    main("data/test-images.gz")
