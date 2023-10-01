import sys
import dl


def main():

    print(f"Welcome to dl, version {dl.__version__}")

    if len(sys.argv) > 1:
        print(sys.argv[1])
    else:
        print("no arg given")


if __name__ == "__main__":
    main()
