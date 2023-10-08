import sys
import dionysus


def main():
    print(f"Welcome to dionysus, version {dionysus.__version__}")

    if len(sys.argv) > 1:
        print(sys.argv[1])
    else:
        print("no arg given")


if __name__ == "__main__":
    main()
