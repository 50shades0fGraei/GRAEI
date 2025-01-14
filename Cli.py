
import argparse

def main():
    parser = argparse.ArgumentParser(description='My AI CLI')
    parser.add_argument('--input', help='Input text for the AI model')
    args = parser.parse_args()

    # Call your AI model here with the input text
    print(f"Input: {args.input}")

if __name__ == '__main__':
    main()
 
