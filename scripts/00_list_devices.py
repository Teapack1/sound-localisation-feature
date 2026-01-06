#!/usr/bin/env python3
import sounddevice as sd


def main():
    print(sd.query_devices())


if __name__ == "__main__":
    main()
