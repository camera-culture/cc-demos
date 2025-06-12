import cv2

def main():
    cap = cv2.VideoCapture(0)  # Use the correct index for your USB camera

    # Set exposure to a low value (may need adjustment or may be ignored if auto mode is on)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -13)         # Lower = darker, range is camera-dependent

    def loop():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Live USB Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    try:
        loop()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()