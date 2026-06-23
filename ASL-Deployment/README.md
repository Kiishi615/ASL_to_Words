

# ASL Sign Language Detector

Real-time American Sign Language (ASL) letter recognition using your webcam.

## How it Works

1. **Browser**: MediaPipe Hands runs in your browser to detect and track hand landmarks from your webcam feed.
2. **Server**: Normalized landmarks (42 values) are sent to the Flask API, which uses a trained scikit-learn model to predict the ASL letter.
3. **Word Building**: Hold a sign steady for 3 seconds to confirm and add it to the word display.

Recognizes all 26 letters (A–Z), plus `space` and `del` gestures.
