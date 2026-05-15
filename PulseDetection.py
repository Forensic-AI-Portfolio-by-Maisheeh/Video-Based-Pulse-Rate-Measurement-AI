import cv2                          # OpenCV for webcam + face detection
import numpy as np                  # Numerical operations (arrays, FFT, etc.)
from scipy.signal import butter, filtfilt  # Signal filtering tools
import time                         # For timestamps
import matplotlib.pyplot as plt     # For live plotting

# -----------------------------
# Parameters
# -----------------------------
BUFFER_SIZE = 300   # Number of samples stored (~10 seconds at 30 FPS)
FPS = 30            # Assumed frame rate (used for frequency calculations)
LOW_HZ = 0.7        # Lower bound of heart rate (Hz) ~ 42 BPM
HIGH_HZ = 4.0       # Upper bound of heart rate (Hz) ~ 240 BPM
bpm_buffer = []

# -----------------------------
# Bandpass filter
# -----------------------------
def bandpass_filter(signal, fs):
    # Create a 3rd-order Butterworth bandpass filter
    b, a = butter(3, [LOW_HZ/(fs/2), HIGH_HZ/(fs/2)], btype='band')
    
    # Apply filter forward + backward (prevents phase distortion)
    return filtfilt(b, a, signal)

# -----------------------------
# FFT to BPM
# -----------------------------
def compute_bpm(signal, fs):
    # Need at least ~3 seconds of data for meaningful frequency estimation
    if len(signal) < fs * 3:
        return None

    # Filter signal to isolate pulse frequencies
    filtered = bandpass_filter(signal, fs)

    # Compute FFT (frequency domain representation)
    fft = np.fft.rfft(filtered)

    # Get frequency bins corresponding to FFT values
    freqs = np.fft.rfftfreq(len(filtered), d=1/fs)

    # Keep only valid heart rate range
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

    # Apply mask
    fft = np.abs(fft[mask])   # Magnitude spectrum
    freqs = freqs[mask]       # Corresponding frequencies

    # Find dominant frequency (highest peak)
    peak_freq = freqs[np.argmax(fft)]

    # Convert Hz → BPM
    return peak_freq * 60

# -----------------------------
# Initialize plotting (3 graphs)
# -----------------------------
plt.ion()  # Turn on interactive mode (real-time updates)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 14))
fig.subplots_adjust(hspace=0.7)

# Create empty line objects (we update these later)
line1, = ax1.plot([], [])
line2, = ax2.plot([], [])
line3, = ax3.plot([], [])
line4, = ax4.plot([], [])


# Titles for clarity
ax1.set_title("Raw Green Signal")
ax2.set_title("Filtered Signal")
ax3.set_title("FFT Spectrum")
ax4.set_title("Live BPM")

# -----------------------------
# Function to update plots
# -----------------------------
def update_plots(signal, fs, bpm_buffer):
    # Don't plot until we have enough data
    if len(signal) < fs * 3:
        return

    sig = np.array(signal)  # Convert list → numpy array

    # Apply bandpass filter
    filtered = bandpass_filter(sig, fs)

    # Compute FFT
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1/fs)

    # Update raw signal plot
    line1.set_data(range(len(sig)), sig)
    ax1.relim()             # Recompute axis limits
    ax1.autoscale_view()    # Autoscale axes

    # Update filtered signal plot
    line2.set_data(range(len(filtered)), filtered)
    ax2.relim()
    ax2.autoscale_view()

    # Only show valid heart-rate band in FFT
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

    line3.set_data(freqs[mask], fft[mask])
    ax3.relim()
    ax3.autoscale_view()

        # ---------------- BPM plot ----------------
    if len(bpm_buffer) > 0:
        line4.set_data(range(len(bpm_buffer)), bpm_buffer)
        ax4.relim()
        ax4.autoscale_view()

    plt.pause(0.001)  # Small pause to refresh plot

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)  # Pretrained face detector

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)   # Open default webcam

signal_buffer = []          # Stores green channel signal over time
times = []                  # Stores timestamps (optional)

start_time = time.time()    # Start timer

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()   # Capture frame

    if not ret:               # Exit if camera fails
        break

    frame = cv2.flip(frame, 1)  # Mirror image (more natural)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:        # If face detected
        (x, y, w, h) = faces[0]  # Use first face

        # -----------------------------
        # Forehead ROI (heuristic)
        # -----------------------------
        fx = int(x + 0.3*w)   # Shift right 30% into face
        fy = int(y + 0.1*h)   # Move down 10% (top of face)
        fw = int(0.4*w)       # ROI width = 40% of face
        fh = int(0.2*h)       # ROI height = 20% of face

        roi = frame[fy:fy+fh, fx:fx+fw]  # Extract ROI

        if roi.size != 0:     # Ensure ROI is valid
            # Compute average green intensity in ROI
            green_avg = np.mean(roi[:, :, 1])

            # Append to time-series buffer
            signal_buffer.append(green_avg)

            # Store timestamp (not strictly needed for FFT)
            times.append(time.time() - start_time)

            # Maintain sliding window (fixed buffer size)
            if len(signal_buffer) > BUFFER_SIZE:
                signal_buffer.pop(0)
                times.pop(0)

            # Compute BPM
            bpm = compute_bpm(np.array(signal_buffer), FPS)
            if bpm is not None:
                bpm_buffer.append(bpm)
                if len(bpm_buffer) > BUFFER_SIZE:
                    bpm_buffer.pop(0)



            # Update live plots
            update_plots(signal_buffer, FPS,  bpm_buffer)

            # Draw rectangle around forehead ROI
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)

            # Display BPM on screen
            if bpm is not None:
                cv2.putText(frame, f"BPM: {int(bpm)}",
                            (30, 50),                    # Text position
                            cv2.FONT_HERSHEY_SIMPLEX,    # Font
                            1,                           # Size
                            (0, 255, 0),                 # Color
                            2)                           # Thickness

    # Show webcam feed
    cv2.imshow("rPPG Heart Rate", frame)

    # Exit if ESC key pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()        # Release webcam
cv2.destroyAllWindows()  # Close OpenCV windows
plt.ioff()           # Turn off interactive plotting
plt.show()           # Keep plots open after exit