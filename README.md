# 🎭 FaceFilter AI — Python + OpenCV Web App

Real-time face detection with **11 cool filters**, photo capture, and video recording. Built with **Python**, **OpenCV**, and **Flask**.

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎯 Face Detection | OpenCV Haar Cascade — real-time 30 FPS |
| 🐶 Dog Filter | Ears, nose, tongue & whiskers |
| 😎 Sunglasses | Cool shades with lens tint |
| 👑 Crown | Golden jewelled crown |
| 🌈 Rainbow | Animated color arc |
| ⚡ Neon | Glowing edge detection |
| 📷 Vintage | Sepia + film grain + vignette |
| 🤖 Cyberpunk | HUD overlay + scanlines |
| 🪞 Mirror | Split-screen mirror |
| ✏️ Sketch | Pencil drawing effect |
| 🎮 Pixelate | 8-bit pixel face |
| 📸 Photo Capture | Downloads PNG with filter |
| 🎬 Video Recording | Records WebM with filter |

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open in browser
open http://localhost:5000
```

> Make sure your webcam is connected and accessible.

---

## ☁️ Deploy Free on Render

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3
5. Click **Deploy** — get a public URL in ~2 minutes!

> **Note:** Render free tier spins down after 15min inactivity. First load may take ~30s.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Capture photo |
| `R` | Start/Stop recording |
| `Esc` | Remove filter |
| `1` | No filter (detect only) |
| `2` | Dog |
| `3` | Sunglasses |
| `4` | Crown |
| `5` | Rainbow |
| `6` | Neon |
| `7` | Vintage |
| `8` | Cyberpunk |
| `9` | Mirror |
| `0` | Sketch |
| `-` | Pixelate |

---

## 📁 Project Structure

```
face-filter-app/
├── app.py              # Flask server + MJPEG stream
├── filters.py          # All 10 OpenCV filter functions
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Premium dark UI
├── static/
│   ├── style.css       # Glassmorphism design system
│   └── app.js          # Filter controls + capture + gallery
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+** + **Flask** — web server & MJPEG streaming
- **OpenCV** — face detection + filter rendering
- **HTML/CSS/JS** — premium dark UI frontend
- **MediaRecorder API** — in-browser video recording
