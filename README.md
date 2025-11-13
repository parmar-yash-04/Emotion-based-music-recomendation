<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Emotion-Based Music Recommendation System — README</title>
  <style>
    :root{
      --bg:#0f1724; --card:#0b1220; --muted:#94a3b8; --accent:#7c3aed;
      --accent-2:#06b6d4; --glass: rgba(255,255,255,0.03);
      --mono: "SFMono-Regular", Consolas, "Roboto Mono", "Liberation Mono", monospace;
      color-scheme: dark;
    }
    html,body{height:100%;margin:0;background:linear-gradient(180deg,#071029 0%, #08131f 100%);font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial; color:#e6eef8;}
    .container{max-width:980px;margin:36px auto;padding:28px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:12px;box-shadow:0 6px 30px rgba(2,6,23,0.6);}
    header{display:flex;gap:18px;align-items:center;margin-bottom:18px}
    .logo{width:64px;height:64px;border-radius:10px;background:linear-gradient(90deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;font-weight:700;font-size:22px;color:white}
    h1{font-size:20px;margin:0}
    p.lead{margin:6px 0 0;color:var(--muted)}
    .badges{margin-left:auto;display:flex;gap:8px}
    .badge{background:var(--glass);padding:6px 10px;border-radius:8px;font-size:13px;color:var(--muted)}
    section{margin-top:18px}
    h2{font-size:16px;margin:12px 0}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    pre.code{background:#071826;border:1px solid rgba(255,255,255,0.03);padding:12px;border-radius:8px;color:#cfeef8;overflow:auto;font-family:var(--mono);font-size:13px}
    ul{margin:8px 0 0 18px;color:var(--muted)}
    .meta{color:var(--muted);font-size:13px}
    .btn{display:inline-block;padding:10px 14px;border-radius:10px;text-decoration:none;font-weight:600}
    .btn-primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:white}
    footer{margin-top:20px;color:var(--muted);font-size:13px;text-align:center}
    .screenshot{width:100%;height:220px;border-radius:8px;background:linear-gradient(180deg,#071c2c,#04101b);display:flex;align-items:center;justify-content:center;color:var(--muted);border:1px dashed rgba(255,255,255,0.03)}
    @media(max-width:720px){.grid{grid-template-columns:1fr}.badges{display:none}.logo{width:56px;height:56px}}
  </style>
</head>
<body>
  <div class="container" role="main">
    <header>
      <div class="logo">EBM</div>
      <div>
        <h1>Emotion-Based Music Recommendation System</h1>
        <p class="lead">AI-powered web app that detects facial emotions and recommends mood-specific songs (Flask + CNN + OpenCV).</p>
        <div class="meta">Tools: TensorFlow · Keras · OpenCV · Flask · NumPy · Pandas · Matplotlib</div>
      </div>

      <div class="badges" aria-hidden="true">
        <span class="badge">Python</span>
        <span class="badge">CNN</span>
        <span class="badge">Flask</span>
      </div>
    </header>

    <section>
      <h2>Quick demo</h2>
      <div class="screenshot">[Insert screenshot / GIF of app here]</div>
      <p class="meta" style="margin-top:8px">Open the Flask web app and allow webcam access to try real-time emotion detection and music recommendations.</p>
    </section>

    <section>
      <h2>Features</h2>
      <ul>
        <li>Real-time facial emotion recognition using a CNN trained on FER2013</li>
        <li>Flask web application for live webcam inference</li>
        <li>30+ curated songs per emotion with instant YouTube redirection</li>
        <li>Emotion analytics dashboard showing detection counts and trends</li>
        <li>Lightweight, fast, and easy to deploy</li>
      </ul>
    </section>

    <section>
      <h2>Installation</h2>
      <pre class="code">git clone https://github.com/parmar-yash-04/emotion-music-recommendation.git
cd emotion-music-recommendation
pip install -r requirements.txt</pre>

      <p class="meta">Run the Flask app:</p>
      <pre class="code">python app.py
# then open: http://127.0.0.1:5000/</pre>
    </section>

    <section>
      <h2>Usage</h2>
      <ul>
        <li>Start the app and allow webcam access</li>
        <li>Model detects your emotion in real-time</li>
        <li>Song list appears based on the detected mood</li>
        <li>Click any song → instantly open on YouTube</li>
        <li>Emotion analytics track detection and user activity</li>
      </ul>
    </section>

    <section class="grid">
      <div>
        <h2>Model Details</h2>
        <ul>
          <li>Dataset: FER2013 (48×48 grayscale faces)</li>
          <li>Architecture: Custom CNN (Conv + Pool + Dense layers)</li>
          <li>Optimizer: Adam, Loss: Categorical Crossentropy</li>
          <li>Accuracy: ~94% on test data</li>
        </ul>
      </div>

      <div>
        <h2>Analytics</h2>
        <p class="meta">Dashboard displays:</p>
        <ul>
          <li>Most detected emotions</li>
          <li>Emotion distribution</li>
          <li>User click behavior on songs</li>
          <li>Recent detection timeline</li>
        </ul>
      </div>
    </section>

    <section>
      <h2>Song JSON Format</h2>
      <pre class="code">[
  {
    "title": "Song Name",
    "artist": "Artist Name",
    "youtube": "https://www.youtube.com/watch?v=xxx"
  }
]</pre>
    </section>

    <section>
      <h2>How It Works</h2>
      <ol>
        <li>Capture webcam frame using OpenCV</li>
        <li>Convert to 48×48 grayscale and preprocess</li>
        <li>Predict emotion using trained CNN model</li>
        <li>Load song list mapped to that emotion</li>
        <li>User clicks → redirect to YouTube</li>
        <li>Log events for analytics</li>
      </ol>
    </section>

    <section style="display:flex;gap:10px;align-items:center;margin-top:12px">
      <a class="btn btn-primary" href="https://github.com/parmar-yash-04" target="_blank">View on GitHub</a>
      <div style="margin-left:auto;color:var(--muted);font-size:13px">Last updated: <strong><!-- Insert date --></strong></div>
    </section>

    <footer>
      Built with ❤️ by Yash Parmar — <a href="https://github.com/parmar-yash-04" style="color:var(--accent)">GitHub</a>
    </footer>
  </div>
</body>
</html>
