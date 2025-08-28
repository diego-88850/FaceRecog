import React, { useRef, useState } from "react";
import Webcam from "react-webcam";

export default function FaceAuthMVP() {
  const webcamRef = useRef(null);
  const [mode, setMode] = useState("id"); // "id" or "auth"
  const [username, setUsername] = useState("");
  const [result, setResult] = useState(null);

  const capture = async () => {
    // Example stub — replace with your real API call
    const imageSrc = webcamRef.current?.getScreenshot({ width: 320, height: 240 });
    if (!imageSrc) return;

    // Convert dataURL -> Blob
    const blob = await (await fetch(imageSrc)).blob();

    const fd = new FormData();
    fd.append("mode", mode);
    if (mode === "auth") fd.append("username", username);
    fd.append("image", new File([blob], "snap.jpg", { type: "image/jpeg" }));

    const res = await fetch("/verify", { method: "POST", body: fd });
    const json = await res.json();
    setResult(json);
  };

  return (
    <div style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>Face Auth MVP</h1>

      <div style={{ display: "flex", gap: "1rem" }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"      // <-- use JPEG
          screenshotQuality={0.92}           // quality knob works here
          videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
        />

        <div>
          <label>
            Mode:&nbsp;
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="id">ID</option>
              <option value="auth">Auth</option>
            </select>
          </label>

          {mode === "auth" && (
            <div style={{ marginTop: "0.5rem" }}>
              <label>
                Username:&nbsp;
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                />
              </label>
            </div>
          )}

          <button
            onClick={capture}
            style={{ padding: "0.5rem 1rem", marginTop: "1rem" }}
          >
            Verify
          </button>

          {result && (
            <pre
              style={{
                background: "#f5f5f5",
                padding: "0.75rem",
                borderRadius: 8,
                marginTop: "1rem",
              }}
            >
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
