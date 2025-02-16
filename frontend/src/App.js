import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

const App = () => {
  const webcamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState("Waiting for answer...");
  let intervalRef = useRef(null);

  const captureAndSendFrame = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      const blob = await fetch(imageSrc).then((res) => res.blob());

      const formData = new FormData();
      formData.append("frame", blob);

      try {
        // Send frame to backend and receive prediction
        const response = await axios.post("http://127.0.0.1:5000/detect", formData);
        setPrediction(response.data.prediction || "No prediction received");
      } catch (error) {
        console.error("Error receiving prediction:", error);
        setPrediction("Error receiving prediction.");
      }
    }
  };

  const startStreaming = () => {
    if (!isStreaming) {
      setIsStreaming(true);
      intervalRef.current = setInterval(captureAndSendFrame, 2000); // Sends a frame every 2 seconds
    }
  };

  const stopStreaming = () => {
    if (isStreaming) {
      clearInterval(intervalRef.current);
      setIsStreaming(false);
      setPrediction("Stopped sending frames.");
    }
  };

  return (
    <div>
      <img src="/logo.png" alt="App Logo" className="logo" />
      <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" className="webcam" />
      <div>
        <button onClick={startStreaming} className="start" disabled={isStreaming}>
          Start
        </button>
        <button onClick={stopStreaming} className="stop" disabled={!isStreaming}>
          Stop
        </button>
      </div>
      <h2 className={`prediction ${isStreaming ? "active" : "stopped"}`}>
        {prediction}
      </h2>
    </div>
  );
};

export default App;
