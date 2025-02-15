import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

const App = () => {
  const webcamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState("");
  let intervalRef = useRef(null);

  const captureAndSendFrame = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      const blob = await fetch(imageSrc).then((res) => res.blob());

      const formData = new FormData();
      formData.append("frame", blob);

      try {
        await axios.post("https://webhook.site/e4dadf41-fe60-4fda-bcd7-2e4da229bbd1", formData);
        setPrediction("Frame sent successfully!");
      } catch (error) {
        setPrediction("Failed to send frame.");
      }
    }
  };

  const startStreaming = () => {
    if (!isStreaming) {
      setIsStreaming(true);
      intervalRef.current = setInterval(captureAndSendFrame, 2000);
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
      <h1>Sign Language Interpreter</h1>
      <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" className="webcam" />
      <div>
        <button onClick={startStreaming} className="start" disabled={isStreaming}>
          Start
        </button>
        <button onClick={stopStreaming} className="stop" disabled={!isStreaming}>
          Stop
        </button>
      </div>
      <h2 className={isStreaming ? "active" : "stopped"}>{prediction}</h2>
    </div>
  );
};

export default App;
