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
      // Create a canvas to manipulate the frame
      const canvas = document.createElement("canvas");
      const video = webcamRef.current.video;
      const context = canvas.getContext("2d");

      // Set the canvas size to 64x64 for resizing
      canvas.width = 64;
      canvas.height = 64;

      // Draw the video frame onto the canvas at 64x64 resolution
      context.drawImage(video, 0, 0, 64, 64);

      // Convert the canvas content to a Blob (JPEG format)
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));

      // Prepare the form data
      const formData = new FormData();
      formData.append("image", blob);

      try {
        // Send the resized frame to your backend API
        const response = await axios.post("http://127.0.0.1:5000/detect", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        setPrediction(response.data.predicted_letter || "No prediction received");
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
      <Webcam 
        audio={false} 
        ref={webcamRef} 
        screenshotFormat="image/jpeg" 
        className="webcam"
        videoConstraints={{
          width: 640,
          height: 480,
          facingMode: { exact: "environment" }
        }} 
      />
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
