import { useState } from "react";
import Button from "./components/Button";
import Alert from "./components/Button";
import Recording from "./components/recording";
import { AudioRecorder, useAudioRecorder } from "react-audio-voice-recorder"

import "./App.css"

function App() {
  const recorderControls = useAudioRecorder()
  const [alertVisible, setAlertVisibility] = useState(false);

  const handleRecordClick = () => {
    if (alertVisible != false) {
      return;
    }
    setAlertVisibility(true);
    console.log("Button clicked");
  }
  const addAudioElement = (blob: Blob | MediaSource) => {
    const url = URL.createObjectURL(blob);
    const audio = document.createElement("audio");
    audio.src = url;
    audio.controls = true;
    document.body.appendChild(audio);
  };

  return (
    <div >
      {alertVisible && <Alert color="secondary" onClick={() => setAlertVisibility(false)}> Stop </Alert>}
      <Button color="primary" onClick={() => handleRecordClick()}>
        Rec
      </Button>
      <AudioRecorder
        onRecordingComplete={addAudioElement}
        audioTrackConstraints={{
          noiseSuppression: true,
          echoCancellation: true,
        }}
        downloadOnSavePress={true}
        downloadFileExtension="wav"
      />
      <Recording />
    </div>
  );
}

export default App;
