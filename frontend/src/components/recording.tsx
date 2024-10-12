import { AudioRecorder } from 'react-audio-voice-recorder';

const recording = () => {
    // const recorderControls = useAudioRecorder()
    const addAudioElement = (blob: Blob | MediaSource) => {
        const url = URL.createObjectURL(blob);
        const audio = document.createElement("audio");
        audio.src = url;
        audio.controls = true;
        document.body.appendChild(audio);
    };

    return (
        <div>
            <AudioRecorder
                onRecordingComplete={addAudioElement}
                audioTrackConstraints={{
                    noiseSuppression: true,
                    echoCancellation: true,
                }}
                downloadOnSavePress={true}
                downloadFileExtension="wav"
            />
            {/* <button onClick={recorderControls.stopRecording}>Stop recording</button> */}
        </div>
    )
}

export default recording