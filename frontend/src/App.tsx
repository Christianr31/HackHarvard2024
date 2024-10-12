import { useState } from "react";
import Button from "./components/Button";
import Alert from "./components/Button";

import "./App.css"

function App() {
  const [alertVisible, setAlertVisibility] = useState(false);

  const handleRecordClick = () => {
    if (alertVisible != false) {
      return;
    }
    setAlertVisibility(true);
    console.log("Button clicked");
  }

  return (
    <div >
      {alertVisible && <Alert color="secondary" onClick={() => setAlertVisibility(false)}> Stop </Alert>}
      <Button color="primary" onClick={() => handleRecordClick()}>
        Rec
      </Button>
    </div>
  );
}

export default App;
