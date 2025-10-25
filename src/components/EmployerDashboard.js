import { useState } from "react";

export default function ExitConfirmation() {
  const [showExitPopup, setShowExitPopup] = useState(false);

  const handleExitClick = () => {
    setShowExitPopup(true);
  };

  const handleConfirmExit = () => {
    setShowExitPopup(false);
    console.log("User chose YES");
    // navigate or close logic here
  };

  const handleCancelExit = () => {
    setShowExitPopup(false);
    console.log("User chose NO");
  };

  return (
    <div style={{ padding: 50 }}>
      <button
        onClick={handleExitClick}
        style={{
          padding: "10px 20px",
          backgroundColor: "tomato",
          color: "white",
          border: "none",
          borderRadius: 5,
          cursor: "pointer",
        }}
      >
        Exit
      </button>

      {showExitPopup && (
        <>
          {/* Dark background overlay */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              height: "100vh",
              width: "100vw",
              backgroundColor: "rgba(0, 0, 0, 0.5)",
              zIndex: 999,
            }}
          />

          {/* Modal box */}
          <div
            style={{
              position: "fixed",
              top: "30%",
              left: "35%",
              width: "30%",
              backgroundColor: "white",
              border: "2px solid #ddd",
              borderRadius: "10px",
              boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
              padding: "20px",
              zIndex: 1000,
            }}
          >
            <p style={{ fontWeight: "bold" }}>Do you want to exit?</p>
            <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-around" }}>
              <button
                onClick={handleConfirmExit}
                style={{
                  padding: "8px 16px",
                  backgroundColor: "green",
                  color: "white",
                  border: "none",
                  borderRadius: 5,
                  cursor: "pointer",
                }}
              >
                Yes
              </button>
              <button
                onClick={handleCancelExit}
                style={{
                  padding: "8px 16px",
                  backgroundColor: "gray",
                  color: "white",
                  border: "none",
                  borderRadius: 5,
                  cursor: "pointer",
                }}
              >
                No
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
