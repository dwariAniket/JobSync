import React from "react";
import AD1 from "./AD1";
import AD2 from "./AD2";

export default function ADyes({ formData, handleInputChange }) {
    const [selectedEducation, setSelectedEducation] = React.useState("");

    const handleEducation = (value) => {
        setSelectedEducation(value);
        handleInputChange({ target: { name: "education", value } });
    };

    return (
        <div style={{
            backgroundColor: 'white',
            borderRadius: '10px',
            padding: '20px',
            marginTop: '20px'
        }}>
            {/* Education Selection Section */}
            <div style={{
                marginBottom: '30px'
            }}>
                <p style={{
                    marginBottom: '15px',
                    fontSize: '16px'
                }}>
                    What are you currently pursuing?
                </p>

                <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '10px'
                }}>
                    {["10th or below 10th", "12th pass", "Diploma", "ITI", "Graduate", "Post Graduate"].map(
                        (education, index) => (
                            <button
                                key={index}
                                style={{
                                    padding: '8px 15px',
                                    borderRadius: '10px',
                                    backgroundColor: selectedEducation === education ? '#DAF7A6' : 'white',
                                    border: '1px solid #ddd',
                                    cursor: 'pointer',
                                    flex: '1 1 calc(33% - 10px)',
                                    minWidth: '120px',
                                    textAlign: 'center'
                                }}
                                onClick={() => handleEducation(education)}
                            >
                                {education}
                            </button>
                        )
                    )}
                </div>
            </div>

            {/* Render Additional Details Based on Education Selection */}
            <div>
                {(selectedEducation === "10th or below 10th" || selectedEducation === "12th pass") && (
                    <AD1 formData={formData} handleInputChange={handleInputChange} />
                )}
                {["Diploma", "ITI", "Graduate", "Post Graduate"].includes(selectedEducation) && (
                    <AD2 formData={formData} handleInputChange={handleInputChange} />
                )}
            </div>
        </div>
    );
}