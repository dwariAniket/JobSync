import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ADyes from './ADyes';
import ADno from './ADno';
import { storage } from './firebase';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';

export default function ApplicantDetails() {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [setyes, setSetyes] = useState(false);
    const [setno, setSetno] = useState(false);
    const [errMsg, setErrMsg] = useState('');
    const [cvURL, setCvURL] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [showConfirmation, setShowConfirmation] = useState(false);


    const [fData, setFData] = useState({
        schoolmed: '',
        haveskills: '',
        skills: '',
        specialization: '',
        collegename: '',
        completionyear: '',
        pursuingeducation: '',
        education: '',
        status: '',
    });

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (!selectedFile) return;

        // Validate file type
        const validTypes = ['application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(selectedFile.type)) {
            setErrMsg('Only PDF, DOC, and DOCX files are allowed');
            return;
        }

        // Validate file size (5MB max)
        if (selectedFile.size > 5 * 1024 * 1024) {
            setErrMsg('File size exceeds 5MB limit');
            return;
        }

        setFile(selectedFile);
        setErrMsg('');
        uploadFile(selectedFile);
    };

    const uploadFile = (file) => {
        setIsUploading(true);
        setUploadProgress(0);

        const fileId = Date.now();
        const storagePath = `cv/${fileId}_${file.name}`;
        const cvRef = ref(storage, storagePath);

        const uploadTask = uploadBytes(cvRef, file);

        uploadTask
            .then((snapshot) => getDownloadURL(snapshot.ref))
            .then((url) => {
                setCvURL(url);
                console.log("CV URL:", url);
            })
            .catch((error) => {
                console.error("Upload failed:", error);
                setErrMsg("CV upload failed. Please try again.");
            })
            .finally(() => {
                setIsUploading(false);
            });
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFData(prev => ({ ...prev, [name]: value }));
    };

    const handlePursuingEducation = (value) => {
        setFData(prev => ({ ...prev, pursuingeducation: value }));
        setSetyes(value === "Yes");
        setSetno(value === "No");
    };

    const handleSubmitClick = () => {
        if (!cvURL) {
            setErrMsg("All fields are required");
            return;
        }
        setShowConfirmation(true);
    };

    const handleConfirmSubmit = async (confirmed) => {
        setShowConfirmation(false);
        if (!confirmed) return;

        try {
            const employeedata = JSON.parse(localStorage.getItem("employeedata"));
            const applicantdata = {
                ...fData,
                cvURL,
                EmployeefName: employeedata.savedUser.fname,
                EmployeelName: employeedata.savedUser.lname,
                EmployeeEmail: employeedata.savedUser.email,
                EmployeeID: employeedata.savedUser._id,
            };

            const response = await fetch('http://192.168.29.18:4000/details/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(applicantdata),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            alert('Submitted successfully');
            localStorage.setItem("employeepost", JSON.stringify(data));

        } catch (error) {
            console.error('Submission error:', error);
            setErrMsg(`Submission failed: ${error.message}`);
        }
    };

    return (
        <div style={{
            minHeight: '100vh',
            backgroundColor: '#F0F8FF',
            padding: '20px',
            position: 'relative'
        }}>
            {/* Upper Container */}
            <div style={{
                height: 'auto',
                display: 'flex',
                flexDirection: 'row',
                gap: '50px'
            }}>
                {/* Left Section */}
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    height: 'auto',
                    width: '60%',
                    backgroundColor: 'white',
                    borderRadius: '10px',
                    padding: '20px',
                }}>
                    <div>
                        {errMsg && <p style={{
                            color: 'red',
                            marginBottom: '15px'
                        }}>{errMsg}</p>}

                        <p style={{
                            marginBottom: '15px',
                            fontSize: '16px'
                        }}>Are you currently pursuing your education?</p>
                    </div>

                    <div style={{
                        display: 'flex',
                        gap: '20px',
                    }}>
                        <button
                            onClick={() => handlePursuingEducation("Yes")}
                            style={setyes ? {
                                padding: '8px 15px',
                                borderRadius: '10px',
                                backgroundColor: '#DAF7A6',
                                border: '1px solid #ddd',
                                cursor: 'pointer'
                            } : {
                                padding: '8px 15px',
                                borderRadius: '10px',
                                backgroundColor: 'white',
                                border: '1px solid #ddd',
                                cursor: 'pointer'
                            }}
                        >
                            Yes
                        </button>
                        <button
                            onClick={() => handlePursuingEducation("No")}
                            style={setno ? {
                                padding: '8px 15px',
                                borderRadius: '10px',
                                backgroundColor: '#DAF7A6',
                                border: '1px solid #ddd',
                                cursor: 'pointer'
                            } : {
                                padding: '8px 15px',
                                borderRadius: '10px',
                                backgroundColor: 'white',
                                border: '1px solid #ddd',
                                cursor: 'pointer'
                            }}
                        >
                            No
                        </button>
                    </div>
                </div>

                {/* Right Section - CV Upload */}
                <div style={{
                    backgroundColor: 'white',
                    height: 'auto',
                    width: '25%',
                    padding: '20px',
                    borderRadius: '10px'
                }}>
                    <div style={{ color: '#333', marginBottom: '15px' }}>
                        Upload your updated CV
                    </div>

                    <div style={{
                        border: '2px dashed #ccc',
                        padding: '20px',
                        textAlign: 'center',
                        borderRadius: '5px',
                        marginBottom: '10px'
                    }}>
                        <input
                            type="file"
                            id="cv-upload"
                            accept=".pdf,.doc,.docx"
                            onChange={handleFileChange}
                            style={{ display: 'none' }}
                            disabled={isUploading}
                        />
                        <label
                            htmlFor="cv-upload"
                            style={{
                                display: 'inline-block',
                                padding: '10px 20px',
                                backgroundColor: '#f0f8ff',
                                borderRadius: '5px',
                                cursor: 'pointer',
                                border: '1px solid #ddd',
                                width: '100%',
                                boxSizing: 'border-box',
                                textOverflow: 'ellipsis',
                                overflow: 'hidden',
                                whiteSpace: 'nowrap'
                            }}
                        >
                            {file ? file.name : 'Browse file'}
                        </label>
                    </div>

                    {isUploading && (
                        <div style={{ marginTop: '10px' }}>
                            <p>Uploading... {uploadProgress}%</p>
                            <progress
                                value={uploadProgress}
                                max="100"
                                style={{
                                    width: '100%',
                                    height: '10px'
                                }}
                            />
                        </div>
                    )}
                    {cvURL && !isUploading && (
                        <p style={{
                            color: 'green',
                            marginTop: '10px'
                        }}>CV uploaded successfully</p>
                    )}
                </div>
            </div>

            <div style={{
                display: 'flex',
                justifyContent: 'flex-end',
                marginRight: '240px',
                marginTop: '10px'
            }}>
                <p style={{
                    fontSize: '12px',
                    color: '#666',
                }}><b>Accepted formats: PDF, DOC, DOCX (Max 5MB)</b></p>
            </div>

            {/* Education Details Section */}
            <div style={{ marginTop: '20px' }}>
                {setyes && <ADyes formData={fData} handleInputChange={handleInputChange} />}
                {setno && <ADno formData={fData} handleInputChange={handleInputChange} />}
            </div>

            {/* Submit Button */}
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                marginTop: '30px',
                paddingBottom: '30px'
            }}>
                <button
                    onClick={handleSubmitClick}
                    style={{
                        padding: '12px 24px',
                        borderRadius: '8px',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '16px',
                        minWidth: '200px'
                    }}
                    disabled={isUploading || !cvURL}
                >
                    {isUploading ? 'Uploading...' : 'Submit'}
                </button>
            </div>

            {/* Confirmation Dialog */}
            {showConfirmation && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.5)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 1000
                }}>
                    <div style={{
                        backgroundColor: 'white',
                        padding: '30px',
                        borderRadius: '10px',
                        width: '350px',
                        textAlign: 'center',
                        boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                    }}>
                        <p style={{
                            marginBottom: '25px',
                            fontSize: '18px',
                            fontWeight: '500'
                        }}>Do you want to submit your application?</p>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'center',
                            gap: '20px'
                        }}>
                            <button
                                onClick={() => handleConfirmSubmit(true)}
                                style={{
                                    padding: '10px 20px',
                                    borderRadius: '5px',
                                    backgroundColor: '#4CAF50',
                                    color: 'white',
                                    border: 'none',
                                    cursor: 'pointer',
                                    fontSize: '16px',
                                    minWidth: '100px'
                                }}
                            >
                                Yes
                            </button>
                            <button
                                onClick={() => handleConfirmSubmit(false)}
                                style={{
                                    padding: '10px 20px',
                                    borderRadius: '5px',
                                    backgroundColor: '#f44336',
                                    color: 'white',
                                    border: 'none',
                                    cursor: 'pointer',
                                    fontSize: '16px',
                                    minWidth: '100px'
                                }}
                            >
                                No
                            </button>
                            {errMsg && (
                                <p>All fields are required</p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}