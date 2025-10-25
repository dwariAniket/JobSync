import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
export default function JobDetails() {
    const navigate = useNavigate();
    const [errMsg, setErrMsg] = useState('');
    const [fdata, setFdata] = useState({
        jobtitle: '',
        joblocation: '',
        openings: '',
        minexp: '',
        maxexp: '',
        minsalary: '',
        maxsalary: '',
        jobdes: '',
        compname: '',
        contactperson: '',
        phonenumber: '',
        email: '',
        contactpersonprofile: '',
    });
    const [showExitPopup, setShowExitPopup] = useState(false);
    const handleExitClick = () => {
        setShowExitPopup(true);
    };
    const handleConfirmExit = () => {
        setShowExitPopup(false);
        console.log("User chose YES");
        // Clear local storage and navigate to home on logout
        localStorage.removeItem("employerdata");
        localStorage.removeItem("employerpost");
        navigate('/');
    };
    const handleCancelExit = () => {
        setShowExitPopup(false);
        console.log("User chose NO");
    };
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFdata((prev) => ({
            ...prev,
            [name]: value,
        }));
        // Clear error message for the specific field when user starts typing
        if (errMsg) setErrMsg('');
    };
    const states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ];
    const profile = [
        "HR", "OWNER", "MANAGER", "OTHER"
    ];
    const handlesubmit = (e) => {
        e.preventDefault(); // Prevent default form submission behavior
        console.log("Submitting data:", fdata);
        const { jobtitle, joblocation, openings, minexp, maxexp, minsalary, maxsalary, jobdes, compname, contactperson, phonenumber, email, contactpersonprofile } = fdata;
        // Basic validation
        if (!jobtitle || !joblocation || !openings || !minexp || !maxexp || !minsalary || !maxsalary || !jobdes || !compname || !contactperson || !phonenumber || !email || !contactpersonprofile) {
            setErrMsg("All fields are required.");
            return;
        }
        // Additional validation for numbers
        if (isNaN(openings) || isNaN(minexp) || isNaN(maxexp) || isNaN(minsalary) || isNaN(maxsalary)) {
            setErrMsg("Openings, Experience, and Salary fields must be numbers.");
            return;
        }
        if (parseInt(minexp) > parseInt(maxexp)) {
            setErrMsg("Minimum experience cannot be greater than maximum experience.");
            return;
        }
        if (parseInt(minsalary) > parseInt(maxsalary)) {
            setErrMsg("Minimum salary cannot be greater than maximum salary.");
            return;
        }
        const employerdata = JSON.parse(localStorage.getItem("employerdata"));
        if (!employerdata || !employerdata.savedUser) {
            setErrMsg("Employer data not found. Please login again.");
            navigate('/employer-login'); // Redirect to login
            return;
        }
        const EmployerData = {
            ...fdata,
            EmployerfName: employerdata.savedUser.fname,
            EmployerlName: employerdata.savedUser.lname,
            EmployerEmail: employerdata.savedUser.email,
            EmployerID: employerdata.savedUser._id,
        };
        fetch('http://192.168.29.18:4000/job/submit', {
            method: 'POST',
            headers: {
                'Content-type': 'application/json',
            },
            body: JSON.stringify(EmployerData),
        })
            .then((res) => {
                console.log("Response received:", res);
                if (!res.ok) {
                    const contentType = res.headers.get("content-type");
                    if (contentType && contentType.indexOf("application/json") !== -1) {
                        return res.json().then(err => { throw new Error(err.error || `HTTP Error! status: ${res.status}`); });
                    } else {
                        return res.text().then(text => { throw new Error(text || `HTTP Error! status: ${res.status}`); });
                    }
                }
                return res.json();
            })
            .then((data) => {
                console.log("Response Data:", data);
                if (data.error) {
                    setErrMsg(data.error);
                } else {
                    alert('Job posted successfully!');
                    localStorage.setItem("employerpost", JSON.stringify(data));
                    console.log(localStorage.getItem("employerpost"));
                    setFdata({
                        jobtitle: '', joblocation: '', openings: '', minexp: '', maxexp: '',
                        minsalary: '', maxsalary: '', jobdes: '', compname: '', contactperson: '',
                        phonenumber: '', email: '', contactpersonprofile: '',
                    });
                    setErrMsg('');
                }
            })
            .catch((error) => {
                console.error('Fetch Error:', error);
                setErrMsg(`Submission failed: ${error.message}. Please try again.`);
            });
    };
    return (
        <div style={{
            backgroundColor: '#F0F8FF',
            minHeight: '1600px', // This might need adjustment based on content
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center', // Center content horizontally
            paddingBottom: '50px', // Add some padding at the bottom
        }}>
            {/* Exit Popup */}
            {showExitPopup && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    height: '100vh',
                    width: '100vw',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    zIndex: 999,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                }}>
                    <div style={{
                        backgroundColor: 'white',
                        width: '90%',
                        maxWidth: '400px',
                        border: '2px solid #ddd',
                        borderRadius: '10px',
                        boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
                        padding: '30px',
                        zIndex: 1000,
                        textAlign: 'center',
                    }}>
                        <p style={{
                            fontWeight: 'bold',
                            fontSize: '18px',
                            marginBottom: '25px',
                            color: '#333',
                        }}>Do you want to sign out?</p>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-around',
                            gap: '20px', // Space between popup buttons
                        }}>
                            <button
                                onClick={handleConfirmExit}
                                style={{
                                    padding: '10px 25px',
                                    backgroundColor: '#4CAF50', // Green
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '16px',
                                    fontWeight: 'bold',
                                    transition: 'background-color 0.3s ease',
                                }}
                            >
                                Yes
                            </button>
                            <button
                                onClick={handleCancelExit}
                                style={{
                                    padding: '10px 25px',
                                    backgroundColor: '#f44336', // Red
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '16px',
                                    fontWeight: 'bold',
                                    transition: 'background-color 0.3s ease',
                                }}
                            >
                                No
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {/* Header */}
            <div style={{
                // backgroundColor: 'white',
                height: '5vh',
                width: '100%',
                // borderBottom: '1px solid #ddd',
                display: 'flex',
                justifyContent: 'flex-end', // Align buttons to the right
                alignItems: 'center',
                paddingRight: '20px',
                boxSizing: 'border-box', // Include padding in width/height
                marginTop:'12px',
                gap:'20px'
            }}>
                <button
                    style={{
                        height: '35px',
                        borderRadius: '10px',
                        backgroundColor: '#0476D0',
                        color: 'white',
                        border: '1px solid #0476D0',
                        cursor: 'pointer',
                        marginLeft: '10px', // Space between buttons
                        padding: '0 15px', // Horizontal padding
                        fontSize: '14px',
                        fontWeight: 'bold',
                    }}
                    onClick={() => navigate('/viewapplicant')}
                >
                    View Applicants
                </button>
                <button
                    style={{
                        height: '35px',
                        borderRadius: '10px',
                        backgroundColor: '#0476D0',
                        color: 'white',
                        border: '1px solid #0476D0',
                        cursor: 'pointer',
                        marginLeft: '10px', // Space between buttons
                        padding: '0 15px', // Horizontal padding
                        fontSize: '14px',
                        fontWeight: 'bold',
                    }}
                    onClick={handleExitClick}
                >
                    Log out
                </button>
            </div>
            {/* Banner Section */}
            <div style={{
                backgroundColor: 'white',
                background: 'linear-gradient(to right, #0476D0 70%, white 30%)',
                height: '20vh',
                width: '96%',
                borderRadius: '10px',
                marginTop: '10vh', // Spacing from header
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between', // Distribute content
                padding: '20px',
                boxSizing: 'border-box',
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)', // Subtle shadow
            }}>
                <div>
                    <p style={{
                        color: 'white',
                        fontSize: '24px',
                        fontWeight: 'bold',
                    }}>Post Job In Just 2 Minutes!</p>
                    <ul style={{
                        color: 'white',
                        fontSize: '18px',
                        marginTop: '10px',
                        listStyleType: 'disc', // Default disc for list items
                        marginLeft: '20px',
                    }}>
                        <li>Get Calls Directly From Candidates</li>
                        <li>Get Access To The DataBase Over 3.5 Cr. Candidates</li>
                    </ul>
                </div>
                <p style={{
                    fontSize: '24px',
                    color: '#035391',
                    textAlign: 'right',
                    lineHeight: '1.2',
                    marginRight: '20px',
                }}>
                    <i>India's <br /> <b>Best</b> <br /> Hiring Platform</i>
                </p>
            </div>
            {/* Error Message */}
            {errMsg && <p style={{
                color: 'red',
                fontSize: '15px',
                fontWeight: 'bold',
                marginTop: '10px',
                marginBottom: '10px',
                alignSelf: 'center', // Center error message
            }}>{errMsg}</p>}
            {/* Basic Job Details */}
            <p style={{
                color: '#0476D0',
                fontSize: '20px',
                fontWeight: 'bold',
                marginTop: '40px', // Spacing from previous section/element
                alignSelf: 'flex-start', // Align to the left
                marginLeft: '2%', // Match container padding
            }}><b>Basic Job Details</b></p>
            <div style={{
                backgroundColor: 'white',
                width: '96%', // Full width of the main content area
                maxWidth: '800px', // Max width for better readability on large screens
                border: '1px solid #ddd',
                borderRadius: '10px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '20px', // Space between form groups
                marginTop: '20px',
                boxSizing: 'border-box',
                boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
            }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="jobtitle" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Job Title</label>
                    <input
                        type="text"
                        id="jobtitle"
                        name="jobtitle"
                        value={fdata.jobtitle}
                        onChange={handleInputChange}
                        placeholder="Enter the job title"
                        style={{
                            height: '45px',
                            width: '100%',
                            borderRadius: '8px',
                            border: '1px solid #ccc',
                            padding: '0 15px',
                            fontSize: '16px',
                            boxSizing: 'border-box',
                        }}
                    />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="joblocation" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Job Location</label>
                    <select
                        id="joblocation"
                        name="joblocation"
                        value={fdata.joblocation}
                        onChange={handleInputChange}
                        style={{
                            height: '45px',
                            width: '100%',
                            borderRadius: '8px',
                            border: '1px solid #ccc',
                            padding: '0 15px',
                            fontSize: '16px',
                            boxSizing: 'border-box',
                            backgroundColor: 'white',
                            cursor: 'pointer',
                        }}
                    >
                        <option value="">Select a state</option>
                        {states.map((state) => (
                            <option key={state} value={state}>{state}</option>
                        ))}
                    </select>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="openings" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Openings</label>
                    <input
                        type="number"
                        id="openings"
                        name="openings"
                        value={fdata.openings}
                        onChange={handleInputChange}
                        placeholder="Enter no. of openings"
                        style={{
                            height: '45px',
                            width: '100%',
                            borderRadius: '8px',
                            border: '1px solid #ccc',
                            padding: '0 15px',
                            fontSize: '16px',
                            boxSizing: 'border-box',
                        }}
                    />
                </div>
            </div>
            {/* Candidate Requirements */}
            <p style={{
                color: '#0476D0',
                fontSize: '20px',
                fontWeight: 'bold',
                marginTop: '40px',
                alignSelf: 'flex-start',
                marginLeft: '2%',
            }}><b>Candidate Requirements</b></p>
            <div style={{
                backgroundColor: 'white',
                width: '96%',
                maxWidth: '800px',
                border: '1px solid #ddd',
                borderRadius: '10px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '20px',
                marginTop: '20px',
                boxSizing: 'border-box',
                boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
            }}>
                <div style={{
                    display: 'flex',
                    gap: '20px', // Space between horizontal inputs
                    alignItems: 'flex-end', // Align inputs at the bottom
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="minexp" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Minimum Experience</label>
                        <input
                            type="number"
                            id="minexp"
                            name="minexp"
                            value={fdata.minexp}
                            onChange={handleInputChange}
                            placeholder="e.g., 0 for freshers"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="maxexp" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Maximum Experience</label>
                        <input
                            type="number"
                            id="maxexp"
                            name="maxexp"
                            value={fdata.maxexp}
                            onChange={handleInputChange}
                            placeholder="e.g., 10 for experienced"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="minsalary" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Monthly In-Hand Salary</label>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        width: '100%',
                    }}>
                        <input
                            type="number"
                            id="minsalary"
                            name="minsalary"
                            value={fdata.minsalary}
                            onChange={handleInputChange}
                            placeholder="e.g., 10000"
                            style={{
                                height: '45px',
                                width: 'calc(50% - 15px)', // Adjusted width to account for "to" text
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                                borderTopLeftRadius: '8px',
                                borderBottomLeftRadius: '8px',
                                borderRight: 'none',
                            }}
                        />
                        <span style={{
                            border: '1px solid #ccc',
                            height: '45px',
                            display: 'flex',
                            alignItems: 'center',
                            padding: '0 10px',
                            backgroundColor: '#f0f0f0',
                            color: '#555',
                            fontWeight: 'bold',
                        }}>to</span>
                        <input
                            type="number"
                            id="maxsalary"
                            name="maxsalary"
                            value={fdata.maxsalary}
                            onChange={handleInputChange}
                            placeholder="e.g., 30000"
                            style={{
                                height: '45px',
                                width: 'calc(50% - 15px)', // Adjusted width to account for "to" text
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                                borderTopRightRadius: '8px',
                                borderBottomRightRadius: '8px',
                                borderLeft: 'none',
                            }}
                        />
                    </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="jobdes" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Job Info./Job Description</label>
                    <textarea
                        id="jobdes"
                        name="jobdes"
                        value={fdata.jobdes}
                        onChange={handleInputChange}
                        placeholder="Provide a detailed job description"
                        rows="4" // Make it a textarea for better input
                        style={{
                            height: 'auto',
                            minHeight: '80px',
                            width: '100%',
                            borderRadius: '8px',
                            border: '1px solid #ccc',
                            padding: '10px 15px', // Adjusted padding for textarea
                            fontSize: '16px',
                            boxSizing: 'border-box',
                        }}
                    />
                </div>
            </div>
            {/* Company Details */}
            <p style={{
                color: '#0476D0',
                fontSize: '20px',
                fontWeight: 'bold',
                marginTop: '40px',
                alignSelf: 'flex-start',
                marginLeft: '2%',
            }}><b>About Your Industry</b></p>
            <div style={{
                backgroundColor: 'white',
                width: '96%',
                maxWidth: '800px',
                border: '1px solid #ddd',
                borderRadius: '10px',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '20px',
                marginTop: '20px',
                boxSizing: 'border-box',
                boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
            }}>
                <div style={{
                    display: 'flex',
                    gap: '20px',
                    alignItems: 'flex-end',
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="compname" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Company Name</label>
                        <input
                            type="text"
                            id="compname"
                            name="compname"
                            value={fdata.compname}
                            onChange={handleInputChange}
                            placeholder="e.g., Tata Consultancy Services"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="contactperson" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Contact Person's Name</label>
                        <input
                            type="text"
                            id="contactperson"
                            name="contactperson"
                            value={fdata.contactperson}
                            onChange={handleInputChange}
                            placeholder="e.g., Sourav Nag"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                </div>
                <div style={{
                    display: 'flex',
                    gap: '20px',
                    alignItems: 'flex-end',
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="phonenumber" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Contact Number</label>
                        <input
                            type="tel"
                            id="phonenumber"
                            name="phonenumber"
                            value={fdata.phonenumber}
                            onChange={handleInputChange}
                            placeholder="e.g., 9876543210"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                        <label htmlFor="email" style={{
                            marginBottom: '8px',
                            fontSize: '16px',
                            color: '#333',
                            fontWeight: '500',
                        }}>Email-ID</label>
                        <input
                            type="email"
                            id="email"
                            name="email"
                            value={fdata.email}
                            onChange={handleInputChange}
                            placeholder="e.g., souravnag@gmail.com"
                            style={{
                                height: '45px',
                                width: '100%',
                                borderRadius: '8px',
                                border: '1px solid #ccc',
                                padding: '0 15px',
                                fontSize: '16px',
                                boxSizing: 'border-box',
                            }}
                        />
                    </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <label htmlFor="contactpersonprofile" style={{
                        marginBottom: '8px',
                        fontSize: '16px',
                        color: '#333',
                        fontWeight: '500',
                    }}>Contact Person Profile</label>
                    <select
                        id="contactpersonprofile"
                        name="contactpersonprofile"
                        value={fdata.contactpersonprofile}
                        onChange={handleInputChange}
                        style={{
                            height: '45px',
                            width: 'calc(50% - 10px)', // Adjust width for this select
                            borderRadius: '8px',
                            border: '1px solid #ccc',
                            padding: '0 15px',
                            fontSize: '16px',
                            boxSizing: 'border-box',
                            backgroundColor: 'white',
                            cursor: 'pointer',
                        }}
                    >
                        <option value="">Select your profile</option>
                        {profile.map((p) => (
                            <option key={p} value={p}>{p}</option>
                        ))}
                    </select>
                </div>
            </div>
            {/* Disclaimer Box */}
            <div style={{
                backgroundColor: '#F5F5F5',
                width: '96%',
                maxWidth: '800px',
                border: '1px solid #ddd',
                borderRadius: '10px',
                padding: '20px',
                marginTop: '40px',
                fontSize: '14px',
                color: '#555',
                lineHeight: '1.5',
                boxSizing: 'border-box',
            }}>
                <p>By submitting this job posting, I provide consent to be contacted by candidates and customer support via WhatsApp, SMS, phone calls, and email.<br /><br />I understand that KYC verification is required and acknowledge the updated refund and privacy policies linked below.</p>
            </div>
            {/* Warning Text */}
            <p style={{
                fontSize: '14px',
                color: 'orange', // Changed to orange for warning
                marginTop: '15px',
                alignSelf: 'flex-start',
                marginLeft: '2%',
            }}>
                ⚠️ Asking job seekers for any kind of payment is strictly prohibited.
            </p>
            {/* Submit Button */}
            <button
                style={{
                    borderRadius: '8px',
                    height: '45px',
                    width: '150px',
                    backgroundColor: '#0476D0',
                    color: 'white',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '18px',
                    fontWeight: 'bold',
                    marginTop: '30px',
                    transition: 'background-color 0.3s ease',
                    alignSelf: 'flex-start',
                    marginLeft: '2%',
                }}
                onClick={handlesubmit}
                onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'} // Apply hover style directly
                onMouseLeave={(e) => e.target.style.backgroundColor = '#0476D0'} // Revert on mouse leave
            >
                Submit
            </button>
        </div>
    );
}