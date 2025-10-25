import React from 'react';
import { useNavigate } from "react-router-dom";

export default function Welcome() {
    const navigate = useNavigate();

    const [findjob, setfindjob] = React.useState(false); // Controls Employee Login/Signup visibility
    const [findcandidate, setfindcandidate] = React.useState(false); // Controls Employer Login/Signup visibility

    const [fdata, setfdata] = React.useState({
        email: '',
        pw: '',
    });

    const [errmsg, seterrmsg] = React.useState('');

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setfdata((prev) => ({
            ...prev,
            [name]: value,
        }));
        // Clear error message when user starts typing
        if (errmsg) seterrmsg('');
    };

    const GetDataFromBackendForEmployer = () => {
        const { email, pw } = fdata;
        if (!email || !pw) {
            seterrmsg("All fields are required.");
            return;
        } else {
            fetch('http://192.168.29.18:4000/employer/signin', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(fdata)
            })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(err => { throw new Error(err.error || `HTTP Error! status: ${res.status}`); });
                    }
                    return res.json();
                })
                .then(
                    data => {
                        localStorage.setItem("employerdata", JSON.stringify(data));
                        console.log('Employer data stored:', localStorage.getItem("employerdata"));
                        alert("Successfully logged in as Employer!");
                        navigate('/jobdetails');
                    })
                .catch((error) => {
                    console.error('Employer Login Error:', error);
                    seterrmsg(`Login failed: ${error.message}. Please try again.`);
                });
        }
    };

    const GetDataFromBackendForEmployee = () => {
        const { email, pw } = fdata;
        if (!email || !pw) {
            seterrmsg("All fields are required.");
            return;
        } else {
            fetch('http://192.168.29.18:4000/signin', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(fdata)
            })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(err => { throw new Error(err.error || `HTTP Error! status: ${res.status}`); });
                    }
                    return res.json();
                })
                .then(async (data) => {
                    localStorage.setItem("employeedata", JSON.stringify(data));
                    console.log('Employee data stored:', localStorage.getItem("employeedata"));

                    const detailsResponse = await fetch(`http://192.168.29.18:4000/employee/employeedetails/${data.savedUser.email}`);
                    if (!detailsResponse.ok) {
                        const detailsError = await detailsResponse.json();
                        throw new Error(detailsError.error || "Failed to fetch employee details.");
                    }
                    const detailsData = await detailsResponse.json();

                    console.log("User Details Data:", detailsData);
                    alert("Successfully logged in as Employee!");
                    if (detailsData.details) {
                        navigate('/applicanthome');
                    } else {
                        navigate('/applicantdetails');
                    }
                })
                .catch((error) => {
                    console.error('Employee Login Error:', error);
                    seterrmsg(`Login failed: ${error.message}. Please try again.`);
                });
        }
    };

    return (
        <div style={{
            minHeight: "100vh",
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center', // Center content vertically
            padding: '20px',
            backgroundImage: "url('https://png.pngtree.com/background/20220714/original/pngtree-happy-young-big-isolated-corporate-man-done-his-job-as-vison-picture-image_1607822.jpg')",
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundAttachment: 'fixed', // Keep background fixed when scrolling
            boxSizing: 'border-box', // Include padding in element's total width and height
            position:'relative'
        }}>
            {/* Overlay for better readability */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                height:'100%',
                width: '100%',
                backgroundColor: 'rgba(255, 255, 255, 0.8)', // Semi-transparent white overlay
                zIndex: 1,
            }}></div>

            <div style={{
                position: 'relative', // Ensures content is above the overlay
                zIndex: 2,
                textAlign: 'center',
                marginBottom: '50px', // Space below the main heading
            }}>
                <h1 style={{
                    fontSize: '3.5em', // Larger font size
                    color: '#0476D0', // Blue
                    fontWeight: '800', // Extra bold
                    lineHeight: '1.2',
                    textShadow: '2px 2px 4px rgba(0,0,0,0.1)', // Subtle text shadow
                    margin: 0, // Remove default margin
                }}>
                    Your Dream Job Is Waiting
                </h1>
                <p style={{
                    fontSize: '1.2em',
                    color: '#333',
                    marginTop: '15px',
                    fontWeight: '500',
                }}>
                    Find the perfect opportunity or the ideal candidate.
                </p>
            </div>

            {/* Role Selection Buttons */}
            <div style={{
                position: 'relative',
                zIndex: 2,
                display: 'flex',
                gap: '25px', // Increased gap between buttons
                marginBottom: '40px', // Space below role selection
                flexWrap: 'wrap', // Allow buttons to wrap on smaller screens
                justifyContent: 'center',
            }}>
                <button
                    style={{
                        padding: '18px 35px', // Larger padding
                        backgroundColor: findjob ? '#FFCFB6' : 'white', // Highlight selected
                        color: '#333',
                        border: findjob ? '2px solid #FFCFB6' : '2px solid #4BAAC8', // Thicker border
                        borderRadius: '12px', // More rounded corners
                        fontSize: '1.2em', // Larger text
                        fontWeight: '600',
                        cursor: 'pointer',
                        boxShadow: '0 4px 10px rgba(0,0,0,0.1)', // Subtle shadow
                        transition: 'all 0.3s ease', // Smooth transition for hover/active states
                        minWidth: '220px', // Minimum width for buttons
                    }}
                    onMouseEnter={(e) => {
                        if (!findjob) e.target.style.transform = 'translateY(-6px)';
                        e.target.style.boxShadow = '0 6px 15px rgba(0,0,0,0.6)';
                    }}
                    onMouseLeave={(e) => {
                        if (!findjob) e.target.style.transform = 'translateY(0)';
                        e.target.style.boxShadow = '0 4px 10px rgba(0,0,0,0.1)';
                    }}
                    onClick={() => {
                        setfindjob(true);
                        setfindcandidate(false);
                        seterrmsg(''); // Clear error when switching tabs
                        setfdata({ email: '', pw: '' }); // Clear form data
                    }}
                >
                    I'm Looking for a Job
                </button>

                <button
                    style={{
                        padding: '18px 35px',
                        backgroundColor: findcandidate ? '#FFCFB6' : 'white',
                        color: '#333',
                        border: findcandidate ? '2px solid #FFCFB6' : '2px solid #4BAAC8',
                        borderRadius: '12px',
                        fontSize: '1.2em',
                        fontWeight: '600',
                        cursor: 'pointer',
                        boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
                        transition: 'all 0.3s ease',
                        minWidth: '220px',
                    }}
                    onMouseEnter={(e) => {
                        if (!findcandidate) e.target.style.transform = 'translateY(-6px)';
                        e.target.style.boxShadow = '0 6px 15px rgba(0,0,0,0.6)';
                    }}
                    onMouseLeave={(e) => {
                        if (!findcandidate) e.target.style.transform = 'translateY(0)';
                        e.target.style.boxShadow = '0 4px 10px rgba(0,0,0,0.1)';
                    }}
                    onClick={() => {
                        setfindcandidate(true);
                        setfindjob(false);
                        seterrmsg(''); // Clear error when switching tabs
                        setfdata({ email: '', pw: '' }); // Clear form data
                    }}
                >
                    I'm Looking for a Candidate
                </button>
            </div>

            {errmsg && (
                <p style={{
                    position: 'relative',
                    zIndex: 2,
                    color: 'red',
                    fontWeight: 'bold',
                    fontSize: '1em',
                    marginBottom: '20px',
                    textAlign: 'center',
                }}>
                    {errmsg}
                </p>
            )}

            {/* Employee Login Form */}
            {findjob && (
                <div style={{
                    position: 'relative',
                    zIndex: 2,
                    backgroundColor: 'white',
                    padding: '40px',
                    borderRadius: '15px',
                    boxShadow: '0 8px 20px rgba(0,0,0,0.15)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '20px', // Space between inputs
                    alignItems: 'center',
                    width: '90%', // Responsive width
                    maxWidth: '450px', // Max width for larger screens
                }}>
                    <h2 style={{
                        color: '#0476D0',
                        fontSize: '2em',
                        marginBottom: '10px',
                        fontWeight: '700',
                    }}>Employee Login</h2>
                    <input
                        type="email"
                        name='email'
                        onChange={handleInputChange}
                        value={fdata.email} // Controlled input
                        placeholder='Enter your Email-ID'
                        style={{
                            height: '50px',
                            width: '100%',
                            padding: '0 15px',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            fontSize: '1em',
                            boxSizing: 'border-box',
                            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                        }}
                        onFocus={(e) => e.target.style.borderColor = '#0476D0'}
                        onBlur={(e) => e.target.style.borderColor = '#ccc'}
                    />
                    <input
                        type="password" // Use type="password" for security
                        name='pw'
                        onChange={handleInputChange}
                        value={fdata.pw} // Controlled input
                        placeholder='Enter your password'
                        style={{
                            height: '50px',
                            width: '100%',
                            padding: '0 15px',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            fontSize: '1em',
                            boxSizing: 'border-box',
                            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                        }}
                        onFocus={(e) => e.target.style.borderColor = '#0476D0'}
                        onBlur={(e) => e.target.style.borderColor = '#ccc'}
                    />
                    <button
                        onClick={GetDataFromBackendForEmployee}
                        style={{
                            height: '50px',
                            width: '60%', // Adjust button width
                            backgroundColor: '#0476D0', // Blue button
                            color: 'white',
                            border: 'none',
                            borderRadius: '10px',
                            fontSize: '1.1em',
                            fontWeight: 'bold',
                            cursor: 'pointer',
                            transition: 'background-color 0.3s ease, transform 0.2s ease',
                            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                        }}
                        onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'}
                        onMouseLeave={(e) => e.target.style.backgroundColor = '#0476D0'}
                    >
                        Login as Employee
                    </button>
                    <p style={{
                        fontSize: '0.95em',
                        color: '#555',
                        marginTop: '10px', // Space above signup link
                    }}>
                        Don't have an account? {' '}
                        <a
                            href='applicantsignup'
                            style={{
                                color: '#0476D0',
                                textDecoration: 'none',
                                fontWeight: 'bold',
                                transition: 'color 0.3s ease',
                            }}
                            onMouseEnter={(e) => e.target.style.color = '#035391'}
                            onMouseLeave={(e) => e.target.style.color = '#0476D0'}
                        >
                            Click here to Sign-up
                        </a>
                    </p>
                </div>
            )}

            {/* Employer Login Form */}
            {findcandidate && (
                <div style={{
                    position: 'relative',
                    zIndex: 2,
                    backgroundColor: 'white',
                    padding: '40px',
                    borderRadius: '15px',
                    boxShadow: '0 8px 20px rgba(0,0,0,0.15)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '20px',
                    alignItems: 'center',
                    width: '90%',
                    maxWidth: '450px',
                }}>
                    <h2 style={{
                        color: '#0476D0',
                        fontSize: '2em',
                        marginBottom: '10px',
                        fontWeight: '700',
                    }}>Employer Login</h2>
                    <input
                        type="email"
                        name='email'
                        onChange={handleInputChange}
                        value={fdata.email} // Controlled input
                        placeholder='Enter your Email-ID'
                        style={{
                            height: '50px',
                            width: '100%',
                            padding: '0 15px',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            fontSize: '1em',
                            boxSizing: 'border-box',
                            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                        }}
                        onFocus={(e) => e.target.style.borderColor = '#0476D0'}
                        onBlur={(e) => e.target.style.borderColor = '#ccc'}
                    />
                    <input
                        type="password" // Use type="password" for security
                        name='pw'
                        onChange={handleInputChange}
                        value={fdata.pw} // Controlled input
                        placeholder='Enter your password'
                        style={{
                            height: '50px',
                            width: '100%',
                            padding: '0 15px',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            fontSize: '1em',
                            boxSizing: 'border-box',
                            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                        }}
                        onFocus={(e) => e.target.style.borderColor = '#0476D0'}
                        onBlur={(e) => e.target.style.borderColor = '#ccc'}
                    />
                    <button
                        onClick={GetDataFromBackendForEmployer}
                        style={{
                            height: '50px',
                            width: '60%',
                            backgroundColor: '#0476D0',
                            color: 'white',
                            border: 'none',
                            borderRadius: '10px',
                            fontSize: '1.1em',
                            fontWeight: 'bold',
                            cursor: 'pointer',
                            transition: 'background-color 0.3s ease, transform 0.2s ease',
                            boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                        }}
                        onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'}
                        onMouseLeave={(e) => e.target.style.backgroundColor = '#0476D0'}
                    >
                        Login as Employer
                    </button>
                    <p style={{
                        fontSize: '0.95em',
                        color: '#555',
                        marginTop: '10px',
                    }}>
                        Don't have an account? {' '}
                        <a
                            href='employersignup'
                            style={{
                                color: '#0476D0',
                                textDecoration: 'none',
                                fontWeight: 'bold',
                                transition: 'color 0.3s ease',
                            }}
                            onMouseEnter={(e) => e.target.style.color = '#035391'}
                            onMouseLeave={(e) => e.target.style.color = '#0476D0'}
                        >
                            Click here to Sign-up
                        </a>
                    </p>
                </div>
            )}
        </div>
    );
}