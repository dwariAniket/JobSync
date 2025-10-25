import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import myImg from '../Screenshot 2025-04-01 202950.jpg'; // Assuming this path is correct

export default function ApplicantHome() {
    const [applicantData, setApplicantData] = useState({ employeedata: null });
    const [showExitPopup, setShowExitPopup] = useState(false);
    const [appliedJobs, setAppliedJobs] = useState([]);
    const [showEmployeeStatusModal, setShowEmployeeStatusModal] = useState(false);
    const [notification, setNotification] = useState(null);
    const [showCountOnButton, setShowCountOnButton] = useState(true); // NEW: State to control count visibility
    const navigate = useNavigate();

    const prevAppliedJobsRef = useRef([]);

    const handleExitClick = () => {
        setShowExitPopup(true);
    };

    const handleConfirmExit = () => {
        setShowExitPopup(false);
        console.log("User chose YES");
        localStorage.removeItem("employeedata");
        localStorage.removeItem("employeejobdata");
        navigate('/');
    };

    const handleCancelExit = () => {
        setShowExitPopup(false);
        console.log("User chose NO");
    };

    const fetchApplicantPersonalDetails = async () => {
        try {
            const employeedata = JSON.parse(localStorage.getItem("employeedata")) || null;
            if (!employeedata || !employeedata.savedUser?.email) return;

            const response = await fetch(
                `http://192.168.29.18:4000/employee/employeedetails/${employeedata.savedUser.email}`
            );
            const data = await response.json();

            if (data.details && data.details[0]) {
                localStorage.setItem("employeejobdata", JSON.stringify(data.details[0]));
                setApplicantData({ employeedata: data.details[0] });
            } else {
                console.warn("No full applicant details found.");
                setApplicantData({ employeedata: employeedata.savedUser });
            }
        } catch (error) {
            console.error("Error fetching employee personal details:", error);
        }
    };

    const fetchAppliedJobs = async (notify = false) => {
        try {
            const employeedata = JSON.parse(localStorage.getItem("employeedata")) || null;
            if (!employeedata || !employeedata.savedUser?.email) {
                setAppliedJobs([]);
                return;
            }

            const response = await fetch(
                `http://192.168.29.18:4000/student/applications?email=${employeedata.savedUser.email}`
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to fetch applications');
            }

            const data = await response.json();
            const fetchedApplications = data.applications || [];
            console.log("Fetched Applied Jobs:", fetchedApplications);

            if (notify && prevAppliedJobsRef.current) {
                const previouslyAppliedJobs = prevAppliedJobsRef.current;
                fetchedApplications.forEach(newJob => {
                    const oldJob = previouslyAppliedJobs.find(old => old.jobId === newJob.jobId);
                    if (oldJob && oldJob.EmployeeStatus !== 'Accepted' && newJob.EmployeeStatus === 'Accepted') {
                        setNotification({
                            message: `Congratulations! Your application for "${newJob.jobTitle}" at "${newJob.companyName}" has been Accepted!`,
                            type: 'success'
                        });
                        setTimeout(() => setNotification(null), 8000);
                    }
                });
            }

            setAppliedJobs(fetchedApplications);
            prevAppliedJobsRef.current = fetchedApplications; // Update the ref

        } catch (error) {
            console.error("Error fetching applied jobs:", error);
            setAppliedJobs([]);
            if (notify) {
                setNotification({
                    message: "Failed to fetch application status.",
                    type: 'error'
                });
                setTimeout(() => setNotification(null), 5000);
            }
        }
    };

    // Initial fetch on component mount
    useEffect(() => {
        fetchApplicantPersonalDetails();
        // Fetch applied jobs on initial load to populate the count
        fetchAppliedJobs(true);
    }, []);

    // Function to handle the "Check Application Status" button click
    const handleCheckStatusClick = async () => {
        await fetchAppliedJobs(true); // Re-fetch and check for status changes
        setShowEmployeeStatusModal(true);
        // We will keep the count visible until the modal is explicitly closed.
        // If you want it to disappear *immediately* on click, you'd set setShowCountOnButton(false) here.
    };

    // Function to close the modal and remove the count
    const handleCloseStatusModal = () => {
        setShowEmployeeStatusModal(false);
        setShowCountOnButton(false); // NEW: Hide the count when modal is closed
    };

    return (
        <div style={{
            minHeight: '100vh',
            backgroundColor: '#f0f8ff',
            paddingBottom: '50px'
        }}>
            {/* Notification Bar */}
            {notification && (
                <div style={{
                    position: 'fixed',
                    top: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    backgroundColor: notification.type === 'success' ? '#4CAF50' : '#f44336',
                    color: 'white',
                    padding: '12px 25px',
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                    zIndex: 1001,
                    fontSize: '16px',
                    fontWeight: 'bold',
                    opacity: 0.95,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px'
                }}>
                    {notification.type === 'success' ? '🎉' : '⚠️'} {notification.message}
                    <button
                        onClick={() => setNotification(null)}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: 'white',
                            fontSize: '20px',
                            cursor: 'pointer',
                            marginLeft: '15px'
                        }}
                    >
                        &times;
                    </button>
                </div>
            )}

            {/* Header/Navigation */}
            <div style={{
                height: '70px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                padding: '0 25px',
                position: 'relative'
            }}>
                <button
                    onClick={() => navigate('/applicantjobsearch')}
                    style={{
                        marginRight: '15px',
                        padding: '8px 16px',
                        borderRadius: '5px',
                        backgroundColor: '#0476D0',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '14px'
                    }}
                >
                    Opportunities
                </button>

                {/* UPDATED: Conditional rendering of count */}
                <button
                    onClick={handleCheckStatusClick}
                    style={{
                        marginRight: '15px',
                        padding: '8px 16px',
                        borderRadius: '5px',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '14px'
                    }}
                >
                    Application Status {showCountOnButton && appliedJobs.length > 0 ? `(${appliedJobs.length})` : ''}
                </button>

                <button
                    onClick={handleExitClick}
                    style={{
                        padding: '8px 16px',
                        borderRadius: '5px',
                        backgroundColor: '#F44336',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '14px'
                    }}
                >
                    Log out
                </button>
            </div>

            {/* Main Content (rest of your home page - unchanged) */}
            <div style={{
                maxWidth: '1200px',
                margin: '20px auto',
                padding: '0 20px'
            }}>
                {/* Profile Section */}
                <div style={{
                    backgroundColor: 'white',
                    borderRadius: '10px',
                    padding: '25px',
                    marginBottom: '20px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '20px'
                }}>
                    {/* Profile Info */}
                    <div style={{ flex: '1 1 300px' }}>
                        <h1 style={{ color: '#0476D0', fontSize: '40px', marginBottom: '10px' }}>
                            <i>Welcome!</i>
                        </h1>
                        <h2 style={{ fontSize: '30px', marginBottom: '15px' }}>
                            {applicantData.employeedata ? `${applicantData.employeedata.EmployeefName} ${applicantData.employeedata.EmployeelName}` : "Applicant"}
                        </h2>
                        <p style={{ fontSize: '18px', marginBottom: '5px' }}>
                            📧 {applicantData.employeedata?.EmployeeEmail || "Not Provided"}
                        </p>
                    </div>

                    {/* Education Info */}
                    <div style={{ flex: '1 1 300px', paddingLeft: '20px', borderLeft: '1px solid #eee' }}>
                        <h3 style={{ marginBottom: '15px' }}>Education Details</h3>
                        <p><b>Last Education:</b> {applicantData.employeedata?.education || "Not Provided"}</p>
                        <p><b>School Medium:</b> {applicantData.employeedata?.schoolmed || "Not Provided"}</p>
                        <p><b>College:</b> {applicantData.employeedata?.collegename || "Not Provided"}</p>
                        <p><b>Completion Year:</b> {applicantData.employeedata?.completionyear || "Not Provided"}</p>
                    </div>

                    {/* Skills Info */}
                    <div style={{ flex: '1 1 300px', paddingLeft: '20px', borderLeft: '1px solid #eee' }}>
                        <h3 style={{ marginBottom: '15px' }}>Professional Details</h3>
                        <p><b>Department:</b> {applicantData.employeedata?.specialization || "Not Provided"}</p>
                        <p><b>Skills:</b> {applicantData.employeedata?.skills || "Not Provided"}</p>
                    </div>

                    {/* Update Profile */}
                    <div style={{
                        flex: '1 1 300px',
                        padding: '15px',
                        backgroundColor: '#f9f9f9',
                        borderRadius: '8px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        <p style={{
                            backgroundColor: '#FFDEAD',
                            padding: '10px',
                            borderRadius: '5px',
                            marginBottom: '15px',
                            textAlign: 'center'
                        }}>
                            To increase your chances of getting recruited, ensure your profile is up-to-date.
                        </p>
                        <button
                            onClick={() => navigate('/applicantdetails')}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: '#0476D0',
                                color: 'white',
                                border: 'none',
                                borderRadius: '5px',
                                cursor: 'pointer',
                                width: '100%',
                                fontSize: '16px'
                            }}
                        >
                            Update Your Details
                        </button>
                    </div>
                </div>

                {/* Dashboard Cards (existing content) */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                    gap: '20px'
                }}>
                    {/* Activity Tracking Card */}
                    <div style={{
                        backgroundColor: 'white',
                        borderRadius: '10px',
                        padding: '20px',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            marginBottom: '15px'
                        }}>
                            <img
                                src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQD_6Y7kf39Bh1PEzI75cwR0CsGIbuOFKB4wQ&s'
                                style={{
                                    width: '60px',
                                    height: '60px',
                                    marginRight: '15px'
                                }}
                                alt="Activity tracking"
                            />
                            <h3 style={{ fontSize: '20px' }}>Keep track of your activities</h3>
                        </div>
                        <ul style={{ listStyleType: 'none', paddingLeft: '0' }}>
                            <li style={{ marginBottom: '8px' }}>📈 Monitor your progress</li>
                            <li style={{ marginBottom: '8px' }}>💡 Get personalized insights</li>
                        </ul>
                    </div>

                    {/* Aptitude Test Card */}
                    <div style={{
                        backgroundColor: 'white',
                        borderRadius: '10px',
                        padding: '20px',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <h3 style={{ fontSize: '20px', marginBottom: '15px' }}>
                            80% companies hire using an aptitude test!
                        </h3>
                        <ul style={{ listStyleType: 'none', paddingLeft: '0' }}>
                            <li style={{ marginBottom: '8px' }}>✔ Test your skills early</li>
                            <li style={{ marginBottom: '8px' }}>✔ Assess your strengths</li>
                            <li style={{ marginBottom: '8px' }}>✔ Understand areas of improvement</li>
                        </ul>
                    </div>

                    {/* Image Card */}
                    <div style={{
                        backgroundColor: 'white',
                        borderRadius: '10px',
                        padding: '20px',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center'
                    }}>
                        <img
                            src={myImg}
                            style={{
                                maxWidth: '100%',
                                height: 'auto',
                                borderRadius: '5px'
                            }}
                            alt="Career guidance"
                        />
                    </div>
                </div>
            </div>

            {/* Exit Confirmation Popup (unchanged) */}
            {showExitPopup && (
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
                        borderRadius: '10px',
                        padding: '25px',
                        width: '350px',
                        textAlign: 'center'
                    }}>
                        <p style={{
                            fontWeight: 'bold',
                            fontSize: '18px',
                            marginBottom: '20px'
                        }}>
                            Do you want to sign out?
                        </p>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'center',
                            gap: '15px'
                        }}>
                            <button
                                onClick={handleConfirmExit}
                                style={{
                                    padding: '8px 20px',
                                    backgroundColor: '#4CAF50',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Yes
                            </button>
                            <button
                                onClick={handleCancelExit}
                                style={{
                                    padding: '8px 20px',
                                    backgroundColor: '#f44336',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                No
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* NEW: Application Status Modal */}
            {showEmployeeStatusModal && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.4)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 1000
                }} onClick={handleCloseStatusModal}> {/* Changed to handleCloseStatusModal */}
                    <div style={{
                        backgroundColor: 'white',
                        borderRadius: '10px',
                        padding: '25px',
                        width: '80%',
                        maxWidth: '900px',
                        maxHeight: '80%',
                        overflowY: 'auto',
                        boxShadow: '0 5px 15px rgba(0,0,0,0.3)',
                        position: 'relative'
                    }} onClick={e => e.stopPropagation()}>
                        <button
                            onClick={handleCloseStatusModal} // Changed to handleCloseStatusModal
                            style={{
                                position: 'absolute',
                                top: '15px',
                                right: '15px',
                                background: 'none',
                                border: 'none',
                                fontSize: '24px',
                                cursor: 'pointer',
                                color: '#333'
                            }}
                        >
                            &times;
                        </button>
                        <h3 style={{ color: '#0476D0', fontSize: '28px', marginBottom: '25px', textAlign: 'center' }}>
                            Your Job Application Status
                        </h3>

                        {appliedJobs.length === 0 ? (
                            <p style={{ fontSize: '20px', textAlign: 'center', color: '#555', padding: '30px' }}>
                                You have not applied for any jobs yet. Start exploring opportunities!
                            </p>
                        ) : (
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                                gap: '20px'
                            }}>
                                {appliedJobs.map((job) => (
                                    <div key={job.jobId} style={{
                                        border: '1px solid #eee',
                                        borderRadius: '8px',
                                        padding: '15px',
                                        boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
                                        backgroundColor: '#fdfdfd',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        justifyContent: 'space-between'
                                    }}>
                                        <div>
                                            <h4 style={{ margin: '0 0 10px', color: '#333', fontSize: '1.1em' }}>{job.jobTitle}</h4>
                                            <p style={{ margin: '5px 0', fontSize: '0.9em' }}>
                                                <strong>Company:</strong> {job.companyName}
                                            </p>
                                            <p style={{ margin: '5px 0', fontSize: '0.9em' }}>
                                                <strong>Location:</strong> {job.jobLocation}
                                            </p>
                                        </div>
                                        <p style={{ margin: '15px 0 0', fontSize: '1em', fontWeight: 'bold' }}>
                                            Status:{" "}
                                            <span style={{
                                                color: job.EmployeeStatus === 'Accepted' ? '#28a745' :
                                                    job.EmployeeStatus === 'Rejected' ? '#dc3545' :
                                                        job.EmployeeStatus === 'Pending' ? '#ffc107' :
                                                            '#007bff'
                                            }}>
                                                {job.EmployeeStatus}
                                            </span>
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}