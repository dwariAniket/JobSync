import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function ViewApplicants() {
  const [email, setEmail] = useState("");
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [applicants, setApplicants] = useState([]);
  const [error, setError] = useState("");
  const [selectedApplicantForScore, setSelectedApplicantForScore] = useState(null);
  const [loadingScore, setLoadingScore] = useState(false);
  const [matchScore, setMatchScore] = useState(null);
  const navigate = useNavigate();


  useEffect(() => {
    const employerData = JSON.parse(localStorage.getItem("employerdata"));
    if (employerData && employerData.savedUser && employerData.savedUser.email) {
      setEmail(employerData.savedUser.email);
    } else {
      setError("Please log in as an employer to view your jobs.");
    }
  }, []);


  useEffect(() => {
    if (email) {
      fetchJobs();
    }
  }, [email]);

  const fetchJobs = async () => {
    try {
      setError("");
      const response = await fetch(
        `http://192.168.29.18:4000/employer/jobs?email=${encodeURIComponent(email)}`
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.message || "Failed to fetch jobs.");
      }
      setJobs(data.jobs || []);
      setSelectedJobId(null);
      setApplicants([]);
    } catch (err) {
      setError(err.message);
    }
  };


  const fetchApplicants = async (jobId) => {
    try {
      setError("");
      const response = await fetch(`http://192.168.29.18:4000/employer/job/${jobId}/applicants`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to load applicants.");
      }
      setSelectedJobId(jobId);
      setApplicants(data.applicants || []);
    } catch (err) {
      setError(err.message);
    }
  };


  const updateApplicantStatus = async (applicantEmail, status) => {

    if (!selectedJobId) {
      setError("No job selected. Please select a job first.");
      return;
    }

    try {
      setError("");
      const response = await fetch(`http://192.168.29.18:4000/employer/updateapplicantstatus`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },

        body: JSON.stringify({
          applicantEmail: applicantEmail,
          status: status,
          jobId: selectedJobId
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || data.error || 'Failed to update status');
      }

      alert(`Applicant status updated to ${status} successfully!`);

      fetchApplicants(selectedJobId);
    } catch (err) {
      console.error("Error updating applicant status:", err);
      setError(err.message);
    }
  };


  const fetchMatchScore = async (jobId, employeeID) => {
    try {
      setLoadingScore(true);
      setError("");
      const response = await fetch(`http://192.168.29.18:4000/job/${jobId}/${employeeID}/skills-jobdes-python`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || data.error || "Failed to fetch score");
      }
      setMatchScore(data);
    } catch (err) {
      console.error("Error fetching score", err);
      setError(err.message);
      setMatchScore(null);
    } finally {
      setLoadingScore(false);
    }
  };


  const handleViewScore = (jobId, employeeID, EmployeeEmail) => {
    if (selectedApplicantForScore === EmployeeEmail) {
      setSelectedApplicantForScore(null);
      setMatchScore(null);
    } else {
      setSelectedApplicantForScore(EmployeeEmail);
      fetchMatchScore(jobId, employeeID);
    }
  };


  return (
    <div style={{
      minHeight: '100vh',
      width: '100%',
      backgroundColor: '#F0F8FF',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      paddingBottom: '50px',
    }}>
      {/* Header */}
      <div style={{
        height: '60px',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 20px',
        boxSizing: 'border-box',
      }}>
        <button
          style={{
            padding: '10px 20px',
            backgroundColor: '#0476D0',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '15px',
            fontWeight: 'bold',
            transition: 'background-color 0.3s ease',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#0476D0'}
          onClick={() => navigate('/jobdetails')}
        >
          Post New Job
        </button>
      </div>
      <h1 style={{
        color: '#0476D0',
        marginTop: '40px',
        marginBottom: '30px',
        fontSize: '28px',
        fontWeight: '800',
        textAlign: 'center',
      }}>View Job Applicants</h1>

      <div style={{
        width: '90%',
        maxWidth: '700px',
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '30px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        alignItems: 'center',
        marginBottom: '30px',
      }}>
        <label htmlFor="employerEmail" style={{
          fontSize: '18px',
          fontWeight: '600',
          color: '#333',
          width: '100%',
          textAlign: 'center',
        }}>Enter Your Employer Email to Fetch Jobs</label>
        <div style={{
          display: 'flex',
          width: '100%',
          gap: '15px',
          justifyContent: 'center',
        }}>
          <input
            id="employerEmail"
            type="email"
            placeholder="your-email@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{
              padding: "12px 15px",
              height: 'auto',
              flexGrow: 1,
              borderRadius: '8px',
              border: '1px solid #ccc',
              fontSize: '16px',
              boxSizing: 'border-box',
              maxWidth: '400px',
            }}
          />
          <button
            style={{
              padding: '12px 25px',
              backgroundColor: '#0476D0',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              transition: 'background-color 0.3s ease',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#0476D0'}
            onClick={fetchJobs}
          >
            Fetch My Jobs
          </button>
        </div>
      </div>

      {error && (
        <p style={{
          color: "red",
          fontWeight: 'bold',
          fontSize: '16px',
          marginBottom: '20px',
          textAlign: 'center',
        }}>
          {error}
        </p>
      )}

      <div style={{
        width: '90%',
        maxWidth: '900px',
        padding: '0 20px',
        marginBottom: '40px',
        boxSizing: 'border-box',
      }}>
        <h2 style={{
          color: '#0476D0',
          fontSize: '22px',
          marginBottom: '20px',
          borderBottom: '2px solid #e0e0e0',
          paddingBottom: '10px',
          fontWeight: '700',
        }}>Your Posted Jobs</h2>
        {jobs.length === 0 ? (
          <p style={{
            textAlign: 'center',
            color: '#555',
            fontSize: '16px',
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
          }}>No jobs found for this email. Please ensure the email is correct and you have posted jobs.</p>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '25px',
          }}>
            {jobs.map((job) => (
              <div
                key={job._id}
                style={{
                  backgroundColor: 'white',
                  padding: "25px",
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                  borderRadius: '10px',
                  border: `1px solid ${selectedJobId === job._id ? '#0476D0' : '#ddd'}`,
                  boxShadow: selectedJobId === job._id ? '0 4px 10px rgba(4,118,208,0.2)' : '0 2px 8px rgba(0,0,0,0.08)',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer',

                  overflowWrap: 'break-word',
                  wordBreak: 'break-word',
                }}
                onClick={() => fetchApplicants(job._id)}
              >
                <p style={{ margin: 0, fontSize: '18px', color: '#333' }}><b>Job Title:</b> {job.jobtitle}</p>
                <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>Location:</b> {job.joblocation}</p>
                <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>Openings:</b> {job.openings}</p>
                <button
                  style={{
                    marginTop: '15px',
                    padding: '10px 15px',
                    backgroundColor: selectedJobId === job._id ? '#035391' : '#0476D0',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    transition: 'background-color 0.3s ease',
                    alignSelf: 'flex-start',
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = '#035391'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = selectedJobId === job._id ? '#035391' : '#0476D0'}
                  onClick={(e) => { e.stopPropagation(); fetchApplicants(job._id); }}
                >
                  {selectedJobId === job._id ? 'Applicants Loaded' : 'View Applicants'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedJobId && (
        <div style={{
          width: '90%',
          maxWidth: '900px',
          padding: '0 20px',
          marginTop: '40px',
          boxSizing: 'border-box',
        }}>
          <h2 style={{
            color: '#0476D0',
            fontSize: '22px',
            marginBottom: '20px',
            borderBottom: '2px solid #e0e0e0',
            paddingBottom: '10px',
            fontWeight: '700',
          }}>Applicants for Selected Job</h2>
          {applicants.length === 0 ? (
            <p style={{
              textAlign: 'center',
              color: '#555',
              fontSize: '16px',
              padding: '20px',
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 2px 5px rgba(0,0,0,0.05)',
            }}>No applicants have applied for this job yet.</p>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '25px',
            }}>
              {applicants.map((app, index) => (
                <div
                  key={index}
                  style={{
                    background: "white",
                    padding: "25px",
                    borderRadius: '10px',
                    border: "1px solid #ddd",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px',
                    justifyContent: 'space-between',
                    overflowWrap: 'break-word',
                    wordBreak: 'break-word',
                  }}
                >
                  <p style={{ margin: 0, fontSize: '18px', color: '#333' }}><b>Name:</b> {app.EmployeefName} {app.EmployeelName}</p>
                  <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>Email:</b> {app.EmployeeEmail}</p>
                  <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>Education:</b> {app.EmployeeEducation}</p>
                  <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>Skills:</b> {app.EmployeeSkills}</p>
                  <p style={{ margin: 0, fontSize: '16px', color: '#555' }}><b>College:</b> {app.EmployeeCollegename}</p>
                  {app.EmployeeCV && (
                    <a
                      href={app.EmployeeCV}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        color: '#0066cc',
                        textDecoration: 'underline',
                        fontSize: '15px',
                        marginTop: '5px',
                        marginBottom: '10px',
                        fontWeight: '600',
                      }}
                    >
                      View CV
                    </a>
                  )}
                  <div style={{
                    // backgroundColor:'green',
                    display: 'flex',
                    flexDirection: 'row',
                    position: 'relative'
                  }}>
                    <p style={{ margin: 0, fontSize: '16px', color: '#555', fontWeight: 'bold' }}>
                      Status:{" "}
                      <span style={{
                        color:
                          app.EmployeeStatus === 'Accepted' ? '#28a745' :
                            app.EmployeeStatus === 'Rejected' ? '#dc3545' :
                              '#ffc107'
                      }}>
                        {app.EmployeeStatus || 'Pending'}
                      </span>
                    </p>
                    <button style={{
                      backgroundColor: 'blue',
                      color: 'white',
                      position: 'absolute',
                      right: '2px'
                    }}
                      onClick={() => handleViewScore(selectedJobId, app._id, app.EmployeeEmail)}
                    >{selectedApplicantForScore === app.EmployeeEmail ? "Hide score" : "View score"}</button>

                    {selectedApplicantForScore === app.EmployeeEmail && (
                      <div style={{
                        backgroundColor: 'rgba(0, 128, 0, 0.9)',
                        color: 'white',
                        padding: '20px',
                        borderRadius: '10px',
                        position: 'fixed',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        width: '300px',
                        maxWidth: '90%',
                        zIndex: 1000,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                        textAlign: 'center'
                      }}>
                        <h3>Match Score for {app.EmployeefName}</h3>
                        {loadingScore ? (
                          <p>Loading match score...</p>
                        ) : matchScore ? (
                          <div>
                            <p style={{ fontSize: '16px', margin: '8px 0' }}>
                              <strong>Prediction:</strong> {matchScore.prediction ? (Number(matchScore.prediction).toFixed(2)) : 'N/A'}
                            </p>
                          </div>
                        ) : (
                          <p style={{ color: '#dc3545' }}>No score data available</p>
                        )}
                        <button
                          onClick={() => setSelectedApplicantForScore(null)}
                          style={{
                            backgroundColor: 'white',
                            color: 'green',
                            border: 'none',
                            padding: '8px 15px',
                            borderRadius: '5px',
                            cursor: 'pointer',
                            marginTop: '10px'
                          }}
                        >
                          Close
                        </button>
                      </div>
                    )}

                  </div>
                  <div style={{
                    display: 'flex',
                    flexDirection: 'row',
                    justifyContent: 'space-around',
                    marginTop: '15px',
                    gap: '10px',
                  }}>
                    <button
                      style={{
                        padding: '10px 15px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '20px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: 'bold',
                        transition: 'background-color 0.3s ease',
                        flex: 1,

                        opacity: app.EmployeeStatus === 'Accepted' ? 0.6 : 1,
                        cursor: app.EmployeeStatus === 'Accepted' ? 'not-allowed' : 'pointer',
                      }}
                      onMouseEnter={(e) => e.target.style.backgroundColor = '#218838'}
                      onMouseLeave={(e) => e.target.style.backgroundColor = '#28a745'}
                      onClick={() => updateApplicantStatus(app.EmployeeEmail, "Accepted")}
                      disabled={app.EmployeeStatus === 'Accepted'}
                    >
                      Accept
                    </button>
                    <button
                      style={{
                        padding: '10px 15px',
                        backgroundColor: '#dc3545',
                        color: 'white',
                        border: 'none',
                        borderRadius: '20px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: 'bold',
                        transition: 'background-color 0.3s ease',
                        flex: 1,
                        opacity: app.EmployeeStatus === 'Rejected' ? 0.6 : 1,
                        cursor: app.EmployeeStatus === 'Rejected' ? 'not-allowed' : 'pointer',
                      }}
                      onMouseEnter={(e) => e.target.style.backgroundColor = '#c82333'}
                      onMouseLeave={(e) => e.target.style.backgroundColor = '#dc3545'}
                      onClick={() => updateApplicantStatus(app.EmployeeEmail, "Rejected")}
                      disabled={app.EmployeeStatus === 'Rejected'}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}