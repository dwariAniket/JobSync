import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function ApplicantJobSearch() {
  const [joblocation, setjoblocation] = useState("");
  const [jobtitle, setjobtitle] = useState("");
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [applyMessage, setApplyMessage] = useState("");
  const [appliedJobs, setAppliedJobs] = useState([]);
  const [applicantDetails, setApplicantDetails] = useState(null);
  const [seedetails, setseedetails] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [loadingAppliedJobs, setLoadingAppliedJobs] = useState(true);
  const navigate = useNavigate();

  const fetchJobs = async () => {
    setLoading(true);
    setError("");
    setApplyMessage("");
    try {
      const response = await fetch(
        `http://192.168.29.18:4000/employee/job-search?joblocation=${encodeURIComponent(joblocation)}&jobtitle=${encodeURIComponent(jobtitle)}`
      );
      const data = await response.json();
      if (!data.jobs || data.jobs.length === 0) {
        setError("No jobs found for the given search.");
        setJobs([]);
      } else {
        setJobs(data.jobs);
      }
    } catch (error) {
      console.error("Error fetching jobs:", error);
      setError(error.message);
    }
    setLoading(false);
  };

  useEffect(() => {
    console.log("applyMessage updated:", applyMessage);
  }, [applyMessage]);


  useEffect(() => {
    try {
      const savedAppliedJobs = localStorage.getItem("appliedJobs")
      if (savedAppliedJobs) {
        setAppliedJobs(JSON.parse(savedAppliedJobs));
      }
    } catch (err) {
      console.err("Error loading applied jobs ", err);
      setAppliedJobs([]);
    }
  }, []);


  useEffect(() => {
    try {
      localStorage.setItem("appliedJobs", JSON.stringify(appliedJobs));
    } catch (err) {
      console.err("Error saving applied jobs ", err);
    }
  }, [appliedJobs]);


  useEffect(() => {
    const fetchAppliedJobsFromBackend = async () => {
      try {
        setLoadingAppliedJobs(true);
        const employeeData = JSON.parse(localStorage.getItem("employeejobdata"));
        if (employeeData && employeeData.EmployeeEmail) {
          const response = await fetch(`http://192.168.29.18:4000/employee/applied-jobs?email=${encodeURIComponent(employeeData.EmployeeEmail)}`);
          if (response.ok) {
            const data = await response.json();
            const backendAppliedJobs = data.appliedJobs || [];
            setAppliedJobs(backendAppliedJobs);
            localStorage.setItem("appliedjobs", JSON.stringify(backendAppliedJobs))
          } else {
            throw new error("Failed to fetch applied jobs")
          }
        }
      } catch (err) {
        console.error("Error fetching applied jobs from backend: ", err);
      }
    };
    fetchAppliedJobsFromBackend();
  }, []);


  const handleApplyJob = async (jobId) => {
    const EmployeeData = JSON.parse(localStorage.getItem("employeejobdata"));
    console.log(EmployeeData)
    const ApplicantData = {
      jobId,
      EmployeefName: EmployeeData.EmployeefName,
      EmployeelName: EmployeeData.EmployeelName,
      EmployeeEmail: EmployeeData.EmployeeEmail,
      EmployeeID: EmployeeData.EmployeeID,
      EmployeeEducation: EmployeeData.education,
      EmployeeSchoolmed: EmployeeData.schoolmed,
      EmployeeCollegename: EmployeeData.collegename,
      EmployeeSpecialization: EmployeeData.specialization,
      EmployeeCompletionyear: EmployeeData.completionyear,
      EmployeeSkills: EmployeeData.skills,
      EmployeeCV: EmployeeData.cvURL,
    }

    fetch('http://192.168.29.18:4000/employee/apply', {
      method: 'POST',
      headers: {
        'Content-type': 'application/json',
      },
      body: JSON.stringify(ApplicantData),
    })
      .then((res) => {
        console.log("Response received:", res);
        if (!res.ok) {
          throw new Error(`HTTP Error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log("Response Data:", data);
        if (data.error) {
          //  seterrMsg(data.error);
        } else {
          alert('Applied Successfully');
          setAppliedJobs((prev) => [...prev, jobId]);
          setApplyMessage("Applied");
          setTimeout(() => {
            setApplyMessage("");
          }, 3000);
        }
      })
      .catch((error) => {
        console.error('Network Error:', error);
      });
  };

  const handleHomeNavigation = () => {
    setShowConfirmModal(true);
  };

  const confirmNavigation = () => {
    setShowConfirmModal(false);
    navigate('/applicanthome');
  };

  const cancelNavigation = () => {
    setShowConfirmModal(false);
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#f0f8ff',
      padding: '20px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'flex-end',
        paddingRight: '20px',
        marginBottom: '30px',
        marginTop: '0px'
      }}>
        <button
          onClick={handleHomeNavigation}
          style={{
            padding: '8px 16px',
            backgroundColor: '#0476D0',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Home
        </button>
      </div>

      {/* Confirmation Modal */}
      {showConfirmModal && (
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
          zIndex: 2000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '10px',
            padding: '25px',
            width: '400px',
            maxWidth: '90%',
            boxShadow: '0 4px 20px rgba(0,0,0,0.2)'
          }}>
            <h5 style={{ marginTop: 0, color: '#333' }}>Go back to home?</h5>

            <div style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: '15px',
              marginTop: '20px'
            }}>
              <button
                onClick={cancelNavigation}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#f0f0f0',
                  color: '#333',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                No
              </button>
              <button
                onClick={confirmNavigation}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#0476D0',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                Yes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rest of your existing code remains unchanged */}
      {/* Search Section */}
      <div style={{
        maxWidth: '800px',
        margin: '0 auto 30px',
        display: 'flex',
        gap: '15px',
        flexWrap: 'wrap'
      }}>
        <input
          style={{
            flex: '1',
            minWidth: '250px',
            padding: '12px 15px',
            border: '1px solid #ddd',
            borderRadius: '5px',
            fontSize: '16px'
          }}
          placeholder="📍 Job Location"
          value={joblocation}
          onChange={(e) => setjoblocation(e.target.value)}
        />
        <input
          style={{
            flex: '1',
            minWidth: '250px',
            padding: '12px 15px',
            border: '1px solid #ddd',
            borderRadius: '5px',
            fontSize: '16px'
          }}
          placeholder="🔍 Job Title"
          value={jobtitle}
          onChange={(e) => setjobtitle(e.target.value)}
        />
        <button
          style={{
            padding: '12px 25px',
            backgroundColor: '#0476D0',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '16px',
            minWidth: '120px'
          }}
          onClick={fetchJobs}
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
      {/* Results Section */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0 20px'
      }}>
        {error && (
          <div style={{
            backgroundColor: '#ffebee',
            color: '#d32f2f',
            padding: '15px',
            borderRadius: '5px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}
        {jobs.length === 0 && !loading && !error && (
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            textAlign: 'center',
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
          }}>
            <p style={{ fontSize: '18px', color: '#666' }}>No jobs found. Try different search criteria.</p>
          </div>
        )}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))',
          gap: '20px'
        }}>
          {jobs.map((job) => (
            <div key={job._id} style={{
              backgroundColor: 'white',
              borderRadius: '10px',
              padding: '20px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
              display: 'flex',
              flexDirection: 'column',
              gap: '15px'
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderBottom: '1px solid #eee',
                paddingBottom: '15px'
              }}>
                <h2 style={{ margin: 0, color: '#0476D0' }}>{job.compname}</h2>
                <div style={{
                  backgroundColor: '#e8f4fc',
                  color: '#0476D0',
                  padding: '5px 10px',
                  borderRadius: '5px',
                  fontSize: '14px'
                }}>
                  ₹{job.minsalary} - {job.maxsalary}/mo
                </div>
              </div>
              <div style={{
                display: 'flex',
                gap: '20px'
              }}>
                <div style={{ flex: 1 }}>
                  <p style={{ margin: '5px 0' }}><b>Title:</b> {job.jobtitle}</p>
                  <p style={{ margin: '5px 0' }}><b>Location:</b> {job.joblocation}</p>
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ margin: '5px 0' }}><b>Experience:</b> {job.minexp} - {job.maxexp} yrs</p>
                  <p style={{ margin: '5px 0' }}><b>Openings:</b> {job.openings}</p>
                </div>
              </div>
              <div style={{
                display: 'flex',
                gap: '10px',
                marginTop: '10px'
              }}>
                <button
                  onClick={() => handleApplyJob(job._id)}
                  style={{
                    flex: 1,
                    padding: '8px',
                    backgroundColor: appliedJobs.includes(job._id) ? '#cccccc' : '#0476D0',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: appliedJobs.includes(job._id) ? 'not-allowed' : 'pointer'
                  }}
                  disabled={appliedJobs.includes(job._id)}
                >
                  {appliedJobs.includes(job._id) ? "Applied" : "Apply"}
                </button>
                <button
                  onClick={() => setseedetails(job)}
                  style={{
                    flex: 1,
                    padding: '8px',
                    backgroundColor: 'white',
                    color: '#0476D0',
                    border: '1px solid #0476D0',
                    borderRadius: '5px',
                    cursor: 'pointer'
                  }}
                >
                  Details
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
      {/* Job Details Modal */}
      {seedetails && (
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
            width: '500px',
            maxWidth: '90%',
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 4px 20px rgba(0,0,0,0.2)'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px'
            }}>
              <h2 style={{ margin: 0, color: '#0476D0' }}>{seedetails.compname}</h2>
              <button
                onClick={() => setseedetails(null)}
                style={{
                  backgroundColor: 'transparent',
                  border: 'none',
                  fontSize: '20px',
                  cursor: 'pointer',
                  color: '#666'
                }}
              >
                ×
              </button>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <h3 style={{ margin: '0 0 10px', color: '#333' }}>{seedetails.jobtitle}</h3>
              <p style={{ margin: '5px 0', color: '#666' }}>{seedetails.joblocation}</p>
            </div>
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ margin: '0 0 10px', color: '#333', borderBottom: '1px solid #eee', paddingBottom: '5px' }}>
                Job Description
              </h4>
              <p style={{ margin: 0, lineHeight: '1.6' }}>{seedetails.jobdes}</p>
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '15px',
              marginBottom: '20px'
            }}>
              <div>
                <p style={{ margin: '5px 0', fontWeight: 'bold' }}>Experience</p>
                <p style={{ margin: '5px 0' }}>{seedetails.minexp} - {seedetails.maxexp} years</p>
              </div>
              <div>
                <p style={{ margin: '5px 0', fontWeight: 'bold' }}>Salary</p>
                <p style={{ margin: '5px 0' }}>₹{seedetails.minsalary} - ₹{seedetails.maxsalary}</p>
              </div>
              <div>
                <p style={{ margin: '5px 0', fontWeight: 'bold' }}>Openings</p>
                <p style={{ margin: '5px 0' }}>{seedetails.openings}</p>
              </div>
            </div>
            <button
              onClick={() => {
                handleApplyJob(seedetails._id);
                setseedetails(null);
              }}
              style={{
                width: '100%',
                padding: '10px',
                backgroundColor: appliedJobs.includes(seedetails._id) ? '#cccccc' : '#0476D0',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: appliedJobs.includes(seedetails._id) ? 'not-allowed' : 'Apply',
                fontSize: '16px',
                marginTop: '15px',
              }}
              disabled={appliedJobs.includes(seedetails._id)}
            >
              {appliedJobs.includes(seedetails._id) ? "Applied" : "Apply"}
            </button>
          </div>
        </div>
      )}
      {applyMessage && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'transparent',
          color: '#4CAF50',
          padding: '12px 24px',
          borderRadius: '5px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          zIndex: 1000
        }}
        >
          {applyMessage}
        </div>
      )}
    </div>
  );
}