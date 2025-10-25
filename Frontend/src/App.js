import './App.css';
import { Routes, Route, BrowserRouter as Router, Link } from 'react-router-dom';
import Welcome from './components/Welcome';
import EmployerSignup from './components/EmployerSignup';
import ApplicantSignup from './components/ApplicantSignup';
import ApplicantDetails from './components/ApplicantDetails';
import ApplicantHome from './components/ApplicantHome';
import JobDetails from './components/JobDetails';
import ViewApplicant from './components/ViewApplicant';
import EmployerDashboard from './components/EmployerDashboard';
import ApplicantJobSearch from './components/ApplicantJobSearch';

function App() {
  return (
    <>


      <Routes>
        
        <Route path="/" element={<Welcome />} />
        <Route path="/employersignup" element={<EmployerSignup />} />
        <Route path="/applicantsignup" element={<ApplicantSignup />} />
        <Route path="/applicantdetails" element={<ApplicantDetails />} />
        <Route path="/applicanthome" element={<ApplicantHome />} />
        <Route path="/jobdetails" element={<JobDetails />} />
        <Route path="/viewapplicant" element={<ViewApplicant />} />
        <Route path="/applicantjobsearch" element={<ApplicantJobSearch />} />
        <Route path="/employerdashboard" element={<EmployerDashboard />} />
        {/* <Route path="/employerdashboard" element={<EmployerDashboard />} /> */}
      </Routes>
      {/* <ApplicantDetails /> */}

    </>
  );
}
export default App;
