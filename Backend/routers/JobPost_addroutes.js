const express = require('express');
const bcrypt = require('bcrypt');
const router = express.Router();
const mongoose = require("mongoose");
const { spawn } = require("child_process");
const jwt = require('jsonwebtoken');
const JobPost = require("../models/JobPost");
const PostDetails = require("../models/PostDetails");
const ApplicantUser = require("../models/ApplicantUser");
const { json } = require('body-parser');
require('dotenv').config();

const executePython = async (scripts, args) => {
  const argumentsList = args.map(arg => arg.toString());
  const py = spawn("python", [scripts, ...argumentsList]);

  const result = await new Promise((resolve, reject) => {
    let output = "";

    py.stdout.on("data", (data) => {
      output += data.toString(); // collect multiple chunks
    });

    py.stderr.on("data", (data) => {
      console.error(`[python] error occurred: ${data}`);
      reject(`Error occurred in ${scripts}`);
    });

    py.on("exit", (code) => {
      console.log(`child process exited with code ${code}`);
      try {
        resolve(JSON.parse(output));
      } catch (err) {
        resolve(output.trim()); // fallback if not JSON
      }
    });
  });
  return result;
};

router.get('/xyz', async (req, res) => {
  try {
    const result = await executePython('./python/scripts.py', [5, 2]);
    res.json({ result: result });
  } catch (err) {
    res.status(500).json({ err: err });
  }
});






router.post('/signup', async (req, res) => {
  if (!req.body.fname || !req.body.lname || !req.body.email || !req.body.pw || !req.body.workstatus) {
    return res.status(422).send({ error: "All fields are required" });
  }
  try {
    let saveduser = await ApplicantUser.findOne({ "email": req.body.email });
    if (saveduser) {
      return res.status(422).send({ error: "User already exists with Email" });
    }
    /* const user = new User({
         name,
         email,
         password,
         Dob
     });*/

    const user = new ApplicantUser(req.body);
    await user.save();
    const token = jwt.sign({ _id: user._id }, process.env.JWT_SECRET);
    res.status(201).json({ message: "User registered succesfully", token });
  } catch (err) {
    res.status(500).send({ error: "Error occured" });
    console.error(err);
  };

});

router.post('/signin', async (req, res) => {
  const { email, pw } = req.body; // Ensure the correct variable names are used

  if (!email || !pw) {
    return res.status(422).json({ error: "All fields are required" });
  }

  try {
    const savedUser = await ApplicantUser.findOne({ email });
    console.log(savedUser);


    if (!savedUser) {
      return res.status(422).json({ error: "User does not exist" });
    }

    const isMatch = await bcrypt.compare(pw, savedUser.pw); // Ensure pw field matches DB

    if (!isMatch) {
      console.log("pw does not match");
      return res.status(422).json({ error: "Invalid pw" });
    }

    console.log("pw matched");
    const token = jwt.sign({ _id: savedUser._id }, process.env.JWT_SECRET, { expiresIn: "1h" });
    res.send({ savedUser });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


router.post('/job/submit', async (req, res) => {
  console.log("Request Body:", req.body); // Debugging line

  try {
    // Destructure the request body to extract all fields
    const {
      jobtitle,
      joblocation,
      openings,
      minexp,
      maxexp,
      minsalary,
      maxsalary,
      jobdes,
      compname,
      contactperson,
      phonenumber,
      email,
      contactpersonprofile,
      EmployerfName,
      EmployerlName,
      EmployerEmail,
      EmployerID
    } = req.body;

    // Check if all fields are present
    if (!jobtitle || !joblocation || !openings || !minexp || !maxexp || !minsalary || !maxsalary || !jobdes || !compname || !contactperson || !phonenumber || !email || !contactpersonprofile) {
      return res.status(422).send({ error: "All fields are required" });
    }
    const jobpost = new JobPost({
      jobtitle,
      joblocation,
      openings,
      minexp,
      maxexp,
      minsalary,
      maxsalary,
      jobdes,
      compname,
      contactperson,
      phonenumber,
      email,
      contactpersonprofile,
      EmployerfName,
      EmployerlName,
      EmployerEmail,
      EmployerID
    });
    console.log("Saving job post:", jobpost); // Debugging line
    // Save to database
    await jobpost.save();
    const token = jwt.sign({ _id: jobpost._id }, process.env.JWT_SECRET);
    res.status(201).json({ message: "Job posted successfully!", jobpost });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).send({ error: "Something went wrong" });
  }
});


router.post('/details/submit', async (req, res) => {
  console.log("Request Body:", req.body);

  try {

    const {
      schoolmed,
      haveskills,
      skills,
      specialization,
      collegename,
      completionyear,
      pursuingeducation,
      education,
      cvURL,
      EmployeefName,
      EmployeelName,
      EmployeeEmail,
      EmployeeID
    } = req.body;


    const postdetails = new PostDetails({
      schoolmed,
      haveskills,
      skills,
      specialization,
      collegename,
      completionyear,
      pursuingeducation,
      education,
      cvURL,
      EmployeefName,
      EmployeelName,
      EmployeeEmail,
      EmployeeID
    });
    const employeeemail = await PostDetails.findOne({ EmployeeEmail: EmployeeEmail });
    if (employeeemail) {
      try {

        const updatedEmployee = await PostDetails.findOneAndUpdate(
          { EmployeeEmail },
          { $set: { education, schoolmed, collegename, specialization, completionyear, skills, cvURL } },
          { new: true, upsert: true }
        );

        res.status(200).json({ message: 'Employee details updated successfully', updatedEmployee });

      } catch (error) {
        console.error('Error updating employee details:', error);
        res.status(500).json({ message: 'Internal Server Error' });
      }
    }
    else {

      console.log("Saving details:", postdetails);
      await postdetails.save();
      const token = jwt.sign({ _id: postdetails._id }, process.env.JWT_SECRET);
      res.send({ postdetails });

    }
  } catch (err) {
    console.error("Error:", err);
    res.status(500).send({ error: "Something went wrong" });
  }

});



router.get('/details/:employeeID', async (req, res) => {
  try {
    const employeeID = req.params.employeeID;
    const employeepost = await PostDetails.findOne({ EmployeeID: employeeID });

    if (!employeepost) {
      return res.status(404).json({ success: false, message: "Details not found" });
    }

    res.json({ success: true, employeepost });
  } catch (error) {
    console.error("Error fetching details:", error);
    res.status(500).json({ success: false, error: "Internal Server Error" });
  }
});




router.get('/employer/job/:jobId/applicants', async (req, res) => {
  const { jobId } = req.params;

  try {
    // Validate Job ID format
    if (!mongoose.Types.ObjectId.isValid(jobId)) {
      return res.status(400).json({ error: 'Invalid Job ID' });
    }

    // Find the job by ID
    const job = await JobPost.findById(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Return the list of applicants
    res.status(200).json({ applicants: job.applicants });
  } catch (error) {
    console.error('Error fetching applicants:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }

});

let applicants = {};
router.get('/employee/job-search', async (req, res) => {
  const { joblocation, jobtitle } = req.query;

  try {
    // Build the search query
    const searchQuery = {};
    if (joblocation) {
      searchQuery.joblocation = { $regex: joblocation, $options: 'i' }; // Case-insensitive search
    }
    if (jobtitle) {
      searchQuery.jobtitle = { $regex: jobtitle, $options: 'i' };
    }

    if (!joblocation) {
      return res.status(400).json({ error: 'Set Joblocation' });
    }
    if (!jobtitle) {
      return res.status(400).json({ error: 'Set jobtitle' });
    }


    const jobs = await JobPost.find(searchQuery).select('compname jobtitle joblocation minexp maxexp minsalary maxsalary jobdes openings');// Fetch jobs based on criteria
    res.status(200).json({ jobs });
  } catch (error) {
    console.error('Error fetching jobs:', error);
    res.status(500).json({ error: 'Server error. Please try again later.' });
  }
});


// Endpoint to apply for a job
router.post('/employee/apply', async (req, res) => {

  const { jobId, EmployeefName, EmployeelName, EmployeeEmail, EmployeeID, EmployeeEducation, EmployeeSchoolmed, EmployeeCollegename, EmployeeSpecialization, EmployeeCompletionyear, EmployeeSkills, EmployeeCV } = req.body;

  try {
    // Find the job in the database
    const job = await JobPost.findById(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Add the applicant to the job's applicants array
    job.applicants.push({ EmployeefName, EmployeelName, EmployeeEmail, EmployeeID, EmployeeEducation, EmployeeSchoolmed, EmployeeCollegename, EmployeeSpecialization, EmployeeCompletionyear, EmployeeSkills, EmployeeCV });

    await job.save();

    res.status(201).json({ message: 'Application submitted successfully.' });
  } catch (error) {
    console.error('Error applying for job:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});


// Extra---->
router.get('/employee/applicant-details/:userId', async (req, res) => {
  try {
    const { userId } = req.params;

    if (!mongoose.Types.ObjectId.isValid(userId)) {
      return res.status(400).json({ error: "Invalid User ID" });
    }

    const applicant = await PostDetails.findById(userId);

    if (!applicant) {
      return res.status(404).json({ error: "Applicant details not found" });
    }

    res.status(200).json(applicant);
  } catch (error) {
    console.error("Error fetching applicant details:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

router.get('/employee/employeedetails/:email', async (req, res) => {
  const { email } = req.params;

  try {
    if (!email) {
      return res.status(400).json({ error: 'Email parameter is required' });
    }

    console.log("Fetching details for email:", email);

    const details = await PostDetails.find({ EmployeeEmail: email });

    console.log("Details Found:", details);

    if (!details || details.length === 0) {
      return res.status(404).json({ error: "No details found for this email" });
    }

    res.json({ details });
  } catch (error) {
    console.error('Error fetching details:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});


// Endpoint to get applicants for a job
router.get('/employer/job/:jobId/applicants', async (req, res) => {
  const { jobId } = req.params;

  try {

    if (!mongoose.Types.ObjectId.isValid(jobId)) {
      return res.status(400).json({ error: 'Invalid Job ID' });
    }


    const job = await JobPost.findById(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }


    res.status(200).json({ applicants: job.applicants });
  } catch (error) {
    console.error('Error fetching applicants:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }

});


router.get('/employer/jobs', async (req, res) => {
  const { email } = req.query;
  try {
    if (!email) {
      return res.status(400).json({ error: 'Email parameter is required' });
    }

    const jobs = await JobPost.find({ "EmployerEmail": email });

    if (jobs.length === 0) {
      return res.status(404).json({ message: 'No jobs found for this employer.' });
    }

    res.status(200).json({ jobs });
  } catch (error) {
    console.error('Error fetching jobs:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});




router.post('/employer/updateapplicantstatus', async (req, res) => {
  const { jobId, applicantEmail, status } = req.body;

  if (!jobId || !applicantEmail || !status) {
    return res.status(400).json({ error: 'jobId, applicantEmail, and status are required' });
  }

  try {
    const result = await JobPost.updateOne(
      { _id: jobId, "applicants.EmployeeEmail": applicantEmail.trim().toLowerCase() },
      { $set: { "applicants.$.EmployeeStatus": status } }
    );
    console.log('Update Result:', result);

    if (result.matchedCount === 0) {
      return res.status(404).json({ error: 'No matching applicant found for this job ID' });
    }

    res.status(200).json({
      message: `Applicant status updated to "${status}" successfully.`,
      updated: true
    });

  } catch (err) {
    console.error('Update Error:', err);
    res.status(500).json({ error: 'Internal server error: ' + err.message });
  }
});




router.get('/student/applications', async (req, res) => {
  const studentEmail = req.query.email;

  if (!studentEmail) {
    return res.status(400).json({ error: 'Student email is required.' });
  }

  try {

    const jobPostsWithApplicant = await JobPost.find(
      { 'applicants.EmployeeEmail': studentEmail },
      {
        jobtitle: 1,
        joblocation: 1,
        compname: 1,
        applicants: {

          $elemMatch: { EmployeeEmail: studentEmail }
        }
      }
    );


    const studentApplications = jobPostsWithApplicant.map(job => {
      const applicant = job.applicants[0];

      return {
        jobId: job._id,
        jobTitle: job.jobtitle,
        jobLocation: job.joblocation,
        companyName: job.compname,
        EmployeefName: applicant.EmployeefName,
        EmployeelName: applicant.EmployeelName,
        EmployeeEmail: applicant.EmployeeEmail,
        EmployeeEducation: applicant.EmployeeEducation,
        EmployeeSchoolmed: applicant.EmployeeSchoolmed,
        EmployeeCollegename: applicant.EmployeeCollegename,
        EmployeeSpecialization: applicant.EmployeeSpecialization,
        EmployeeCompletionyear: applicant.EmployeeCompletionyear,
        EmployeeSkills: applicant.EmployeeSkills,
        EmployeeCV: applicant.EmployeeCV,
        EmployeeStatus: applicant.EmployeeStatus || 'Pending'
      };
    });

    res.status(200).json({ applications: studentApplications });

  } catch (err) {
    console.error('Server Error fetching student applications:', err);
    res.status(500).json({ error: 'Internal server error: ' + err.message });
  }
});


// fetch employee skills and job description
router.get('/job/:jobId/skills-jobdes', async (req, res) => {
  try {
    const { jobId } = req.params;

    // Fetch job by ID
    const job = await JobPost.findById(jobId).lean();

    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }

    // Extract job description
    const jobDescription = job.jobdes;

    // Extract all applicants' EmployeeSkills
    const employeeSkills = job.applicants.map(app => ({
      skills: app.EmployeeSkills
    }));

    res.json({
      jobId,
      jobDescription,
      employeeSkills
    });

  } catch (err) {
    console.error("Error fetching data:", err);
    res.status(500).json({ message: "Internal Server Error" });
  }
});





// router.get("/job/:jobId/:EmployeeID/skills-jobdes-python", async (req, res) => {
//   try {
//     const { jobId } = req.params;
//     const { EmployeeID } = req.params;
//     const job = await JobPost.findById(jobId).lean();
//   //  const employee= job.applicants[].findById(EmployeeID).lean();
//    // console.log(job)

//     if (!job) {
//       return res.status(404).json({ message: "Job not found" });
//     }
//     const employee = job.applicants.find(app => String(app._id) === String(EmployeeID));
//     const employeeCV = employee.EmployeeCV

//      if (!employee) {
//        return res.status(404).json({ message: "Employee not found" });
//      }
//     //res.json(employee)
//     const jobDescription = job.jobdes;
//     // const employeeCV = employee[employeeCV];
//     if (!employee) {
//       return res.status(404).json({ message: "Employee CV not found" });
//     }
//     const result = await executePython('./python/scripts.py',employeeCV );
//     res.json({ result: result });
//   } catch (err) {
//     res.status(500).json({ err:err});
// }
// });

router.get("/job/:jobId/:EmployeeID/skills-jobdes-python", async (req, res) => {
  try {
    const { jobId, EmployeeID } = req.params;
    const job = await JobPost.findById(jobId).lean();
    const jobdes = job.jobdes;
    console.log(jobdes);
    if (!job) {
      return res.status(404).json({ message: "Job not found" });
    }

    // Find applicant by subdocument _id
    const employee = job.applicants.find(app => String(app._id) === String(EmployeeID));
    if (!employee) {
      return res.status(404).json({ message: "Employee not found" });
    }

    const employeeCV = employee.EmployeeCV;
    if (!employeeCV) {
      return res.status(404).json({ message: "Employee CV not found" });
    }

    const result = await executePython("./python/scripts.py", [employeeCV, jobdes]);
    console.log(result)
    const r = result.similarity_score;
    // res.json({ result, employee, r }); // return both for debugging
    try {
      // send JSON body to the Python API
      const pyResponse = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ similarity: r }),
      });

      if (!pyResponse.ok) {
        throw new Error(`Python API returned ${pyResponse.status}`);
      }

      // get the JSON back
      const data = await pyResponse.json();
      res.json(data);
    } catch (err) {
      console.error(err);
      res.status(500).send(err.message);
    }
  } catch (err) {
    console.error("Error in route:", err);
    res.status(500).json({ error: err.message || err });
  }
});


// router.post('/predict', async (req, res) => {
//   try {
//     // send JSON body to the Python API
//     const pyResponse = await fetch('http://localhost:8000/predict', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ similarity: req.body.similarity }),
//     });

//     if (!pyResponse.ok) {
//       throw new Error(`Python API returned ${pyResponse.status}`);
//     }

//     // get the JSON back
//     const data = await pyResponse.json();
//     res.json(data);
//   } catch (err) {
//     console.error(err);
//     res.status(500).send(err.message);
//   }
// });


router.get("/employee/applied-jobs", async (req, res) => {
  try {
    const { email } = req.query;
    const appliedJobs = await JobPost.find({
      "applicants.EmployeeEmail": email
    });

    const appliedJobIds = appliedJobs.map(job => job._id);
    res.json({ appliedJobs: appliedJobIds });
  } catch (err) {
    res.status(500).json({ err: err.message });
  }
});







module.exports = router;





