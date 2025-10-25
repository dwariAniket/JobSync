import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});



const handleApplyJob = async (jobId) => {
  const userId = localStorage.getItem("userId"); // Get userId from storage
  if (!userId) {
      alert("You must submit your applicant details before applying for a job.");
      return;
  }

  try {
      const response = await fetch(`http://192.168.29.18:5000/employee/applicant-details/${userId}`);
      const data = await response.json();

      if (!response.ok || !data || data.error) {
          alert("You need to complete your profile before applying for jobs.");
          return;
      }

      // Proceed with job application if details exist
      const applyResponse = await fetch(`http://192.168.29.18:5000/jobs/apply/${jobId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ userId }),
      });

      const applyData = await applyResponse.json();
      if (applyResponse.ok) {
          alert("Application submitted successfully!");
      } else {
          alert(applyData.error || "Failed to apply.");
      }
  } catch (error) {
      console.error("Error applying for job:", error);
      alert("Something went wrong. Please try again.");
  }
};


router.post('/jobs/apply/:jobId', async (req, res) => {
  try {
      const { userId } = req.body;
      const { jobId } = req.params;

      if (!mongoose.Types.ObjectId.isValid(userId)) {
          return res.status(400).json({ error: "Invalid User ID" });
      }

      // Check if the user has submitted details
      const applicant = await PostDetails.findById(userId);
      if (!applicant) {
          return res.status(403).json({ error: "You must complete your profile before applying for jobs." });
      }

      // Proceed with job application logic
      res.status(200).json({ message: "Job application successful!" });

  } catch (error) {
      console.error("Error applying for job:", error);
      res.status(500).json({ error: "Internal Server Error" });
  }
});

