const mongoose = require('mongoose');


const applicantschema = new mongoose.Schema({
    
    jobId:{
        type: String,
        
    },
    EmployeefName:{
        type: String,
        
    },
    EmployeelName:{
        type: String,
        
    },
     EmployeeEmail:{
        type: String,

    },
    EmployeeID:{
        type: String,
        
    },
    EmployeeEducation:{
        type: String,
        
    },
    EmployeeSchoolmed:{
        type: String,
    
    },
    EmployeeCollegename:{
        type: String,
    
    },
    EmployeeSpecialization:{
        type: String,
    
    },
     EmployeeCompletionyear:{
        type: String,
    
    },
     EmployeeSkills:{
        type: String,
    
    },
    EmployeeCV:{
        type: String,
    },
    EmployeeStatus:{
        type: String,
        enum:['Accepted','pending'],
        default: 'pending'
    }
});
const jobpostschema = new mongoose.Schema({

    jobtitle: {
        type: String,
        required: true
    },
    joblocation: {
        type: String,
        required: true
    },
    openings: {
        type: Number,
        required: true,

    },
    minexp: {
        type: Number,
        required: true

    },
    maxexp: {
        type: Number,
        required: true
    },
    minsalary: {
        type: Number,
        required: true
    },
    maxsalary: {
        type: Number,
        required: true,

    },
    jobdes: {
        type: String,
        required: true

    },
    compname: {
        type: String,
        required: true
    },
    contactperson: {
        type: String,
        required: true,
    },
    phonenumber: {
        type: String,
        required: true,

    },
    email: {
        type: String,
        required: true,

    },
    contactpersonprofile: {
        type: String,
        required: true
    },
    EmployerfName: {
        type: String,
        required: true
    },
    EmployerlName: {
        type: String,
        required: true
    },
    EmployerEmail: {
        type: String,
        required: true
    },
    EmployerID: {
        type: String,
        required: true
    },
    applicants: [applicantschema],
}, 
{ timestamps: true });






const JobPost = mongoose.model("JobPost", jobpostschema);
module.exports = JobPost

