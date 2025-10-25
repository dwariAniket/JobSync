const mongoose = require('mongoose');
const postdetailsschema = new mongoose.Schema({
    // _id: { type: mongoose.Schema.Types.ObjectId, auto: true },
    schoolmed: {
        type: String,
    },
    haveskills: {
        type: String,
        required: true
    },
    skills: {
        type: String,
    },
    specialization: {
        type: String,

    },
    collegename: {
        type: String,

    },
    completionyear: {
        type: Number,

    },
    pursuingeducation: {
        type: String,
        required: true,

    },
    education: {
        type: String,
        required: true
    },
    status: {
        type: String,
        enum: ['Pending', 'Accepted', 'Rejected'], 
        default: 'Pending'
    },
    cvURL: {
        type: String,
        required: true,
           validate: {
            validator: function(v) {
                return /^https?:\/\/.+\..+/.test(v);  // Simple URL validation
            },
            message: props => `${props.value} is not a valid URL!`
        }
    },
    EmployeefName: {
        type: String,
        required: true
    },
    EmployeelName: {
        type: String,
        required: true
    },
    EmployeeEmail: {
        type: String,
        required: true
    },
    EmployeeID: {
        type: String,
        required: true
    },
})

const PostDetails = mongoose.model("PostDetails", postdetailsschema);
module.exports = PostDetails

