const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const userschema = new mongoose.Schema({

    fname: {
        type: String,
        required: true
    },
    lname: {
        type: String,
        required: true
    },
    companyname: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true
    },
    pw: {
        type: String,
        required: true
    }


})
userschema.pre('save', async function (next) {
    const user = this;
    console.log(user);
    if (!user.isModified('pw')) {
        return next();
    }
    user.pw = await bcrypt.hash(user.pw, 8);
    console.log(user);
    next();



})
const User = mongoose.model("User", userschema);
module.exports = User

