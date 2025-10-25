const express = require('express');
const bcrypt = require('bcrypt');
const router = express.Router();
const mongoose = require("mongoose");
const jwt = require('jsonwebtoken');
const User = require("../models/User");
require('dotenv').config();
router.post('/employer/signup', async (req, res) => {

    if (!req.body.fname || !req.body.lname || !req.body.companyname || !req.body.email || !req.body.pw) {
        return res.status(422).send({ error: "All fields are required" });
    }
    try {
        let saveduser = await User.findOne({ "email": req.body.email });
        if (saveduser) {
            return res.status(422).send({ error: "User already exists with Email" });
        }
        /* const user = new User({
             name,
             email,
             pw,
             Dob
         });*/

        const user = new User(req.body);
        await user.save();
        const token = jwt.sign({ _id: user._id }, process.env.JWT_SECRET);
        res.status(201).json({ message: "User registered succesfully", token });
    } catch (err) {
        res.status(500).send({ error: "Error occured" });
        console.error(err);
    };

});

router.post('/employer/signin', async (req, res) => {
    const { email, pw } = req.body; // Ensure the correct variable names are used

    if (!email || !pw) {
        return res.status(422).json({ error: "All fields are required" });
    }

    try {
        const savedUser = await User.findOne({ email });
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

module.exports = router;