const jwt = require('jsonwebtoken');
const User = require("../models/User");
module.exports = (req,res,next) => {
    const { authorization } = req.headers;
    console.log(authorization);
    if(!authorization){
        return res.status(422).send({ error: "User must login"});
    }
    const token = authorization.replace("Bearer ","");
    // console.log(token);
    //next();
    jwt.verify(token,process.env.JWT_SECRET,(err,payload)=>{
        if(err){
            return res.status(422).send({ error: "U must be logged in, Invalid token"});
        }
        const {_id}=payload;
        User.findById(_id).then(UserData=>{
            req.user=UserData;
            console.log(UserData);
            next();
        })
    })
}