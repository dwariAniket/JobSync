const express = require ('express');
const app = express();
const port = 4000;
var cors=require('cors');
const bodyparser = require ('body-parser');
var jsonParser=bodyparser.json();
var urlencodedParser=bodyparser.urlencoded({ extended:false});
app.use(jsonParser);
app.use(urlencodedParser);
app.use(cors())
require('./db');
require('./models/User')
require('./models/JobPost')
require('./models/PostDetails')
require('./models/ApplicantUser')
const authroute1=require('./routers/User_addroutes')
const authroute2=require('./routers/JobPost_addroutes')
const requiredtoken = require ('./middlewires/Authtokenrequired')
app.get('/',requiredtoken,(req,res) => {
    res.send ('This is Index page1');
    console.log(req.user);
})
app.use(authroute1);
app.use(authroute2);
app.listen (port,() => {    
    console.log (`server is running on port ${port}`);
})

